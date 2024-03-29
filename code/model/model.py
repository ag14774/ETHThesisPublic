from itertools import chain

import model.module as modules
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel, MoEModel
from model.util import (DistributedSampleDistributor, SampleDistributor,
                        conv1x1, cv_squared)
from sklearn.mixture import GaussianMixture


class EmbedPoolMoE(MoEModel):
    def __init__(self,
                 num_experts,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 pregate_out_size,
                 sparse_gradient=True,
                 topk=3,
                 distributed_moe=False,
                 embedding_checkpoint=None,
                 cluster_init=False,
                 final_layer=False):
        super().__init__()

        num_classes = list_num_classes[all_levels.index(selected_level)]

        self.pre_gate = modules.EmbedPoolMoEGate(mlp_dim, vocab_size,
                                                 embedding_dim,
                                                 pregate_out_size,
                                                 sparse_gradient)

        expert_fn = {
            'constructor': EmbedPool,
            'args': {
                'list_num_classes': list_num_classes,
                'all_levels': all_levels,
                'selected_level': selected_level,
                'mlp_dim': mlp_dim,
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'sparse_gradient': sparse_gradient
            }
        }
        if distributed_moe:
            self.moe = DistributedMoE(pregate_out_size,
                                      expert_fn,
                                      num_experts,
                                      topk=topk)
        else:
            self.moe = SimpleMoE(pregate_out_size,
                                 expert_fn,
                                 num_experts,
                                 topk=topk)

        self.final = None
        if final_layer:
            self.final = nn.Linear(num_classes, num_classes)

        self.reset_parameters()

        if embedding_checkpoint:
            checkpoint = torch.load(str(embedding_checkpoint),
                                    map_location=torch.device('cpu'))
            checkpoint = checkpoint['state_dict']
            state_dict = self.pre_gate.state_dict()
            for k in checkpoint:
                for k2 in state_dict:
                    if k == k2 and checkpoint[k].size() == state_dict[k2].size(
                    ):
                        print('Loading: ', k2)
                        state_dict[k2] = checkpoint[k]
            self.pre_gate.load_state_dict(state_dict, strict=False)

            if cluster_init:
                embedding = checkpoint["embedding.weight"]
                embedding = embedding.numpy()
                print("Running EM on embedding matrix...")
                gmm = GaussianMixture(num_experts,
                                      max_iter=500,
                                      random_state=23)
                clusters = gmm.fit_predict(embedding)
                print(list(clusters))
                print("Did EM converge? ", gmm.converged_)
                for i, e in enumerate(self.moe.experts):
                    if e.expert:
                        print(f"Loading expert {i} embedding matrix...")
                        embedding_temp = embedding.copy()
                        embedding_temp[clusters != i] = 0.0
                        num_in_cluster = (clusters == i).sum()
                        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                            torch.as_tensor(gmm.means_[i]),
                            precision_matrix=torch.as_tensor(
                                gmm.precisions_[i]))
                        embedding_temp[clusters == i] = distribution.rsample(
                            [num_in_cluster])
                        expert_state = e.expert.state_dict()
                        expert_state['embedding.weight'] = torch.as_tensor(
                            embedding_temp)
                        e.expert.load_state_dict(expert_state, strict=False)

        # for p in self.named_parameters():
        #     print(p)
        # check_parameter_number(self, True, True)

    def use_strategy(self, strategy):
        other_strategy = strategy['other']
        self.pre_gate = other_strategy(self.pre_gate)
        self.moe.use_strategy(strategy)
        if self.final:
            self.final = other_strategy(self.final)
        return self

    def reset_parameters(self):
        self.pre_gate.reset_parameters()
        for e in self.moe.experts:
            if e.expert:
                e.expert.reset_parameters()
        self.moe.noisy_gate.reset_parameters()

        if self.final:
            nn.init.xavier_uniform_(self.final.weight)
            nn.init.zeros_(self.final.bias)

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            return super().parameters(recurse, return_groups, requires_grad)
        else:
            res = {'embeddings': [], 'rest': [], 'gate': None}
            for i, e in enumerate(self.moe.experts):
                if e.expert:
                    temp = e.expert.parameters(recurse, return_groups,
                                               requires_grad)
                    for k, v in temp.items():
                        if 'embedding' in k:
                            res['embeddings'].append(v)
                        else:
                            res['rest'].append(v)

            res['embeddings'].append(
                self.pre_gate.embedding.parameters(recurse=recurse))
            res['rest'].append(self.pre_gate.fc1.parameters(recurse=recurse))
            res['rest'].append(self.pre_gate.fc2.parameters(recurse=recurse))
            res['rest'].append(self.pre_gate.fc3.parameters(recurse=recurse))
            if self.final:
                res['rest'].append(self.final.parameters(recurse=recurse))

            res['embeddings'] = chain(*res['embeddings'])
            res['rest'] = chain(*res['rest'])

            res['gate'] = self.moe.noisy_gate.parameters(recurse=recurse)

            if requires_grad:
                for k, v in res.items():
                    res[k] = filter(lambda p: p.requires_grad, v)

            return res

    def forward(self, x):
        original_x = x
        x = self.pre_gate(x)

        (x, l_importance, l_load) = self.moe(x, expert_x=original_x)

        if self.final:
            x = F.relu(x, inplace=True)
            x = self.final(x)

        return x, l_importance, l_load


class SimpleMoE(MoEModel):
    def __init__(self, in_dim, expert_fn, num_experts, topk=3):
        super(SimpleMoE, self).__init__()
        self.in_dim = in_dim
        self.expert_fn_args = expert_fn['args']
        self.expert_constructor = expert_fn['constructor']
        self.num_experts = num_experts
        self.topk = topk

        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            expert = modules.Expert(
                self.expert_constructor(**self.expert_fn_args), 'input')
            self.experts.append(expert)

        self.noisy_gate = modules.NoisyTopK(self.in_dim,
                                            self.num_experts,
                                            k=self.topk,
                                            noisy_gating=True,
                                            noise_epsilon=1e-2)

        self.reset_parameters()

    def use_strategy(self, strategy):
        expert_strategy = strategy['experts']
        gate_strategy = strategy['gate']
        self.noisy_gate = gate_strategy(self.noisy_gate)
        for i, e in enumerate(self.experts):
            expert = expert_strategy(i, e)
            self.experts[i] = expert
        return self

    def reset_parameters(self):
        def weight_init(m):
            pass

        self.apply(weight_init)
        self.noisy_gate.reset_parameters()

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            return super().parameters(recurse, return_groups, requires_grad)
        else:
            res = {
                'gate': self.noisy_gate.parameters(recurse=recurse),
                'experts': self.experts.parameters(recurse=recurse)
            }
            if requires_grad:
                for k, v in res.items():
                    res[k] = filter(lambda p: p.requires_grad, v)

            return res

    def forward(self, x, expert_x=None):
        x_flat = x.view(x.size(0), -1)

        gates, load = self.noisy_gate(x_flat)

        importance = torch.sum(gates, 0)
        l_importance = cv_squared(importance)

        load = load.view(-1, gates.size(1))
        load = torch.sum(load, dim=0)
        if not self.training:
            print(load, importance)
        l_load = cv_squared(load)

        # dispatch to experts (x)
        distributor = SampleDistributor(gates, self.num_experts)
        if expert_x is None:
            expert_x = x
        expert_inputs = distributor.split_input(expert_x)

        expert_outs = [e(i) for i, e in zip(expert_inputs, self.experts)]

        # combine output
        y = distributor.combine(expert_outs)

        return y, l_importance, l_load


class DistributedMoE(MoEModel):
    def __init__(self,
                 in_dim,
                 expert_fn,
                 num_experts,
                 topk=3,
                 group=dist.group.WORLD):
        super(DistributedMoE, self).__init__()
        self.in_dim = in_dim
        self.expert_fn_args = expert_fn['args']
        self.expert_constructor = expert_fn['constructor']
        self.num_experts = num_experts
        self.topk = topk

        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            expert = modules.RemoteExpert(
                self.expert_constructor(**self.expert_fn_args), 'input')
            self.experts.append(expert)

        self.noisy_gate = modules.NoisyTopK(self.in_dim,
                                            self.num_experts,
                                            k=self.topk,
                                            noisy_gating=True,
                                            noise_epsilon=1e-2)

        self.reset_parameters()

    def use_strategy(self, strategy):
        expert_strategy = strategy['experts']
        gate_strategy = strategy['gate']
        self.noisy_gate = gate_strategy(self.noisy_gate)
        for i, e in enumerate(self.experts):
            expert = expert_strategy(i, e)
            self.experts[i] = expert
        return self

    def reset_parameters(self):
        def weight_init(m):
            pass

        self.apply(weight_init)
        self.noisy_gate.reset_parameters()

    def forward(self, x, expert_x=None):
        x_flat = x.view(x.size(0), -1)

        gates, load = self.noisy_gate(x_flat)

        importance = torch.sum(gates, 0)
        l_importance = cv_squared(importance)

        load = load.view(-1, gates.size(1))
        load = torch.sum(load, dim=0)
        l_load = cv_squared(load)

        # dispatch to experts (x)
        distributor = DistributedSampleDistributor(gates,
                                                   self.experts,
                                                   training=self.training)
        if expert_x is None:
            expert_x = x
        expert_inputs = distributor.split_input(expert_x)

        expert_outs = [e(i) for i, e in zip(expert_inputs, self.experts)]

        # combine output
        y = distributor.combine(expert_outs)

        return y, l_importance, l_load


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class GeNetModel(BaseModel):
    def __init__(self,
                 list_num_classes,
                 rmax=10000,
                 kernel_h=3,
                 num_filters=128,
                 resnet_out=1024):
        super(GeNetModel, self).__init__()
        self.kernel_h = kernel_h

        self.preresnet = modules.GeNetPreResnet(rmax, kernel_h, num_filters)
        self.gresnet = modules.GeNetResnet(kernel_h, num_filters, resnet_out)
        self.logits = modules.GeNetLogitLayer2(resnet_out, list_num_classes)

        self.reset_parameters()
        # for p in self.parameters():
        #     print(p)

    def reset_parameters(self):
        def init_genet(m):
            try:
                if isinstance(m, nn.Embedding):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        self.apply(init_genet)

    def forward(self, x):
        x = self.preresnet(x)

        x = self.gresnet(x)

        x = self.logits(x)
        return x


class GeNetModelDeepSet(BaseModel):
    def __init__(self,
                 list_num_classes,
                 rmax=10000,
                 kernel_h=3,
                 num_filters=128,
                 resnet_out=1024,
                 deepset_hidden=2048,
                 deepset_out=1024,
                 extra_phi_layer=True,
                 deepset_activation='relu',
                 deepset_dropout=0.5,
                 reset_weights=False,
                 logit_layer_type='type2',
                 skip='none',
                 bn_running_stats=True,
                 log_output=False,
                 resnet_checkpoint=None):
        super(GeNetModelDeepSet, self).__init__()
        self.kernel_h = kernel_h

        self.preresnet = modules.GeNetPreResnet(rmax, kernel_h, num_filters)
        self.gresnet = modules.GeNetResnet(kernel_h,
                                           num_filters,
                                           resnet_out,
                                           bn_running_stats=bn_running_stats)

        assert skip in ['none', 'connection', 'completely']
        self.skip = skip

        # \rho(\sum(\phi(x)))
        if self.skip == 'completely':
            assert extra_phi_layer == False
            self.deepset = modules.DeepSet(resnet_out,
                                           deepset_hidden,
                                           deepset_out,
                                           activation=deepset_activation,
                                           dropout=deepset_dropout,
                                           average=True,
                                           rho=False)
            logit_input_size = resnet_out
        else:
            if extra_phi_layer:
                phi = nn.Sequential(nn.Linear(resnet_out, resnet_out),
                                    nn.ReLU(inplace=True))
            else:
                phi = None
            self.deepset = modules.DeepSet(resnet_out,
                                           deepset_hidden,
                                           deepset_out,
                                           activation=deepset_activation,
                                           dropout=deepset_dropout,
                                           phi=phi,
                                           average=True)
            logit_input_size = deepset_out
            if self.skip == 'connection':
                assert logit_input_size == resnet_out

        assert logit_layer_type in ['type1', 'type2']

        if logit_layer_type == 'type1':
            self.logits = modules.GeNetLogitLayer(logit_input_size,
                                                  list_num_classes)
        elif logit_layer_type == 'type2':
            self.logits = modules.GeNetLogitLayer2(logit_input_size,
                                                   list_num_classes)

        if log_output:
            self.prob = nn.LogSoftmax(dim=1)
        else:
            self.prob = nn.Softmax(dim=1)

        if reset_weights:
            self.reset_parameters()
        # for p in self.parameters():
        #     print(p)

        if resnet_checkpoint:
            checkpoint = torch.load(str(resnet_checkpoint),
                                    map_location=torch.device('cpu'))
            checkpoint = checkpoint['state_dict']
            state_dict = self.state_dict()
            for k in checkpoint:
                if k.startswith('preresnet') or k.startswith('gresnet'):
                    state_dict[k] = checkpoint[k]
            self.load_state_dict(state_dict, strict=False)
            self.preresnet.requires_grad_(False)
            self.gresnet.requires_grad_(False)

    def reset_parameters(self):
        def init_genet(m):
            try:
                if isinstance(m, nn.Embedding):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        self.apply(init_genet)

    def forward(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.preresnet(x)

        x = self.gresnet(x)
        resout = x  # [batch*bag, rest]

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.deepset(x)  # [batch_size, rest]

        if self.skip == 'connection':
            resout = resout.view(batch_size, bag_size, *resout.size()[1:])
            resout = torch.mean(resout, dim=1)
            x = x + resout

        x = self.logits(x)
        # print(x)
        x = [self.prob(j) for j in x]
        # print(x)
        return x


class GeNetModelMILAttention(BaseModel):
    def __init__(self,
                 list_num_classes,
                 rmax=10000,
                 kernel_h=3,
                 num_filters=128,
                 resnet_out=1024,
                 reset_weights=False,
                 logit_layer_type='type2',
                 bn_running_stats=True,
                 pool_hidden=128,
                 pool_n_attentions=30,
                 pool_gate=False,
                 resnet_checkpoint=None):
        super(GeNetModelMILAttention, self).__init__()
        self.kernel_h = kernel_h

        self.preresnet = modules.GeNetPreResnet(rmax, kernel_h, num_filters)
        self.gresnet = modules.GeNetResnet(kernel_h,
                                           num_filters,
                                           resnet_out,
                                           bn_running_stats=bn_running_stats)

        self.pool = modules.MILAttentionPool(resnet_out,
                                             pool_hidden,
                                             pool_n_attentions,
                                             gated=pool_gate)
        logit_input_size = resnet_out * pool_n_attentions

        self.logits = modules.GeNetLogitLayer2(logit_input_size,
                                               list_num_classes)

        self.prob = nn.LogSoftmax(dim=1)

        if reset_weights:
            self.reset_parameters()
        # for p in self.parameters():
        #     print(p)

        if resnet_checkpoint:
            checkpoint = torch.load(str(resnet_checkpoint),
                                    map_location=torch.device('cpu'))
            checkpoint = checkpoint['state_dict']
            state_dict = self.state_dict()
            for k in checkpoint:
                if k.startswith('preresnet') or k.startswith('gresnet'):
                    state_dict[k] = checkpoint[k]
            self.load_state_dict(state_dict, strict=False)
            self.preresnet.requires_grad_(False)
            self.gresnet.requires_grad_(False)

    def reset_parameters(self):
        def init_genet(m):
            try:
                if isinstance(m, nn.Embedding):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif isinstance(m, nn.Conv2d):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        self.apply(init_genet)

    def forward(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.preresnet(x)

        x = self.gresnet(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.pool(x)  # [batch_size, rest]

        x = self.logits(x)
        # print(x)
        x = [self.prob(j) for j in x]
        # print(x)
        return x


class DeepMicrobes(BaseModel):
    def __init__(self,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 lstm_dim,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 row,
                 da,
                 keep_prob,
                 sparse_gradient=True):
        super().__init__()
        self.num_classes = list_num_classes[all_levels.index(selected_level)]
        self.lstm_dim = lstm_dim
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.row = row
        self.da = da
        self.keep_prob = keep_prob

        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      sparse=sparse_gradient)
        self.bilstm = modules.BidirectionalLSTM(self.embedding_dim,
                                                self.lstm_dim)
        self.attention = modules.AttentionLayer(2 * self.lstm_dim, self.da,
                                                self.row)
        self.fc1 = nn.Linear(self.row * 2 * self.lstm_dim, self.mlp_dim)
        self.drop1 = nn.Dropout(p=1 - self.keep_prob)

        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.drop2 = nn.Dropout(p=1 - self.keep_prob)

        self.fc3 = nn.Linear(self.mlp_dim, self.num_classes)

        self.reset_parameters()
        # check_parameter_number(self, True, True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        self.bilstm.reset_parameters()
        self.attention.reset_parameters()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            return super().parameters(recurse, return_groups, requires_grad)
        else:
            res = {
                'embedding':
                self.embedding.parameters(recurse=recurse),
                'bilstm':
                self.bilstm.parameters(recurse=recurse),
                'attention':
                self.attention.parameters(recurse=recurse),
                'fc':
                chain(self.fc1.parameters(recurse=recurse),
                      self.drop1.parameters(recurse=recurse),
                      self.fc2.parameters(recurse=recurse),
                      self.drop2.parameters(recurse=recurse),
                      self.fc3.parameters(recurse=recurse))
            }
            if requires_grad:
                for k, v in res.items():
                    res[k] = filter(lambda p: p.requires_grad, v)

            return res

    def forward(self, x):
        # print(x)
        x = self.embedding(x)
        x = self.bilstm(x)
        x = self.attention(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        x = self.drop2(x)

        logits = self.fc3(x)
        return logits


class EmbedPool(BaseModel):
    def __init__(self,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 sparse_gradient=True):
        super().__init__()
        self.num_classes = list_num_classes[all_levels.index(selected_level)]
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      sparse=sparse_gradient)

        self.fc1 = nn.Linear(2 * self.embedding_dim, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.fc3 = nn.Linear(self.mlp_dim, self.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            return super().parameters(recurse, return_groups, requires_grad)
        else:
            res = {
                'embedding': self.embedding.parameters(recurse=recurse),
                'fc1': self.fc1.parameters(recurse=recurse),
                'fc2': self.fc2.parameters(recurse=recurse),
                'fc3': self.fc3.parameters(recurse=recurse)
            }
            if requires_grad:
                for k, v in res.items():
                    res[k] = filter(lambda p: p.requires_grad, v)

            return res

    def forward(self, x):
        x = self.embedding(x)

        x_max, _ = x.max(dim=1)
        x_mean = x.mean(dim=1)

        x = torch.cat((x_mean, x_max), dim=1)

        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)

        logits = self.fc3(x)
        return logits


class EmbedPoolDS(EmbedPool):
    def __init__(self,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 sparse_gradient=True,
                 ds_dropout=0.0,
                 ds_position='after_embedding'):
        super().__init__(list_num_classes, all_levels, selected_level, mlp_dim,
                         vocab_size, embedding_dim, sparse_gradient)
        self.deepset = modules.DeepSet(2 * self.embedding_dim,
                                       0,
                                       0,
                                       phi=None,
                                       activation='relu',
                                       dropout=ds_dropout,
                                       average=True,
                                       rho=False)
        self.prob = nn.LogSoftmax(dim=1)

        assert ds_position in [
            'after_embedding', 'before_logits', 'after_logits'
        ]
        ds_position_dict = {
            'after_embedding': self.forward_ae,
            'before_logits': self.forward_bl,
            'after_logits': self.forward_al
        }
        self.forward = ds_position_dict[ds_position]

    def embedding_forward(self, x):
        x = self.embedding(x)

        x_max, _ = x.max(dim=1)
        x_mean = x.mean(dim=1)

        x = torch.cat((x_mean, x_max), dim=1)
        return x

    def fc_forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        return x

    def forward_ae(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.deepset(x)

        x = self.fc_forward(x)
        logits = self.fc3(x)
        return self.prob(logits)

    def forward_bl(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)
        x = self.fc_forward(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.deepset(x)

        logits = self.fc3(x)
        return self.prob(logits)

    def forward_al(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)
        x = self.fc_forward(x)
        x = self.fc3(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        logits = self.deepset(x)

        return self.prob(logits)


class EmbedPoolMILAttention(EmbedPool):
    def __init__(self,
                 list_num_classes,
                 all_levels,
                 selected_level,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 sparse_gradient=True,
                 pool_position='after_embedding',
                 pool_hidden=128,
                 pool_n_attentions=30,
                 pool_gate=False,
                 embedding_checkpoint=None):
        super(EmbedPool, self).__init__()
        self.num_classes = list_num_classes[all_levels.index(selected_level)]
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      sparse=sparse_gradient)

        assert pool_position in [
            'after_embedding', 'before_logits', 'after_logits'
        ]

        if pool_position == 'after_embedding':
            self.pool = modules.MILAttentionPool(2 * self.embedding_dim,
                                                 pool_hidden,
                                                 pool_n_attentions,
                                                 gated=pool_gate)
            self.fc1 = nn.Linear(2 * self.embedding_dim * pool_n_attentions,
                                 self.mlp_dim)
            self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
            self.fc3 = nn.Linear(self.mlp_dim, self.num_classes)
        elif pool_position == 'before_logits':
            self.fc1 = nn.Linear(2 * self.embedding_dim, self.mlp_dim)
            self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
            self.pool = modules.MILAttentionPool(self.mlp_dim,
                                                 pool_hidden,
                                                 pool_n_attentions,
                                                 gated=pool_gate)
            self.fc3 = nn.Linear(self.mlp_dim * pool_n_attentions,
                                 self.num_classes)
        elif pool_position == 'after_logits':
            assert pool_n_attentions == 1
            self.fc1 = nn.Linear(2 * self.embedding_dim, self.mlp_dim)
            self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
            self.fc3 = nn.Linear(self.mlp_dim, self.num_classes)
            self.pool = modules.MILAttentionPool(self.num_classes,
                                                 pool_hidden,
                                                 pool_n_attentions,
                                                 gated=pool_gate)

        self.prob = nn.LogSoftmax(dim=1)
        self.reset_parameters()

        pool_position_dict = {
            'after_embedding': self.forward_ae,
            'before_logits': self.forward_bl,
            'after_logits': self.forward_al
        }
        self.forward = pool_position_dict[pool_position]
        if embedding_checkpoint:
            checkpoint = torch.load(str(embedding_checkpoint),
                                    map_location=torch.device('cpu'))
            checkpoint = checkpoint['state_dict']
            state_dict = self.state_dict()
            for k in checkpoint:
                if k.startswith('embedding'):
                    print('Loading embedding...')
                    state_dict[k] = checkpoint[k]
            self.load_state_dict(state_dict, strict=False)
            self.embedding.requires_grad_(False)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        nn.init.xavier_uniform_(self.pool.fc1.weight)
        nn.init.zeros_(self.pool.fc1.bias)

        nn.init.xavier_uniform_(self.pool.fc2.weight)
        nn.init.zeros_(self.pool.fc2.bias)

        if self.pool.fc3:
            nn.init.xavier_uniform_(self.pool.fc3.weight)
            nn.init.zeros_(self.pool.fc3.bias)

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            return super().parameters(recurse, return_groups, requires_grad)
        else:
            res = {
                'embedding': self.embedding.parameters(recurse=recurse),
                'pool': self.pool.parameters(recurse=recurse),
                'fc1': self.fc1.parameters(recurse=recurse),
                'fc2': self.fc2.parameters(recurse=recurse),
                'fc3': self.fc3.parameters(recurse=recurse)
            }
            if requires_grad:
                for k, v in res.items():
                    res[k] = filter(lambda p: p.requires_grad, v)

            return res

    def embedding_forward(self, x):
        x = self.embedding(x)

        x_max, _ = x.max(dim=1)
        x_mean = x.mean(dim=1)

        x = torch.cat((x_mean, x_max), dim=1)
        return x

    def fc_forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)

        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        return x

    def forward_ae(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.pool(x)

        x = self.fc_forward(x)
        logits = self.fc3(x)
        return self.prob(logits)

    def forward_bl(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)
        x = self.fc_forward(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        x = self.pool(x)

        logits = self.fc3(x)
        return self.prob(logits)

    def forward_al(self, x):
        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])

        x = self.embedding_forward(x)
        x = self.fc_forward(x)
        x = self.fc3(x)

        x = x.view(batch_size, bag_size, *x.size()[1:])
        logits = self.pool(x)

        return self.prob(logits)


class ResNet(BaseModel):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, modules.ResNetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, modules.ResNetBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

from functools import reduce

import model.util as model_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import get_global_rank


class Expert(nn.Module):
    def __init__(self, expert, output_device='default'):
        '''
        Note on output_device:
        if set to 'default' then the output of the expert is untouched, which
        means that in simple cases where the whole expert is on a single device,
        it should be equal to self.device

        if set to 'input', then the output is sent back to the same device as
        the input
        '''
        super(Expert, self).__init__()
        self.expert = expert
        self.device = torch.device('cpu')
        if output_device not in ['default', 'input']:
            output_device = torch.device(output_device)
        self.output_device = output_device

    def set_expert_device(self, device):
        self.device = torch.device(device)
        return self

    def _get_output_device(self, input):
        if self.output_device == 'default':
            return None
        elif self.output_device == 'input':
            return model_utils.infer_device(input)
        return self.output_device

    def forward(self, x):
        output_device = self._get_output_device(x)

        x = model_utils.all_tensors_to(x, self.device)

        if model_utils.get_a_tensor(x).size(0) > 0:
            x = self.expert(x)
        else:
            x = x.new_tensor([])

        if output_device:
            x = model_utils.all_tensors_to(x, output_device)
        return x


class RemoteExpert(nn.Module):
    def __init__(self, expert, local_output_device='default'):
        '''
        Note on output_device:
        if set to 'default' then the output of the expert is untouched, which
        means that in simple cases where the whole expert is on a single device,
        it should be equal to self.device

        if set to 'input', then the output is sent back to the same device as
        the input
        '''
        super(RemoteExpert, self).__init__()
        self.expert = expert
        self.device = torch.device('cpu')
        self.rank = get_global_rank()
        if local_output_device not in ['default', 'input']:
            local_output_device = torch.device(local_output_device)
        self.local_output_device = local_output_device

    def set_expert_device(self, assigned_rank, device):
        self.device = torch.device(device)
        self.rank = assigned_rank
        if self.rank != get_global_rank():
            self.expert = None
        return self

    def _get_output_device(self, input):
        if self.local_output_device == 'default':
            return None
        elif self.local_output_device == 'input':
            return model_utils.infer_device(input)
        return self.local_output_device

    def forward(self, x):
        output_device = self._get_output_device(x)

        returnEmpty = True
        if self.expert:
            x = model_utils.all_tensors_to(x, self.device)

            if model_utils.get_a_tensor(x).size(0) > 0:
                x = self.expert(x)
                returnEmpty = False

        if returnEmpty:
            x = x.new_tensor([])

        if output_device:
            x = model_utils.all_tensors_to(x, output_device)

        return x


class NoisyTopK(nn.Module):
    def __init__(self,
                 in_dim,
                 num_experts,
                 k=2,
                 noisy_gating=True,
                 noise_epsilon=1e-2):
        super(NoisyTopK, self).__init__()
        self.in_dim = in_dim
        if isinstance(self.in_dim, list):
            self.in_size = int(reduce((lambda x, y: x * y), self.in_dim))
        else:
            self.in_size = self.in_dim
        self.num_experts = num_experts
        self.k = k
        self.noise_epsilon = noise_epsilon

        self.W_g = nn.Linear(self.in_size, self.num_experts, bias=False)
        self.W_noise = None
        if noisy_gating:
            self.W_noise = nn.Linear(self.in_size,
                                     self.num_experts,
                                     bias=False)

    def reset_parameters(self):
        nn.init.zeros_(self.W_g.weight)
        nn.init.zeros_(self.W_noise.weight)

    def _gates_to_load(self, gates):
        """Calculates true load"""
        return torch.sum((gates > 0).float(), dim=0)

    def _prob_in_top_k(self, clean_logits, noisy_logits, noise_stddev,
                       top_logits):

        # threshold_if_out = top_logits[:, self.k - 1]
        # threshold_if_in = top_logits[:, self.k]

        thresholds = top_logits[:, self.k - 1:self.k + 1]
        is_in = noisy_logits.gt(thresholds[:, 1:]).long()
        kthresh = torch.gather(thresholds, 1, is_in)
        return model_utils.normal_cdf(clean_logits - kthresh, noise_stddev)

    def _prob_in_top_k_2(self, clean_logits, noisy_logits, noise_stddev,
                         top_logits):

        # threshold_if_out = top_logits[:, self.k - 1]
        # threshold_if_in = top_logits[:, self.k]
        batch = clean_logits.size(0)
        m = top_logits.size(1)
        top_logits_flat = top_logits.view([-1])

        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_logits_flat, 0, threshold_positions_if_in), 1)
        is_in = noisy_logits.gt(threshold_if_in).long()
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_logits_flat, 0, threshold_positions_if_out), 1)
        prob_if_in = model_utils.normal_cdf(clean_logits - threshold_if_in,
                                            noise_stddev)
        prob_if_out = model_utils.normal_cdf(clean_logits - threshold_if_out,
                                             noise_stddev)
        prob = torch.where(is_in.byte(), prob_if_in, prob_if_out)
        return prob

    def forward(self, x):
        x = x.float()
        clean_logits = self.W_g(x)
        gates = clean_logits.new_zeros(clean_logits.size())
        if self.W_noise:
            noise_stddev = self.W_noise(x)
            noise_stddev = F.softplus(noise_stddev) + self.noise_epsilon
            noise_stddev = noise_stddev * self.training
            noisy_logits = clean_logits + torch.normal(gates) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
        # We take the k + 1 top values because the k+1_th value is used for
        # calculating the load of the experts (eq. 9 in paper)
        top_logits, top_indices = torch.topk(logits,
                                             min(self.k + 1, self.num_experts),
                                             dim=-1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        eps = 0.01
        # Top K gates should have non-zero weights
        top_k_gates = top_k_gates + eps
        top_k_gates = top_k_gates / torch.sum(top_k_gates, dim=-1).view(-1, 1)

        gates.scatter_(1, top_k_indices, top_k_gates)

        # k strictly less than num_experts so there is at least 1 expert
        # that is not selected.
        if self.W_noise and self.k < self.num_experts:
            probs = self._prob_in_top_k(clean_logits, noisy_logits,
                                        noise_stddev, top_logits)
            load = torch.sum(probs, dim=0)
        else:
            load = self._gates_to_load(gates)

        return gates, load


class PositionalEmbedding(nn.Module):
    def __init__(self, voc_size, seq_size, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(voc_size, embedding_dim)
        self.pos_embedding = nn.Embedding(seq_size, embedding_dim)
        self.register_buffer("positions", torch.arange(seq_size).long())

    def forward(self, x):
        x = self.embedding(x)
        p = self.pos_embedding(self.positions)
        x = x + p
        return x


class OneHot(nn.Module):
    def __init__(self, voc_size=6):
        """
        Converts array of size B x C x W to
        one-hot array of size B x C x W x V
        """
        super(OneHot, self).__init__()
        self.voc_size = voc_size

    def forward(self, x):
        # Expects shape (batch_size, C, W)
        shape = list(x.size())
        shape.append(self.voc_size)
        out = torch.zeros(shape, device=x.device)
        x = torch.unsqueeze(x, -1)
        out.scatter_(3, x, 1)
        return out


class Linear2(nn.Module):
    def __init__(self,
                 in_features1,
                 out_features,
                 in_features2=None,
                 activation=lambda x: x):
        '''
        y = activation(Wx_1) + activation(Ux_2)
        '''
        super(Linear2, self).__init__()
        self.linear1 = nn.Linear(in_features1, out_features)
        self.linear2 = None
        if in_features2 is not None:
            self.linear2 = nn.Linear(in_features2, out_features)

        self.activation = activation

    def forward(self, h, y=0):
        h = self.activation(self.linear1(h))
        if self.linear2 is not None:
            y = self.activation(self.linear2(y))

        return h + y


class GeNetPreResnet(nn.Module):
    def __init__(self, rmax=10000, kernel_h=3, num_filters=128):
        super(GeNetPreResnet, self).__init__()
        self.pos_embedding = PositionalEmbedding(6, rmax, 6)
        self.one_hot = OneHot(6)
        self.conv1 = model_utils.conv_h_w(kernel_h,
                                          6,
                                          1,
                                          num_filters,
                                          stride=(kernel_h, 0),
                                          bias=True,
                                          padding='valid')

    def forward(self, x):
        identity = x

        x = self.pos_embedding(x)
        x = x + self.one_hot(identity)

        # ResNet
        x = self.conv1(x)  # out: (N, 128, rmax/3, 1)

        return x


class GeNetResnet(nn.Module):
    def __init__(self,
                 kernel_h=3,
                 num_filters=128,
                 resnet_out=1024,
                 bn_running_stats=True):
        super(GeNetResnet, self).__init__()

        self.block1 = GeNetResnetBlock(num_filters,
                                       num_filters,
                                       kernel_h,
                                       bn_running_stats=bn_running_stats)
        self.block2 = GeNetResnetBlock(num_filters,
                                       num_filters,
                                       kernel_h,
                                       bn_running_stats=bn_running_stats)
        self.block3 = GeNetResnetBlock(num_filters,
                                       2 * num_filters,
                                       kernel_h,
                                       bn_running_stats=bn_running_stats)
        self.block4 = GeNetResnetBlock(2 * num_filters,
                                       2 * num_filters,
                                       kernel_h,
                                       bn_running_stats=bn_running_stats)

        self.bn1 = model_utils.batch_normalization(
            2 * num_filters, track_running_stats=bn_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = model_utils.batch_normalization(
            2 * num_filters, track_running_stats=bn_running_stats)
        self.fc = nn.Linear(2 * num_filters, resnet_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.bn2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)

        return x


class GeNetLogitLayer(nn.Module):
    def __init__(self, in_features, sizes):
        '''
        Args:
            sizes (list): a list of integers representing the number of
            classes in each of the softmax layers.
        '''
        super(GeNetLogitLayer, self).__init__()
        self.layers = nn.ModuleList()
        in_features2 = None
        for i, s in enumerate(sizes):
            self.layers.append(Linear2(in_features, s, in_features2, F.relu))
            in_features2 = s

    def forward(self, x):
        out = []
        y = 0
        for layer in self.layers:
            y = layer(x, y)
            out.append(y)

        return out


class GeNetLogitLayer2(nn.Module):
    def __init__(self, in_features, sizes):
        '''
        Args:
            sizes (list): a list of integers representing the number of
            classes in each of the softmax layers.
        '''
        super(GeNetLogitLayer2, self).__init__()
        self.logits = nn.ModuleList()
        for i, s in enumerate(sizes):
            self.logits.append(nn.Linear(in_features, s))
        self.logits_add = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.logits_add.append(nn.Linear(sizes[i - 1], sizes[i]))

    def forward(self, x):
        logits = [F.relu(fc(x), inplace=True) for fc in self.logits]
        logits_add = [0] + [
            F.relu(fc(logits[i - 1]), inplace=True)
            for i, fc in enumerate(self.logits_add, 1)
        ]

        return [out1 + out2 for (out1, out2) in zip(logits, logits_add)]


class GeNetResnetBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_h=3,
                 input_width=1,
                 bn_running_stats=True):
        super(GeNetResnetBlock, self).__init__()
        self.avgpool = None
        if inplanes != planes:
            self.avgpool = model_utils.avgpool2x1(stride=(2, 1))
        self.bn1 = model_utils.batch_normalization(
            inplanes, track_running_stats=bn_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = model_utils.conv_h_w(kernel_h, input_width, inplanes,
                                          planes, (1, 0))
        self.bn2 = model_utils.batch_normalization(
            planes, track_running_stats=bn_running_stats)
        self.conv2 = model_utils.conv_h_w(kernel_h, 1, planes, planes, (1, 0))
        self.downsample = None
        if inplanes != planes:
            self.downsample = model_utils.conv1x1(inplanes, planes, stride=1)

    def forward(self, x):
        if self.avgpool:
            x = self.avgpool(x)

        identity = x

        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2(x)

        if self.downsample:
            identity = self.downsample(identity)

        x = x + identity
        return x


class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features,
                                 eps=eps,
                                 momentum=momentum,
                                 affine=affine,
                                 track_running_stats=track_running_stats)

    def forward(self, x):
        samples = x.size(0)
        is_training = self.training
        if samples < 2:
            self.bn.eval()

        x = self.bn(x)

        if samples < 2:
            self.bn.train(mode=is_training)

        return x


class DeepSet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer,
                 output_size,
                 phi=None,
                 activation='relu',
                 dropout=0.5,
                 average=False,
                 rho=True):
        super().__init__()
        self.phi = phi
        self.average = average
        if self.average:
            self.pool = torch.mean
        else:
            self.pool = torch.sum

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

        self.fc1 = None
        self.activation = None
        self.fc2 = None
        if rho:
            self.fc1 = nn.Linear(input_size, hidden_layer)

            assert activation in ['relu', 'tanh', 'elu']
            if activation == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'elu':
                self.activation = nn.ELU(inplace=True)

            self.fc2 = nn.Linear(hidden_layer, output_size)

        self.rho = rho

    def forward(self, x):
        # [batch, bag, *rest]
        batch_size = x.size(0)
        bag_size = x.size(1)
        if self.phi:
            x = x.view(-1, *x.size()[2:])
            x = self.phi(x)
            x = x.view(batch_size, bag_size, *x.size()[1:])

        x = self.pool(x, dim=1)

        if self.dropout:
            x = self.dropout(x)

        if self.rho:
            x = self.fc1(x)
            x = self.activation(x)

            x = self.fc2(x)

        return x


class MILAttentionPool(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_attentions, gated=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_attentions = n_attentions

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_dim, self.n_attentions)
        self.fc3 = None
        self.sigmoid = None
        if gated:
            self.fc3 = nn.Linear(self.input_dim, self.hidden_dim)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x  # [batch, bag, input_dim]

        batch_size = x.size(0)
        bag_size = x.size(1)

        x = x.view(-1, *x.size()[2:])  # [batch*bag, input_dim]
        A = self.fc1(x)
        A = self.tanh(A)
        if self.fc3:
            A = A * self.sigmoid(self.fc3(x))

        A = self.fc2(A)  # [batch*bag, n_attentions]
        A = A.view(batch_size, bag_size, *A.size()[1:])
        A = torch.transpose(A, 2, 1)
        A = F.softmax(A, dim=2)  # [batch, bag, n_attentions]

        x = torch.bmm(A, input)  # [batch, n_attentions, input_dim]
        x = x.view(batch_size, -1)
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, lstm_size, disable_hidden_init=False):
        super().__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size

        self.LSTM = nn.LSTM(input_size,
                            lstm_size,
                            batch_first=True,
                            bidirectional=True)

        self.disable_hidden_init = disable_hidden_init

    def reset_parameters(self):
        all_param_names = [n for l in self.LSTM._all_weights for n in l]
        for param_name in all_param_names:
            param = self.LSTM.__getattr__(param_name)
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def init_hidden(self, input, batch_size):
        h_0 = input.new_empty(2, 1, self.lstm_size)
        c_0 = input.new_empty(2, 1, self.lstm_size)

        nn.init.xavier_uniform_(h_0)
        nn.init.xavier_uniform_(c_0)

        h_0 = h_0.repeat(1, batch_size, 1)
        c_0 = c_0.repeat(1, batch_size, 1)

        return h_0, c_0

    def forward(self, x):
        if not self.disable_hidden_init:
            h0 = self.init_hidden(x, x.size(0))
        else:
            h0 = None
        out, _ = self.LSTM(x, h0)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, input_size, da, row):
        super().__init__()
        self.input_size = input_size
        self.da = da
        self.row = row

        self.register_parameter(
            'ws1', nn.Parameter(torch.empty(self.input_size, self.da)))
        self.tanh = nn.Tanh()

        self.register_parameter('ws2',
                                nn.Parameter(torch.empty(self.da, self.row)))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ws1)
        nn.init.xavier_uniform_(self.ws2)

    def forward(self, x):
        # x.size() -> [b, seq_l, input_size]
        x = x.contiguous()

        batch_size = x.size(0)

        int_input1 = x
        int_input2 = x

        int_input1 = int_input1.view(-1, self.input_size)
        int_input1 = self.tanh(int_input1 @ self.ws1)
        int_input1 = int_input1 @ self.ws2
        int_input1 = int_input1.view(batch_size, -1, self.row)
        int_input1 = F.softmax(int_input1, dim=1)

        int_input2 = torch.transpose(int_input2, 2, 1)

        return torch.bmm(int_input2, int_input1)


class EmbedPoolMoEGate(nn.Module):
    def __init__(self,
                 mlp_dim,
                 vocab_size,
                 embedding_dim,
                 out_size,
                 sparse_gradient=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embedding_dim,
                                      sparse=sparse_gradient)

        self.fc1 = nn.Linear(2 * self.embedding_dim, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.fc3 = nn.Linear(self.mlp_dim, out_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

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


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = model_utils.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = model_utils.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = model_utils.conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = model_utils.conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = model_utils.conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

import importlib
import math
from itertools import zip_longest

import diffdist.functional as distF
import model.module as modules
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from logger.visualization import WriterTensorboardX


def import_matplotlib():
    global mpl
    global plt
    mpl = importlib.import_module('matplotlib')
    mpl.use('Agg')
    plt = importlib.import_module('matplotlib.pyplot')


import_matplotlib()


class SampleDistributor(object):
    def __init__(self, gates, num_experts):
        self.gates = gates
        self.num_experts = num_experts

        where = gates.t().nonzero()
        self.expert_index, self.batch_index = where.unbind(dim=1)
        self.samples_per_expert = torch.sum((gates > 0), dim=0)
        self.nonzero_gates = torch.gather(
            self.gates.view(-1), 0,
            self.batch_index * self.num_experts + self.expert_index)

    def split_input(self, inp):
        inp = torch.index_select(inp, 0, self.batch_index)
        return torch.split(inp, self.samples_per_expert.tolist(), dim=0)

    def _combine(self, expert_outs):
        expert_outs = list(expert_outs)
        example = get_a_tensor(expert_outs, exclude_empty=True)
        for i, e in enumerate(expert_outs):
            expert_outs[i] = e.to(example)

        gates = self.nonzero_gates.view(-1, 1)
        stitched = gates * torch.cat(expert_outs, dim=0)
        combined = stitched.new_zeros((self.gates.size(0), stitched.size(1)))
        combined.index_add_(0, self.batch_index, stitched)
        return combined

    def combine(self, expert_outs):
        if is_list_of_tensors(expert_outs):
            return self._combine(expert_outs)
        x = get_a_tensor(expert_outs, exclude_empty=True)
        return list(
            map(self._combine,
                zip_longest(*expert_outs, fillvalue=x.new_tensor([]))))


class DistributedSampleDistributor(object):
    def __init__(self, gates, experts, group=dist.group.WORLD, training=True):
        self.num_experts = len(experts)
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.samples_per_device = gates.size(0)
        self.exp2rank = [e.rank for e in experts]

        self.gates = gates

        where = gates.t().nonzero()
        self.expert_index, self.batch_index = where.unbind(dim=1)
        self.samples_per_expert = torch.sum((gates > 0), dim=0)
        self.nonzero_gates = torch.gather(
            self.gates.view(-1), 0,
            self.batch_index * self.num_experts + self.expert_index)

        rank2exp_samples = [
            torch.zeros_like(self.samples_per_expert)
            for _ in range(self.world_size)
        ]
        dist.all_gather(rank2exp_samples,
                        self.samples_per_expert,
                        group=self.group)
        self.rank2exp_samples = torch.stack(rank2exp_samples)
        if self.rank == 0 and not training:
            print(torch.sum(self.rank2exp_samples, dim=0))

    def split_input(self, inp):
        inp = torch.index_select(inp, 0, self.batch_index)
        res = list(torch.split(inp, self.samples_per_expert.tolist(), dim=0))

        # Distribute to remote experts
        gather_lists = []
        for i in range(self.num_experts):
            dst = self.exp2rank[i]
            rank2exp_i = self.rank2exp_samples[:, i]

            #####################################
            max_samples = torch.max(rank2exp_i).item()
            size = torch.Size((max_samples, *inp.size()[1:]))
            extra_samples = max_samples - res[i].size(0)
            padding = [0] * res[i].dim() * 2
            padding[-1] = extra_samples
            res[i] = torch.nn.functional.pad(res[i], padding)
            #####################################

            if self.rank == dst:
                #####################################
                gather_list = [inp.new_zeros(size) for _ in rank2exp_i]
                #####################################
                # gather_list = [
                #     inp.new_zeros((s, *inp.size()[1:])).requires_grad_(True)
                #     for s in rank2exp_i
                # ]
            else:
                gather_list = []
            gather_lists.append(gather_list)

        final_res = distF.multi_gather(gather_lists,
                                       res,
                                       list_dst=self.exp2rank,
                                       group=self.group,
                                       inplace=True)

        #####################################
        for i in range(self.num_experts):
            for j in range(len(final_res[i])):
                samples = self.rank2exp_samples[j, i]
                final_res[i][j] = final_res[i][j][:samples]
        #####################################

        for i, e in enumerate(final_res):
            if e:
                final_res[i] = torch.cat(e, dim=0)
            else:
                final_res[i] = inp.new_tensor([])

        return final_res

    def _combine(self, expert_outs):
        expert_outs = list(expert_outs)
        example = get_a_tensor(expert_outs, exclude_empty=True)
        for i, e in enumerate(expert_outs):
            expert_outs[i] = e.to(example)

        scatter_lists = []
        buffers = []
        for i, e in enumerate(expert_outs):
            src = self.exp2rank[i]
            rank2exp_i = self.rank2exp_samples[:, i]

            max_samples = torch.max(rank2exp_i).item()

            if self.rank == src:
                scatter_list = list(torch.split(e, rank2exp_i.tolist(), dim=0))

                #####################################
                padding = [0] * scatter_list[0].dim() * 2
                for i in range(len(scatter_list)):
                    extra_samples = max_samples - scatter_list[i].size(0)
                    padding[-1] = extra_samples
                    scatter_list[i] = torch.nn.functional.pad(
                        scatter_list[i], padding)
                #####################################
            else:
                scatter_list = []
            scatter_lists.append(scatter_list)

            # n_samples = rank2exp_i[self.rank]
            # temp = example.new_zeros(
            # (n_samples, *example.size()[1:])).requires_grad_(True)

            #####################################
            temp = example.new_zeros((max_samples, *example.size()[1:]))
            #####################################

            buffers.append(temp)

        final_outs = distF.multi_scatter(buffers,
                                         scatter_lists,
                                         list_src=self.exp2rank,
                                         group=self.group,
                                         inplace=True)

        #####################################
        for i in range(len(final_outs)):
            final_outs[i] = final_outs[i][:self.samples_per_expert[i]]
        #####################################

        expert_outs = final_outs
        # print(expert_outs)

        gates = self.nonzero_gates.view(-1, 1)
        # Use masked_select on nonzero_gates to select elements on current device
        stitched = gates * torch.cat(expert_outs, dim=0)
        combined = stitched.new_zeros((self.gates.size(0), stitched.size(1)))
        combined.index_add_(0, self.batch_index, stitched)
        return combined

    def combine(self, expert_outs):
        if is_list_of_tensors(expert_outs):
            return self._combine(expert_outs)
        x = get_a_tensor(expert_outs, exclude_empty=True)
        return list(
            map(self._combine,
                zip_longest(*expert_outs, fillvalue=x.new_tensor([]))))


def check_parameter_number(model, return_groups, requires_grad):
    res = model.parameters(return_groups=return_groups,
                           requires_grad=requires_grad)
    correct_params = model.parameters(return_groups=False,
                                      requires_grad=requires_grad)
    correct_params = sum([np.prod(p.size()) for p in correct_params])

    actual_params = 0
    for k, v in res.items():
        temp = sum([np.prod(p.size()) for p in v])
        actual_params += temp

    assert actual_params == correct_params, (
        f"actual_params: {actual_params}, correct_params: {correct_params}")


def convert_batch_to_counts(num_of_classes, batch_labels):
    res_vector = torch.bincount(batch_labels, minlength=num_of_classes).float()
    res_vector = res_vector.view(1, -1)
    return res_vector


def convert_batch_to_prob(num_of_classes, batch_labels, eps=0.01):
    res_vector = torch.bincount(batch_labels, minlength=num_of_classes).float()
    res_vector = res_vector + eps
    res_vector = res_vector / torch.sum(res_vector)
    res_vector = res_vector.view(1, -1)
    return res_vector


def convert_batch_to_log_prob(num_of_classes, batch_labels, eps=0.01):
    res_vector = torch.bincount(batch_labels, minlength=num_of_classes).float()
    res_vector = res_vector + eps
    res_vector = torch.log(res_vector) - torch.log(torch.sum(res_vector))
    res_vector = res_vector.view(1, -1)
    return res_vector


def write_prob_dist(output, target, label):
    output = output.view(-1)
    target = target.view(-1)
    writer = WriterTensorboardX(None, None, None)  # Singleton
    if writer.mode == 'train':
        return
    if writer.step % 100 != 0:
        return
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(len(output)), output.cpu(), alpha=0.6)
    ax.plot(np.arange(len(target)), target.cpu(), alpha=0.6)
    ax.legend(['Output', 'Target'], loc=2)
    ax.set_xlabel('Taxon ID', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f"Distribution of taxons at the {label} rank")
    writer.add_figure(label, fig)


def get_all_experts(model):
    return get_all_instances(model, modules.Expert)


def get_all_instances(model, cls):
    return [m for m in model.modules() if isinstance(m, cls)]


def get_a_tensor(input, exclude_empty=False):
    '''
    Attempts to iterate a given input datastructure and finds the first
    instance of torch.Tensor that it finds
    '''
    if isinstance(input, torch.Tensor):
        if not exclude_empty or input.size(0) > 0:
            return input

    if not isinstance(input, list) and not isinstance(
            input, dict) and not isinstance(input, tuple):
        raise TypeError('Cannot find tensor in input..')

    if isinstance(input, dict):
        input = input.items()

    for i in input:
        try:
            return get_a_tensor(i, exclude_empty=exclude_empty)
        except TypeError:
            if i is input[-1]:
                raise TypeError('Cannot find tensor in input..')


def collect_tensors(input):
    if isinstance(input, torch.Tensor):
        return [input]

    if not isinstance(input, list) and not isinstance(
            input, dict) and not isinstance(input, tuple):
        return []

    if isinstance(input, dict):
        input = input.items()

    res = []
    for i in input:
        try:
            res += collect_tensors(i)
        except TypeError:
            pass
    return res


def all_tensors_to(input, device, non_blocking=False):
    try:
        return input.to(device=device, non_blocking=non_blocking)
    except AttributeError:
        pass

    if not isinstance(input, list) and not isinstance(
            input, dict) and not isinstance(input, tuple):
        return input

    is_dict = False
    if isinstance(input, dict):
        is_dict = True
        input = input.items()

    res = []
    for i in input:
        try:
            res.append(all_tensors_to(i, device, non_blocking=non_blocking))
        except TypeError:
            pass
    if is_dict:
        res = dict(res)
    elif isinstance(input, tuple):
        res = tuple(res)
    return res


def is_list_of_tensors(input):
    for i in input:
        if not torch.is_tensor(i):
            return False
    return True


def infer_device(input):
    tensors = collect_tensors(input)
    device = tensors[0].device
    for t in tensors:
        if t.device != device:
            raise TypeError('Cannot infer device from input..')
    return device


def normal_cdf(x, stddev):
    """Evaluates the CDF of the normal distribution.
    Normal distribution with mean 0 and standard deviation stddev,
    evaluated at x=x.
    input and output `Tensor`s have matching shapes.
    Args:
        x: a `Tensor`
        stddev: a `Tensor` with the same shape as `x`.
     Returns:
        a `Tensor` with the same shape as `x`.
    """
    return 0.5 * (1.0 + torch.erf(x / (math.sqrt(2) * stddev + 1e-20)))


def cv_squared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
      x: a `Tensor`.
    Returns:
      a `Scalar`.
    """
    epsilon = 1e-10
    float_size = torch.numel(x) + epsilon
    mean = torch.sum(x) / float_size
    squared_difference = (x - mean)**2
    variance = torch.sum(squared_difference) / float_size

    return variance / (mean**2 + epsilon)


def batch_normalization(inplanes, adaptive=True, track_running_stats=True):
    if adaptive:
        return modules.AdaptiveBatchNorm2d(
            inplanes,
            momentum=0.99,
            eps=0.001,
            track_running_stats=track_running_stats)
    else:
        return nn.BatchNorm2d(inplanes,
                              momentum=0.99,
                              eps=0.001,
                              track_running_stats=track_running_stats)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=True)


def avgpool2x1(stride=(2, 1)):
    return nn.AvgPool2d(kernel_size=(2, 1), stride=stride)


def conv_h_w(h, w, in_planes, out_planes, stride=1, bias=True, padding='same'):
    if padding == 'valid':
        stride = tuple([max(1, i) for i in stride])
        return nn.Conv2d(in_planes,
                         out_planes,
                         kernel_size=(h, w),
                         stride=stride,
                         bias=bias)
    elif padding == 'same':
        padding_h = (h - 1) // 2
        remainder_h = (h - 1) % 2
        padding_w = (w - 1) // 2
        remainder_w = (w - 1) % 2
        if isinstance(stride, int):
            stride = (stride, stride)
        stride_h, stride_w = stride
        if stride_h == 0:
            padding_h = 0
            remainder_h = 0
            stride_h = 1
        if stride_w == 0:
            padding_w = 0
            remainder_w = 0
            stride_w = 1
        return nn.Sequential(
            nn.ConstantPad2d((padding_w, padding_w + remainder_w, padding_h,
                              padding_h + remainder_h), 0),
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=(h, w),
                      stride=(stride_h, stride_w),
                      bias=bias))

    else:
        raise ValueError(
            f'padding must be either "same" or "valid". Got {padding}')

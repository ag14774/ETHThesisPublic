import diffdist.functions as funcs
import torch.distributed as dist
import torch.nn as nn


class ConsumeVariable(nn.Module):
    def __init__(self, set_ones_grad=False):
        """
        If set_ones_grad=True then the gradient w.r.t tensor_to_consume
        is set to 1 during backprop. Otherwise, it is set to 0.
        """
        super(ConsumeVariable, self).__init__()
        self.set_ones_grad = set_ones_grad

    def forward(self, tensor_to_consume, *tensors_to_return):
        tensors_to_return = funcs.ConsumeVariableFunc.apply(
            tensor_to_consume, self.set_ones_grad, *tensors_to_return)
        return tensors_to_return


class CoupleVariables(nn.Module):
    def __init__(self, set_ones_grad=None):
        """
        If set_ones_grad=True then the gradient w.r.t tensor_to_consume
        is set to 1 during backprop. Otherwise, it is set to 0.
        """
        super(CoupleVariables, self).__init__()
        if not set_ones_grad:
            set_ones_grad = []
        self.set_ones_grad = set_ones_grad

    def forward(self, tensor_list):
        return list(
            funcs.CoupleVariablesFunc.apply(self.set_ones_grad, *tensor_list))


class Send(nn.Module):
    def __init__(self, dst, group=dist.group.WORLD, tag=0):
        super(Send, self).__init__()
        self.dst = dst
        self.group = group
        self.tag = tag

    def forward(self, tensor):
        return funcs.SendFunc.apply(tensor, self.dst, self.group, self.tag)


class Recv(nn.Module):
    def __init__(self,
                 src=None,
                 group=dist.group.WORLD,
                 tag=0,
                 next_backprop=None,
                 inplace=True):
        super(Recv, self).__init__()
        self.next_backprop = next_backprop
        self.src = src
        self.group = group
        self.tag = tag
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        tensor, sender = funcs.RecvFunc.apply(tensor, self.src, self.group,
                                              self.tag, self.inplace)
        return tensor, sender.item()


class Broadcast(nn.Module):
    def __init__(self,
                 src,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(Broadcast, self).__init__()
        self.src = src
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        return funcs.BroadcastFunc.apply(tensor, self.src, self.group,
                                         self.inplace)


class Gather(nn.Module):
    def __init__(self,
                 dst=None,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(Gather, self).__init__()
        self.dst = dst
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor, gather_list=None):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        if dist.get_rank(self.group) == self.dst:
            return list(
                funcs.GatherFunc.apply(tensor, self.dst, self.group,
                                       self.inplace, *gather_list))
        else:
            return funcs.GatherFunc.apply(tensor, self.dst, self.group,
                                          self.inplace, None)


class Scatter(nn.Module):
    def __init__(self,
                 src=None,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(Scatter, self).__init__()
        self.src = src
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, tensor, scatter_list=None):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        if dist.get_rank(self.group) == self.src:
            return funcs.ScatterFunc.apply(tensor, self.src, self.group,
                                           self.inplace, *scatter_list)
        else:
            return funcs.ScatterFunc.apply(tensor, self.src, self.group,
                                           self.inplace, None)


class AllGather(nn.Module):
    def __init__(self,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(AllGather, self).__init__()
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, gather_list, tensor):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        return list(
            funcs.AllGatherFunc.apply(tensor, self.group, self.inplace,
                                      *gather_list))


class MultiGather(nn.Module):
    def __init__(self, list_dst, group=dist.group.WORLD, inplace=True):
        super(MultiGather, self).__init__()
        self.list_dst = list_dst
        self.group = group
        self.inplace = inplace

    def forward(self, list_of_gather_lists, list_of_inputs):
        lengths = [len(l) for l in list_of_gather_lists]
        flattened = [t for sublist in list_of_gather_lists for t in sublist]
        flattened.extend(list_of_inputs)
        result = funcs.MultiGatherFunc.apply(self.list_dst, self.group,
                                             self.inplace, *flattened)
        start_index = 0
        final = []
        for l in lengths:
            final.append(list(result[start_index:start_index + l]))
            start_index += l

        # print(final)
        return final


class MultiScatter(nn.Module):
    def __init__(self, list_src, group=dist.group.WORLD, inplace=True):
        super(MultiScatter, self).__init__()
        self.list_src = list_src
        self.group = group
        self.inplace = inplace

    def forward(self, list_of_buffers, list_of_scatter_lists):
        flattened = [t for sublist in list_of_scatter_lists for t in sublist]
        flattened.extend(list_of_buffers)
        result = funcs.MultiScatterFunc.apply(self.list_src, self.group,
                                              self.inplace, *flattened)

        return list(result)

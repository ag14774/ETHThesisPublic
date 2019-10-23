import diffdist.extra_collectives as dist_extra
import torch
import torch.distributed as dist
from torch.autograd import Function


class ConsumeVariableFunc(Function):
    @staticmethod
    def forward(ctx, tensor_to_consume, set_ones_grad, *tensors_to_return):
        ctx.save_for_backward(tensor_to_consume)
        ctx.set_ones_grad = set_ones_grad
        return tensors_to_return

    @staticmethod
    def backward(ctx, *grad_outputs):
        tensor_to_consume, = ctx.saved_tensors
        if ctx.set_ones_grad:
            fake_grad = torch.ones_like(tensor_to_consume)
        else:
            fake_grad = torch.zeros_like(tensor_to_consume)

        return (fake_grad, None) + grad_outputs


class CoupleVariablesFunc(Function):
    '''
    Identity function. This forces the coupling of variables in the DAG.
    set_ones_grad is a list of indices that specifies whether a tensor's grad
    should be set equal to all 1s.
    '''

    @staticmethod
    def forward(ctx, set_ones_grad, *tensors):
        ctx.set_ones_grad = set_ones_grad
        return tensors

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_outputs = list(grad_outputs)
        for i in ctx.set_ones_grad:
            grad_outputs[i] = torch.ones_like(grad_outputs[i])
        return (None, ) + tuple(grad_outputs)


class SendFunc(Function):
    @staticmethod
    def forward(ctx, tensor, dst, group=dist.group.WORLD, tag=0):
        ctx.save_for_backward(tensor)
        ctx.dst = dst
        ctx.group = group
        ctx.tag = tag
        dist.send(tensor, dst, group, tag)
        return tensor.new_tensor([])

    @staticmethod
    def backward(ctx, grad_output):
        tensor, = ctx.saved_tensors
        # TODO: Add ctx.needs_input_grad check
        grad_tensor = torch.empty_like(tensor)
        dist.recv(grad_tensor, ctx.dst, ctx.group, ctx.tag)

        return grad_tensor, None, None, None


class RecvFunc(Function):
    @staticmethod
    def forward(ctx,
                tensor,
                src=None,
                group=dist.group.WORLD,
                tag=0,
                inplace=True):
        if not inplace:
            tensor = torch.empty_like(tensor).requires_grad_(False)
        ctx.src = src
        ctx.group = group
        ctx.tag = tag
        sender = dist.recv(tensor, src, group, tag)
        if src:
            assert sender == src
        else:
            ctx.src = sender
        sender = torch.tensor(sender)
        ctx.mark_non_differentiable(sender)
        return tensor, sender

    @staticmethod
    def backward(ctx, grad_tensor, grad_sender):
        dist.send(grad_tensor, ctx.src, ctx.group, ctx.tag)
        return grad_tensor, None, None, None, None


class BroadcastFunc(Function):
    @staticmethod
    def forward(ctx, tensor, src, group=dist.group.WORLD, inplace=True):
        ctx.src = src
        ctx.group = group
        if dist.get_rank(group) == src:
            if not inplace:
                with torch.no_grad():
                    tensor = tensor.clone().requires_grad_(False)
        else:
            if not inplace:
                tensor = torch.empty_like(tensor).requires_grad_(False)
        dist.broadcast(tensor, src, group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        dist.reduce(grad_output,
                    ctx.src,
                    op=dist.ReduceOp.SUM,
                    group=ctx.group)
        return grad_output, None, None, None


class AllReduceFunc(Function):
    @staticmethod
    def forward(ctx, i):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class ReduceFunc(Function):
    @staticmethod
    def forward(ctx, i):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AllGatherFunc(Function):
    @staticmethod
    def forward(ctx, tensor, group, inplace, *gather_list):
        ctx.save_for_backward(tensor)
        ctx.group = group
        gather_list = list(gather_list)
        if not inplace:
            gather_list = [torch.empty_like(g) for g in gather_list]
        dist.all_gather(gather_list, tensor, group)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        dist_extra.reduce_scatter(grad_out, list(grads), group=ctx.group)
        return (grad_out, None, None) + grads


class GatherFunc(Function):
    @staticmethod
    def forward(ctx, input, dst, group, inplace, *gather_list):
        ctx.dst = dst
        ctx.group = group
        ctx.save_for_backward(input)
        if dist.get_rank(group) == dst:
            gather_list = list(gather_list)
            if not inplace:
                gather_list = [torch.empty_like(g) for g in gather_list]
            dist.gather(input, gather_list=gather_list, dst=dst, group=group)
            return tuple(gather_list)
        else:
            dist.gather(input, [], dst=dst, group=group)
            return input.new_tensor([])

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_input = torch.empty_like(input)
        if dist.get_rank(ctx.group) == ctx.dst:
            grad_outputs = list(grads)
            dist.scatter(grad_input,
                         grad_outputs,
                         src=ctx.dst,
                         group=ctx.group)
            return (grad_input, None, None, None) + grads
        else:
            dist.scatter(grad_input, [], src=ctx.dst, group=ctx.group)
            return grad_input, None, None, None, None


class ScatterFunc(Function):
    @staticmethod
    def forward(ctx,
                tensor,
                src,
                group=dist.group.WORLD,
                inplace=True,
                *scatter_list):
        ctx.src = src
        ctx.group = group
        if not inplace:
            tensor = torch.empty_like(tensor)
        if dist.get_rank(group) == src:
            ctx.save_for_backward(*scatter_list)
            scatter_list = list(scatter_list)
            dist.scatter(tensor, scatter_list, src=src, group=group)
        else:
            dist.scatter(tensor, [], src=src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_tensor):
        if dist.get_rank(ctx.group) == ctx.src:
            grad_outputs = [torch.empty_like(g) for g in ctx.saved_tensors]
            dist.gather(grad_tensor, grad_outputs, ctx.src, group=ctx.group)
            return (grad_tensor, None, None, None) + tuple(grad_outputs)
        else:
            dist.gather(grad_tensor, [], ctx.src, group=ctx.group)
            return grad_tensor, None, None, None, None


class MultiGatherFunc(Function):
    @staticmethod
    def forward(ctx, list_dst, group=dist.group.WORLD, inplace=True, *tensors):
        ctx.list_dst = list_dst
        ctx.group = group
        initial_tensors_per_rank = len(list_dst)
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        output = []
        start_index = 0
        for i in range(initial_tensors_per_rank):
            if list_dst[i] == rank:
                output.append(
                    list(tensors[start_index:start_index + world_size]))
                start_index += world_size
            else:
                output.append([])
        input = tensors[start_index:]
        assert len(input) == initial_tensors_per_rank
        ctx.save_for_backward(*input)
        if not inplace:
            tmp = []
            for l in output:
                tmp.append([torch.zeros_like(t) for t in l])
            output = tmp
        dist_extra.multi_gather(output,
                                input,
                                list_dst,
                                group=dist.group.WORLD)

        output = tuple(t for sublist in output for t in sublist)

        return output

    @staticmethod
    def backward(ctx, *grads):
        world_size = dist.get_world_size(ctx.group)
        rank = dist.get_rank(ctx.group)
        initial_tensors_per_rank = len(ctx.list_dst)
        grad_outputs = []
        start_index = 0
        for i in range(initial_tensors_per_rank):
            if ctx.list_dst[i] == rank:
                grad_outputs.append(
                    list(grads[start_index:start_index + world_size]))
                start_index += world_size
            else:
                grad_outputs.append([])
        input = ctx.saved_tensors
        grad_input = [torch.zeros_like(t) for t in input]
        dist_extra.multi_scatter(grad_input,
                                 grad_outputs,
                                 ctx.list_dst,
                                 group=ctx.group)
        return (None, None, None) + grads + tuple(grad_input)


class MultiScatterFunc(Function):
    @staticmethod
    def forward(ctx, list_src, group=dist.group.WORLD, inplace=True, *tensors):
        ctx.list_src = list_src
        ctx.group = group
        num_of_scatters = len(list_src)
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        scatter_lists = []
        start_index = 0
        sizes = []
        for i in range(num_of_scatters):
            if list_src[i] == rank:
                scatter_lists.append(
                    list(tensors[start_index:start_index + world_size]))
                sizes.append(scatter_lists[-1][-1].size())
                start_index += world_size
            else:
                scatter_lists.append([])
                sizes.append(None)
        buffers = list(tensors[start_index:])
        assert len(buffers) == num_of_scatters
        ctx.device = buffers[0].device
        ctx.sizes = sizes
        if not inplace:
            buffers = [torch.empty_like(t) for t in buffers]
        dist_extra.multi_scatter(buffers,
                                 scatter_lists,
                                 list_src,
                                 group=dist.group.WORLD)

        return tuple(buffers)

    @staticmethod
    def backward(ctx, *grads):
        world_size = dist.get_world_size(ctx.group)
        grad_outputs = list(grads)

        gather_lists = []
        for s in ctx.sizes:
            if s:
                gather_lists.append([
                    torch.empty(s, device=ctx.device)
                    for _ in range(world_size)
                ])
            else:
                gather_lists.append([])

        dist_extra.multi_gather(gather_lists,
                                grad_outputs,
                                ctx.list_src,
                                group=ctx.group)

        flattened = tuple([t for sublist in gather_lists for t in sublist])

        return (None, None, None) + flattened + grads

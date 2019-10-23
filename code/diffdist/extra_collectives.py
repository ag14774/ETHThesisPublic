import diffdist.util as util
import torch.distributed as dist
from torch.distributed import ReduceOp


def reduce_scatter(tensor,
                   tensor_list,
                   op=ReduceOp.SUM,
                   group=dist.group.WORLD,
                   async_op=False):
    rank = dist.get_rank(group)
    if tensor is None:
        tensor = tensor_list[rank]
    if tensor.dim() == 0:
        tensor = tensor.view(-1)
    tensor[:] = tensor_list[rank]
    ops = []
    for i in range(dist.get_world_size(group)):
        if i == rank:
            tmp = dist.reduce(tensor, rank, op, group, async_op=True)
        else:
            tmp = dist.reduce(tensor_list[i], i, op, group, async_op=True)
        ops.append(tmp)

    oplist = util.AsyncOpList(ops)
    if async_op:
        return oplist
    else:
        oplist.wait()


def multi_gather(output,
                 input,
                 list_dst,
                 group=dist.group.WORLD,
                 async_op=False):
    '''
    output: list of list of tensors (buffer)
    input: (list of tensors)
    '''
    rank = dist.get_rank(group)
    ops = []
    for i, dst in enumerate(list_dst):
        if rank != dst:
            assert output[i] == []

        op = dist.gather(input[i],
                         output[i],
                         dst=dst,
                         group=group,
                         async_op=True)
        ops.append(op)

    oplist = util.AsyncOpList(ops)
    if async_op:
        return oplist
    else:
        oplist.wait()


def multi_scatter(output,
                  input,
                  list_src,
                  group=dist.group.WORLD,
                  async_op=False):
    '''
    output: list of tensors (buffers)
    input: (list of list of tensors)
    '''
    rank = dist.get_rank(group)
    ops = []
    for i, src in enumerate(list_src):
        if rank != src:
            assert input[i] == []
        op = dist.scatter(output[i],
                          input[i],
                          src=src,
                          group=group,
                          async_op=True)
        ops.append(op)

    oplist = util.AsyncOpList(ops)
    if async_op:
        return oplist
    else:
        oplist.wait()

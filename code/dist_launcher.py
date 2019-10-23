import os
import signal
import subprocess
import sys
from argparse import REMAINDER, ArgumentParser

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import SpawnContext, _prctl_pr_set_pdeathsig


def _wrap(cmd, env, error_queue):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                            "helper utilty that will spawn up "
                            "multiple distributed processes"
                            " - Modified for use with bsub cluster utility")

    # Optional arguments for the launch helper
    parser.add_argument(
        "--nnodes",
        type=int,
        default=None,
        help="The number of nodes to use for distributed "
        "training. Default will use all nodes allocated by bsub")
    parser.add_argument(
        "--node_rank",
        type=int,
        default=None,
        help="The rank of the node for multi-node distributed "
        "training. Default is determined by the position of the current "
        "node in the environment variable LSB_HOSTS (after sorting)")
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="The number of processes to launch on each node, "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU. "
        "By default this is determined by CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        help="Master node (rank 0)'s address, should be either "
        "the IP address or the hostname of node 0, for "
        "single node multi-proc training, the "
        "--master_addr can simply be 127.0.0.1. "
        "By default this is set to the first device in the *sorted* LSB_HOSTS")
    parser.add_argument("--master_port",
                        default=29500,
                        type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communciation during distributed "
                        "training")

    # positional
    parser.add_argument("training_script",
                        type=str,
                        help="The full path to the single GPU training "
                        "program/script to be launched in parallel, "
                        "followed by all the arguments for the "
                        "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)

    return parser.parse_args()


def set_defaults(args, current_env=None):
    if not current_env:
        current_env = os.environ

    hosts = current_env.get("LSB_HOSTS", "").split()
    hosts = list(set(hosts))
    hosts.sort()

    if not hosts:
        raise EnvironmentError(
            "This script should only run within an LSF environment.")

    current_hostname = current_env.get("HOSTNAME")

    if not args.nnodes:
        args.nnodes = len(hosts)
    if not args.node_rank:
        for i, h in enumerate(hosts):
            if h == current_hostname:
                args.node_rank = i
                break
    if not args.nproc_per_node:
        devices = current_env.get("CUDA_VISIBLE_DEVICES").split(',')
        args.nproc_per_node = len(devices)

    if args.nnodes > 1:
        # TODO: Change this to use /etc/hosts
        master_hostname = hosts[0]
        args.master_addr = f"{master_hostname}.ib"


def get_remote_pids(master_addr, master_port, node_rank, nnodes, processes):
    old_environ = os.environ.copy()
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(nnodes)
    dist.init_process_group("gloo", rank=node_rank)

    tensor = torch.zeros(len(processes), dtype=torch.long)
    for i, p in enumerate(processes):
        tensor[i] = p.pid

    tensor_list = []
    for i in range(nnodes):
        tensor_list.append(torch.zeros_like(tensor))

    dist.all_gather(tensor_list, tensor)

    dist.destroy_process_group()
    os.environ = old_environ

    return tensor_list


def main():
    args = parse_args()

    current_env = os.environ.copy()
    set_defaults(args, current_env)

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["LOCAL_SIZE"] = str(args.nproc_per_node)

    processes = []
    error_queues = []
    mp = multiprocessing.get_context('spawn')

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        env = current_env.copy()
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        env["RANK"] = str(dist_rank)
        env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args

        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(cmd, env, error_queue),
            daemon=False,
        )

        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    spawn_context = SpawnContext(processes, error_queues)

    while not spawn_context.join():
        pass


if __name__ == "__main__":
    main()

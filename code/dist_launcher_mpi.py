import os
from argparse import REMAINDER, ArgumentParser
import subprocess


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
    parser.add_argument("--hosts", default=None, type=str)
    parser.add_argument("--master_port",
                        default=29500,
                        type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communciation during distributed "
                        "training")
    parser.add_argument("--env_vars",
                        default="",
                        type=str,
                        help="Comma separated environment variables to "
                        "pass to subprocesses through mpirun.")

    # positional / run_train.sh
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

    LSF = True

    if not hosts:
        LSF = False
        print("Running outside LSF environment...")

    if LSF:
        current_hostname = current_env.get("HOSTNAME")
    else:
        current_hostname = subprocess.run(['hostname'],
                                          text=True,
                                          check=True,
                                          capture_output=True).stdout.strip()
        hosts = [current_hostname]

    if not args.nnodes:
        args.nnodes = len(hosts)

    assert len(
        hosts
    ) == args.nnodes, "args.nnodes not equal to the number of detected hosts"

    if not args.nproc_per_node:
        try:
            devices = current_env.get("CUDA_VISIBLE_DEVICES").split(',')
        except AttributeError:
            raise EnvironmentError(
                'Cannot infer nproc_per_node..No GPU devices detected. Please set it manually.'
            )
        args.nproc_per_node = len(devices)

    if not args.master_addr:
        master_hostname = hosts[0]
        if LSF:
            args.master_addr = f"{master_hostname}.ib"
        else:
            args.master_addr = master_hostname

    if not args.hosts:
        args.hosts = ",".join(
            [f"{h}" for h in hosts for _ in range(args.nproc_per_node)])
        # TODO: HAVE SAME HOST MULTIPLE TIMES IN THE LIST INSTEAD OF USING :

    if args.env_vars:
        args.env_vars = args.env_vars.split(',')
    else:
        args.env_vars = []


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
    current_env["NNODES"] = str(args.nnodes)

    cmd = [args.training_script] + args.training_script_args

    # mpi_cmd = [
    #     'mpirun', '--allow-run-as-root', '--tag-output', '-np',
    #     str(dist_world_size), '-H', args.hosts, '-bind-to', 'none', '-map-by',
    #     'slot', '-mca', 'pml', 'ucx', '-mca', 'btl', '^vader,tcp,openib,uct',
    #     '-x', 'NCCL_DEBUG=INFO', '-x', 'LD_LIBRARY_PATH', '-x', 'PATH'
    # ]
    mpi_cmd = [
        'mpirun', '--allow-run-as-root', '--tag-output', '-np',
        str(dist_world_size), '-H', args.hosts, '-bind-to', 'none', '-map-by',
        'slot', '-mca', 'pml', 'ob1', '-mca', 'btl', '^openib', '-x',
        'NCCL_DEBUG=INFO', '-x', 'LD_LIBRARY_PATH', '-x', 'PATH'
    ]

    KEYS = [
        "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_SIZE", "NNODES"
    ] + args.env_vars

    extra_envs = ' '.join([f"-x {key}" for key in KEYS]).split(' ')

    final_cmd = mpi_cmd + extra_envs + cmd
    final_cmd = ' '.join(final_cmd)

    print("Executing: ", final_cmd)

    os.execve('/bin/sh', ['/bin/sh', '-c', final_cmd], current_env)


if __name__ == "__main__":
    main()

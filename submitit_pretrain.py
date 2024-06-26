# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import main_pretrain as trainer
import submitit

import shutil 
import time

use_slurm_temp_dir = True
src = '/home/a4ferrei/scratch/' 
cc_data_path = '/data/spencer_cutout/valid2/'

def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=4, type=int, help="Number of gpus to request on each node") #4
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=20, type=int, help="Duration of the job (minutes)") 
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    return parser.parse_args()


def get_shared_folder() -> Path:
    p = Path("/home/a4ferrei/scratch/astro-mask/jobs/")
    p.mkdir(exist_ok=True)
    return p


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_pretrain as trainer

        self._setup_gpu_args()
        trainer.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

        # move files from $SCRATCH to $SLURM_TMPDIR if True
        if use_slurm_temp_dir:

            # do transfer just once
            if job_env.global_rank == 0:
                print('start of data transfer to $SLURM_TMPDIR')
                dest = '$SLURM_TMPDIR'
                initial_start_time = time.time()
                destination = shutil.copytree(src+cc_data_path, dest+cc_data_path, dirs_exist_ok=True)  
                transfer_time = time.time() - initial_start_time
                print(destination)

                print('end of data transfer to $SLURM_TMPDIR')
                print(f'transfer time of {transfer_time} seconds')


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=6,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_account='rrg-kyi',
    )

    executor.update_parameters(name="mae")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()

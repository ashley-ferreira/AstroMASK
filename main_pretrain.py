# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

use_slurm_temp_dir = True

# TRY TO MAKE THIS RELATIVE? FOR USE ON CANFAR AND CC
canfar_dataloader_path = '/arc/home/ashley/SSL/git/dark3d/src/models/training_framework/'
canfar_data_path = '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/'
canfar_output_path = '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/'

src = '/home/a4ferrei/scratch/' # instead of '$SCRATCH'

# all of these will have some prefix of src or dest before them
cc_dataloader_path = '/github/dark3d/src/models/training_framework/'
cc_data_path = '/data/spencer_cutout/valid2/'
cc_output_path = '/astro-mask/'

n_cutouts = 5*10000
norm_method = 'min_max' 
patch_size = 8

import argparse
import datetime
import json
import os
import numpy as np
from pathlib import Path
import time
date_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

import sys
sys.path.insert(0, src+cc_dataloader_path)
import dataloaders.dataloaders

import wandb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler

import timm 
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import models_mae
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain import train_one_epoch

import shutil 
import time

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=64, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5,type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true', 
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default=f'{cc_data_path}', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default=f'{cc_output_path}/output_dir/{date_time}/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint') 

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    # sometimes this does not work for >0, error is: "ERROR: Unexpected bus error 
    # encountered in worker. This might be caused by insufficient shared memory (shm).""
    parser.add_argument('--num_workers', default=6, type=int) # CPUs per GPU right now

    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    print('main_pretrain.main')

    # Set the number of GPUs you want to use
    num_gpus = torch.cuda.device_count()

    # Define the device IDs you want to use
    device_ids = list(range(num_gpus))  # Using all available GPUs

    if use_slurm_temp_dir:
        print('start of data transfer to $SLURM_TMPDIR')
        # move files from $SCRATCH to $SLURM_TMPDIR
        dest = '$SLURM_TMPDIR'

        '''
        # do transfer just once (this does not work!)
        if num_gpus == 1 or torch.cuda.current_device() == device_ids[0]:
            initial_start_time = time.time()
            destination = shutil.copytree(src+cc_data_path, dest+cc_data_path, dirs_exist_ok=True)  
            transfer_time = time.time() - initial_start_time
            print(destination)

        else:
            transfer_time = 0
        '''

        # do transfer just once
        if misc.get_rank() == 0:
            initial_start_time = time.time()
            destination = shutil.copytree(src+cc_data_path, dest+cc_data_path, dirs_exist_ok=True)  
            transfer_time = time.time() - initial_start_time
            print(destination)

        else:
            transfer_time = 0

        print('end of data transfer to $SLURM_TMPDIR')
        print(f'transfer time of {transfer_time} seconds')

    else:
        dest = src #'$SCRATCH'
        transfer_time = 0
    
    temp_out_path = dest+str(args.output_dir)
    os.makedirs(temp_out_path, exist_ok=True)

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # currently doing my own center-crop and normalization
    #train_transforms = transforms.Compose([
            #transforms.CenterCrop(args.input_size), 
            #prep_data.Normalize(method='None'), <-- only add once we have settled on one
            #transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            #transforms.RandomHorizontalFlip(),
           #transforms.ToTensor(),]) 

    # split up the dataset in dataset_path into train and validation
    dataset_indices = list(range(n_cutouts))
    np.random.shuffle(dataset_indices)
    frac = 0.1 # decrease when we have more data
    val_split_index = int(np.floor(frac * n_cutouts))
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx) 
    
    # define keywords for dataset
    kwargs_train = {
        'n_cutouts': n_cutouts,
        'bands': ['u', 'g', 'r', 'i', 'z'],
        'cutout_size': args.input_size,
        'cutouts_per_file': 10000,
        'h5_directory': dest+cc_data_path
    }
    
    # define fataset and dataloaders
    train_dataset = dataloaders.SpencerHDF5ReaderDataset(**kwargs_train)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)

    print('workers:', args.num_workers)
    log_writer = None
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, norm_method=norm_method)#, patch_size=patch_size)
    model.to(device)
    model.to(torch.float32)

    model_without_ddp = model 
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set the project where this run will be logged
    run = wandb.init(
            entity="ashleyferreira",
            project="compute-can-mae",
            config={
                "base_lr": args.blr,
                "batch_size": args.batch_size,
                "mask_ratio": args.mask_ratio,
                "norm_pix_loss": args.norm_pix_loss,
                "model": args.model,
                "norm_method": norm_method,
                "checkpoint_loc": str(args.output_dir),
                "note": "",
                "data_path": src+cc_data_path,
                "norm_method": norm_method,
                "patch_size": patch_size,
                "train_val_split": frac,
                "use_slurm_temp_dir": use_slurm_temp_dir,
                "transfer_time": transfer_time,
            })
    
    print(f"Start training for {args.epochs} epochs")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        train_stats = train_one_epoch(model, train_loader, val_loader, optimizer, 
                                      device, epoch, loss_scaler, log_writer=log_writer, 
                                      args=args, norm_method=norm_method)

        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, 
                            optimizer=optimizer,loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            temp_out_path = dest + str(args.output_dir)
            with open(os.path.join(temp_out_path, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n") 

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        if use_slurm_temp_dir: # clean this all up later!!
            dest = '$SLURM_TMPDIR'
        else: 
            dest = src
        temp_out_path = dest+str(args.output_dir)
        Path(temp_out_path).mkdir(parents=True, exist_ok=True)
    main(args)
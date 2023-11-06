# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched
import wandb

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, test_loader, optimizer: torch.optim.Optimizer,
                    train_sampler, val_sampler,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, batch_size=1024):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20 # not used rn

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    total_batches = len(data_loader)
    print('TRAINING') # put some of this stuff in function for re-use in validation   
    for data_iter_step in range(total_batches):
        
        
         # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / total_batches + epoch, args)
        
        samples = data_loader.__getitem__(data_iter_step)
        real_batch_size = len(samples)
        samples = torch.from_numpy(samples).to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio) 
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= real_batch_size #accum_iter 
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            print(header + ' Batch [{}/{}]'.format(data_iter_step, total_batches) + ' Train Loss: {:.4f}'.format(loss))

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    print('VALIDATION') 
    model.eval()
    total_batches = len(test_loader)
    for data_iter_step in range(total_batches):
        samples = test_loader.__getitem__(data_iter_step)
        samples = torch.from_numpy(np.array(samples)).to(device, non_blocking=True)
       
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, skipping".format(loss_value))
            torch.cuda.empty_cache() # still happens and runs into mem error
        
        else:

            loss /= real_batch_size #accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
                print(header + ' Batch [{}/{}]'.format(data_iter_step, total_batches) + ' Val Loss: {:.4f}'.format(loss)) 

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_validation = misc.all_reduce_mean(loss_value)
    
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    wandb.log({
        "learning_rate": lr,
        "epochs": epoch,
        "train_loss": loss_value_reduce, 
        "val_loss": loss_value_validation,
    })
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
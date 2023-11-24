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
import random
import torch
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched
import wandb

def run_batch():
    pass 

def train_one_epoch(model: torch.nn.Module,
                    train_loader: Iterable, val_loader: Iterable, 
                    norm_method: str, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, args=None, batch_size=1024):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20 # not used rn

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    train_loss, train_unnorm_loss = [], []
    total_batches = len(train_loader) 
    print('TRAINING') # put some of this stuff in function for re-use in validation   
    for i_train, samples in enumerate(train_loader):       
         # we use a per iteration (instead of per epoch) lr scheduler
        if i_train % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, i_train / total_batches + epoch, args)
        
        real_batch_size = len(samples)
        #samples = torch.from_numpy(samples).to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss, unnorm_loss, _, _ = model(samples, mask_ratio=args.mask_ratio)#, norm_method=norm_method, norm_method_args=kwargs_lst)  # how is this the output? need to get into where loss is calculated
            
        loss_value = loss.item()
        unnorm_loss_value = unnorm_loss, loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= real_batch_size #accum_iter 
        unnorm_loss /= real_batch_size
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(i_train + 1) % accum_iter == 0)
        if (i_train + 1) % accum_iter == 0:
            optimizer.zero_grad # need to change this to count
            print(header + ' Batch [{}/{}]'.format(i_train, total_batches) + ' Train Loss: {:.6f}'.format(loss))

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (i_train + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((i_train / len(train_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
        train_loss.append(loss)
        train_unnorm_loss.append(unnorm_loss)
        
    # currently used - will need to reconsider for distributed
    train_loss_avg = sum(train_loss) / len(train_loss)
    train_unnorm_loss_avg = sum(train_unnorm_loss) / len(train_unnorm_loss)

    print('VALIDATION') 
    model.eval()
    val_loss, val_unnorm_loss = [], []
    total_batches = len(val_loader)
    for i_val, samples in enumerate(val_loader):
        #samples = torch.from_numpy(np.array(samples)).to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
       
        with torch.cuda.amp.autocast():
            loss, unnorm_loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        loss_value = loss.item()
        unnorm_loss_value = unnorm_loss.value()

        if not math.isfinite(loss_value):
            print("Loss is {}, skipping".format(loss_value))
            torch.cuda.empty_cache()
        
        else:

            loss /= real_batch_size #accum_iter
            unnorm_loss /= real_batch_size
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(i_val + 1) % accum_iter == 0)
            if (i_val + 1) % accum_iter == 0:
                optimizer.zero_grad()
                print(header + ' Batch [{}/{}]'.format(i_val, total_batches) + ' Val Loss: {:.6f}'.format(loss)) 
                
            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_validation = misc.all_reduce_mean(loss)
            
            val_loss.append(loss)
            val_unnorm_loss.append(unnorm_loss)
            
        val_loss_avg = sum(val_loss) / len(val_loss)
        val_unnorm_loss_avg = sum(val_unnorm_loss) / len(val_unnorm_loss)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    wandb.log({
        "learning_rate": lr,
        "epochs": epoch,
        "train_loss": train_loss_avg, 
        "val_loss": val_loss_avg, 
        "train_unnorm_loss": train_unnorm_loss_avg, 
        "val_unnorm_loss": val_unnorm_loss_avg
    })
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
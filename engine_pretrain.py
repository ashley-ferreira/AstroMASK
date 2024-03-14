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
import time
from loss_func import uniformity_loss
import util.cutout_scaling as sc
loss_method = 'square'

src = '/home/a4ferrei/scratch/'
cc_dataloader_path = '/github/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import run_training_step

def train_one_iter(model: torch.nn.Module, iter_num,
                    train_transforms, dataset, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, iter: int, loss_scaler,
                    log_writer=None, args=None, norm_method=None,
                    iterations=100, batch_size=64, val_frac=0.1):
    
    start_time = time.time()

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Iter: [{}]'.format(iter_num)

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    train_loss, train_unnorm_loss = [], []

    cutouts, catalog, tile = dataset.__next__()
    run_training_step((cutouts, catalog, tile))
    del catalog
    del tile
    n_cutouts, channels, pix_len1, pix_len2 = cutouts.shape
    print(type(cutouts))
    print(cutouts.shape)
    print(f'data loaded, {n_cutouts} cutouts extracted')

    # TEMP
    def normalize(cutout):
        min_overall = np.min(cutout)
        max_overall = np.max(cutout)
        normed_cutout = (cutout - min_overall) / (max_overall - min_overall)
        return normed_cutout
    cutouts = train_transforms(torch.from_numpy(normalize(cutouts)))
    cutouts = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norm_method = None # TEMP
    print('data transformed')

    # TEMP
    val_frac = 0.2

    # split up the dataset in dataset_path into train and validation
    dataset_indices = list(range(n_cutouts))
    np.random.shuffle(dataset_indices)
    val_split_index = int(np.floor(val_frac * n_cutouts))
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

    print(type(cutouts), type(train_idx), type(val_idx))
    train_cutouts = cutouts[train_idx]
    print(type(cutouts))
    print(len(train_idx), len(val_idx))
    val_cutouts = cutouts[val_idx] # why does it work for train but not val? --> maybe it is just one value    
    old_i_train = 0

    print('train len', len(train_idx))
    for i_train in range(len(train_idx)%batch_size):    
        samples = train_cutouts[old_i_train:i_train+1]
        old_i_train = i_train
        print(type(samples))
        print(samples.dtype)

        # we want all to be in float32 or else we get the following error:
        # Input type (double) and bias type (c10::Half) should be the same
        samples = samples.float() #.type(torch.cuda.FloatTensor)
        #print(samples.dtype)

        # we use a per iteration (instead of per epoch) lr scheduler
        if i_train % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, iter_num, args)
            #lr_sched.adjust_learning_rate(optimizer, i_train / len(train_idx) + iter, args)
        
        real_batch_size = len(samples)
        samples = samples.to(device, non_blocking=True)

        try: 
            with torch.cuda.amp.autocast():
                model_out = model(samples, mask_ratio=args.mask_ratio, loss_method=loss_method)

                if loss_method == 'UMAE':
                    loss_mae, _, _, cls_feats, outputs = model_out

                    if args.reg == 'none':
                        loss = torch.zeros_like(loss_mae)
                    else:
                        loss = uniformity_loss(cls_feats)

                else: 
                    loss, unnorm_loss = model_out

            loss_value = loss.item()
            unnorm_loss_value = unnorm_loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= (real_batch_size * accum_iter)
            unnorm_loss /= (real_batch_size * accum_iter)

            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(i_train + 1) % accum_iter == 0)
            if (i_train + 1) % accum_iter == 0:
                optimizer.zero_grad 
                print(header + ' Batch [{}/{}]'.format(i_train, len(train_idx)%batch_size) + ' Train Loss: {:.12f}'.format(loss))

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (i_train + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((i_train / len(train_idx) + iter) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                
            train_loss.append(loss_value) # mayeb div by batch size
            train_unnorm_loss.append(unnorm_loss_value) # stop doing this as a list

        except Exception as e:
            print('##############\n',e,'##############\n')
        
    # currently used - will need to reconsider for distributed
    train_loss_avg = sum(train_loss) / len(train_loss) 
    train_unnorm_loss_avg = sum(train_unnorm_loss) / len(train_unnorm_loss)
    
    model.eval()
    val_loss, val_unnorm_loss = [], []

    # should validation set be center cropped?
    old_i_val = 0
    print('val_len', len(val_idx))
    for i_val in range(len(val_idx)%batch_size):    
        samples = val_cutouts[old_i_val:i_val+1]
        old_i_val = i_val

        samples = samples.to(device, non_blocking=True)
        samples = samples.float()
       
        try: 
            with torch.cuda.amp.autocast():
                loss, unnorm_loss = model(samples, mask_ratio=args.mask_ratio)
            loss_value = loss.item()
            unnorm_loss_value = unnorm_loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, skipping".format(loss_value))
                torch.cuda.empty_cache()
            
            else:
                loss /= (real_batch_size * accum_iter)
                unnorm_loss /= (real_batch_size * accum_iter)
                
                print(header + ' Batch [{}/{}]'.format(i_val, len(val_idx)%batch_size) + ' Val Loss: {:.12f}'.format(loss)) 
                loss_value_validation = misc.all_reduce_mean(loss)
                
            val_loss.append(loss_value)
            val_unnorm_loss.append(unnorm_loss_value)

        except Exception as e:
            print('##############\n',e,'##############\n')

    val_loss_avg = sum(val_loss) / len(val_loss)
    val_unnorm_loss_avg = sum(val_unnorm_loss) / len(val_unnorm_loss)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    #_, _, pred, mask = model(samples, mask_ratio=args.mask_ratio, return_img=True) 
    
    #pred = model.unpatchify(pred) 
    #pred = pred.detach().cpu()
    #reconstructed_images = wandb.Image(pred[0,0:3,:,:]) 
    #truth_images = wandb.Image(samples.detach().cpu()[0,0:3,:,:])

    wandb.log({
                "learning_rate": lr,
                "iter": iter_num,
                "val_loss": val_loss_avg, 
                "train_loss": train_loss_avg,
                "train_unnorm_loss": train_unnorm_loss_avg, 
                "val_unnorm_loss": val_unnorm_loss_avg,
                "epoch_runtime": time.time()-start_time,
                #"truth_image": truth_images,
                #"reconstruction": reconstructed_images
                })
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

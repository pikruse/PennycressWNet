# imports 
import time
import os, sys, glob
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm.auto import tqdm
from PIL import Image
from importlib import reload
from torch.utils.data import DataLoader
from IPython.display import clear_output

# custom imports
sys.path.append('../')

from utils.GetLowestGPU import GetLowestGPU
from utils.GetLR import get_lr
import utils.Loss as Loss
import utils.BuildWNet as BuildWNet
import utils.WNetTileGenerator as TG

reload(Loss)

def train_model(model,
                optimizer,
                train_generator,
                val_generator,
                log_path,
                chckpnt_path,
                device,
                batch_size = 1,
                batches_per_eval = 1000,
                warmup_iters = 1000,
                lr_decay_iters = 120000,
                max_lr = 1e-3,
                min_lr = 1e-5,
                max_iters = 150000,
                log_interval = 1,
                eval_interval = 1000,
                early_stop = 50,
                n_workers = 32,
                ):
    
    """
    Runs training loop for a deep learning model

    Parameters:
        model (torch.nn.Module): model to train
        optimizer(torch.optim): optimizer to use

        train_generator (torch.utils.data.Dataset): training data generator
        val_generator (torch.utils.data.Dataset): validation data generator

        log_path (str): path to save log
        chckpnt_path (str): path to save model checkpoints
        device (torch.device): device to train on (e.g. cuda:0)

        batch_size (int): batch size
        batches_per_eval (int): number of batches to evaluate
        warmup_iters (int): number of warmup iterations for learning rate
        lr_decay_iters (int): number of iterations to decay learning rate over
        max_lr (float): maximum learning rate
        min_lr (float): minimum learning rate
        max_iters(int): maximum number of iterations to train for
        log_interval (int): number of iterations between logging
        eval_interval (int): number of iterations between evaluation
        early_stop (int): number of iterations to wait for improvement before stopping
        n_workers (int): number of workers for data loader

    Returns:
        None
    """




    # non-customizable options
    iter_update = 'train loss {1:.4e}, val loss {2:.4e}\r'
    best_val_loss = None # initialize best validation loss
    last_improved = 0 # start early stopping counter
    iter_num = 0 # initialize iteration counter
    t0 = time.time() # start timer

    # training loop
    # refresh log
    with open(log_path, 'w') as f: 
        f.write(f'iter_num,train_loss,val_loss\n')

    # keep training until break
    while True:

        # clear print output
        clear_output(wait=True)

        if best_val_loss is not None:
            print('---------------------------------------\n',
                f'Iteration: {iter_num} | Best Loss: {best_val_loss:.4e}\n', 
                '---------------------------------------', sep = '')
        else:
            print('-------------\n',
                f'Iteration: {iter_num}\n', 
                '-------------', sep = '')

        #
        # checkpoint
        #

        # shuffle dataloaders
        train_loader = DataLoader(
            train_generator, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True)
        val_loader = DataLoader(
            val_generator, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True)

        # estimate loss
        model.eval()
        with torch.no_grad():
            train_loss, val_loss = 0, 0
            with tqdm(total=batches_per_eval, desc=' Eval') as pbar:
                for (xbt, ybt), (xbv, ybv) in zip(train_loader, val_loader):
                    xbt, ybt = xbt.to(device), ybt.to(device)
                    xbv, ybv = xbv.to(device), ybv.to(device)

                    #compute train loss
                    train_segmentations, train_reconstructions = model(xbt)
                    train_l_soft_n_cut = Loss.soft_n_cut_loss(xbt, train_segmentations, device = device)
                    train_l_reconstruction = Loss.reconstruction_loss(ybt, train_reconstructions)
                    train_loss += (train_l_soft_n_cut + train_l_reconstruction).item()

                    #compute val loss
                    val_segmentations, val_reconstructions = model(xbv)
                    val_l_soft_n_cut = Loss.soft_n_cut_loss(xbv, val_segmentations, device = device)
                    val_l_reconstruction = Loss.reconstruction_loss(ybv, val_reconstructions)
                    val_loss += (val_l_soft_n_cut + val_l_reconstruction).item()

                    pbar.update(1)
                    if pbar.n == pbar.total:
                        break
            train_loss /= batches_per_eval
            val_loss /= batches_per_eval

        # set model back to training mode
        model.train()

        # update user
        print(iter_update.format(iter_num, train_loss, val_loss)) 

        # update log
        with open(log_path, 'a') as f: 
            f.write(f'{iter_num},{train_loss},{val_loss}\n')

        # checkpoint model
        if iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, chckpnt_path.format(iter_num))

        # book keeping
        if best_val_loss is None:
            best_val_loss = val_loss

        if iter_num > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                last_improved = 0
                print(f'*** validation loss improved: {best_val_loss:.4e} ***')
            else:
                last_improved += 1
                print(f'validation has not improved in {last_improved} steps')
            if last_improved > early_stop:
                print()
                print(f'*** no improvement for {early_stop} steps, stopping ***')
                break

        # --------
        # backprop
        # --------

        # shuffle dataloaders
        train_loader = DataLoader(
            train_generator, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True)

        # iterate over batches
        with tqdm(total=eval_interval, desc='Train') as pbar:
            for xb, yb in train_loader:

                # update the model
                xb, yb = xb.to(device), yb.to(device)

                #compute loss
                segmentations, reconstructions = model(xb)
                l_soft_n_cut = Loss.soft_n_cut_loss(xb, segmentations, device = device)
                l_reconstruction = Loss.reconstruction_loss(yb, reconstructions)
                loss += (l_soft_n_cut + l_reconstruction).item()

                if torch.isnan(loss):
                    print('loss is NaN, stopping')
                    break
                
                # apply learning rate schedule
                lr = get_lr(it = iter_num,
                            warmup_iters = warmup_iters, 
                            lr_decay_iters = lr_decay_iters, 
                            max_lr = max_lr, 
                            min_lr = min_lr)
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                loss.backward()

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                # update book keeping
                pbar.update(1)
                iter_num += 1
                if pbar.n == pbar.total:
                    break

        # break once hitting max_iters
        if iter_num > max_iters:
            print(f'maximum iterations reached: {max_iters}')
            break
            

    return None

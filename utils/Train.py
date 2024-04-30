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
import utils.SobelRegularization as Sobel

reload(Loss)
reload(Sobel)

def train_model(model,
                optimizers,
                train_generator,
                val_generator,
                log_path,
                chckpnt_path,
                device,
                batch_size = 32,
                batches_per_eval = 1000,
                warmup_iters = 10000,
                lr_decay_iters = 90000,
                max_lr = 1e-3,
                min_lr = 1e-5,
                max_iters = 150000,
                log_interval = 1,
                eval_interval = 1000,
                early_stop = 50,
                n_workers = 64,
                ):
    
    """
    Runs training loop for a deep learning model

    Parameters:
        model (torch.nn.Module): model to train
        optimizer(list of torch.optim): optimizers to use

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


    # split optimizers
    optE = optimizers[0]
    optW = optimizers[1]
    
    # non-customizable options
    iter_update = 'train N-cut loss: {1:.4e}, train reconstruction loss: {2:.4e}\nval N-cut loss: {3:.4e}, val reconstruction loss: {4:.4e}\r'

    best_n_cut = None # initialize best validation loss
    best_reconstruction = None # initialize best validation loss

    last_improved = 0 # start early stopping counter
    iter_num = 0 # initialize iteration counter
    # shrinkage = 1e-6 # scale rec. loss

    # init. losses
    n_cut_loss = Loss.NCutLoss2D(device = device)

    # training loop
    # refresh log
    with open(log_path, 'w') as f: 
        f.write(f'iter_num,train_n_cut,train_reconstruction,val_n_cut,val_reconstruction\n')

    # keep training until break
    while True:

        # clear print output
        clear_output(wait=True)

        if (best_n_cut and best_reconstruction) is not None:
            print('------------------------------------------------------------------------------\n',
                f'Iteration: {iter_num} | Best N-cut: {best_n_cut:.4e} | Best Reconstruction: {best_reconstruction:.4e}\n', 
                '------------------------------------------------------------------------------', sep = '')
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
            train_l_n_cut, train_l_reconstruction, val_l_n_cut, val_l_reconstruction = 0, 0, 0, 0
            with tqdm(total=batches_per_eval, desc=' Eval') as pbar:
                for (xbt, ybt), (xbv, ybv) in zip(train_loader, val_loader):
                    xbt, ybt = xbt.to(device), ybt.to(device)
                    xbv, ybv = xbv.to(device), ybv.to(device)

                    #compute train loss
                    train_segmentations, train_reconstructions = model(xbt)
                    train_l_n_cut += n_cut_loss(train_segmentations, xbt)
                    train_l_reconstruction += (Loss.reconstruction_loss(ybt, train_reconstructions))

                    #compute val loss
                    val_segmentations, val_reconstructions = model(xbv)
                    val_l_n_cut += n_cut_loss(val_segmentations, xbv)
                    val_l_reconstruction += (Loss.reconstruction_loss(ybv, val_reconstructions))

                    pbar.update(1)
                    if pbar.n == pbar.total:
                        break

            train_l_n_cut /= batches_per_eval
            train_l_reconstruction /= batches_per_eval

            val_l_n_cut /= batches_per_eval
            val_l_reconstruction /= batches_per_eval

        # set model back to training mode
        model.train()

        # update user
        print(iter_update.format(iter_num, 
                                 train_l_n_cut,
                                 train_l_reconstruction, 
                                 val_l_n_cut, 
                                 val_l_reconstruction)) 

        # update log
        with open(log_path, 'a') as f: 
            f.write(f'{iter_num},{train_l_n_cut},{train_l_reconstruction},{val_l_n_cut},{val_l_reconstruction}\n')

        # checkpoint model
        if iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer_E': optE.state_dict(),
                'optimizer_W': optW.state_dict(),
                'iter_num': iter_num,
                'best_n_cut': best_n_cut,
                'best_reconstruction': best_reconstruction
            }

            torch.save(checkpoint, chckpnt_path.format(iter_num))

        # book keeping
        if (best_n_cut and best_reconstruction) is None:
            best_n_cut = val_l_n_cut
            best_reconstruction = val_l_reconstruction

        if iter_num > 0:
            if val_l_n_cut < best_n_cut or val_l_reconstruction < best_reconstruction:
                if val_l_n_cut < best_n_cut:
                    best_n_cut = val_l_n_cut
                if val_l_reconstruction < best_reconstruction:
                    best_reconstruction = val_l_reconstruction
                last_improved = 0
                print(f'*** validation loss improved ***\n*** N-cut: {best_n_cut:.4e}, reconstruction: {best_reconstruction:.4e} ***')
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

                # apply learning rate schedule
                lr = get_lr(it = iter_num,
                            warmup_iters = warmup_iters, 
                            lr_decay_iters = lr_decay_iters, 
                            max_lr = max_lr, 
                            min_lr = min_lr)
                
                for param_group1, param_group2 in zip(optE.param_groups, optW.param_groups):
                    param_group1['lr'] = lr
                    param_group2['lr'] = lr
                
                # compute n_cut loss and update
                segmentations = model.forward_encoder(xb)
                l_n_cut = n_cut_loss(segmentations, xb)

                l_n_cut.backward(retain_graph=False)
                optE.step()
                optE.zero_grad(set_to_none=True)

                # compute reconstruction loss
                segmentations, reconstructions = model(xb)
                l_reconstruction = (Loss.reconstruction_loss(yb, reconstructions))

                l_reconstruction.backward(retain_graph=False)
                optW.step()
                optW.zero_grad(set_to_none=True)

                if torch.isnan(l_n_cut) or torch.isnan(l_reconstruction):
                    print('loss is NaN, stopping')
                    break

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

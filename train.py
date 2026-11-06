# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import utils
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import datasets
import numpy as np
import losses
import time
import os
import radam
from seg_hrnet import HighResolutionNet
from tqdm import tqdm

def parse_args():

    parser = argparse.ArgumentParser(description = 'Train HRNet')
    parser.add_argument('--data_root', help = 'dataset root dir', default = 'data/thyroid_500', type = str)
    parser.add_argument('--num_gpus', help = 'number of gpus', default = 1, type = int)
    parser.add_argument('--train_bsize', help = 'train batch size per gpu', default = 8, type = int)
    parser.add_argument('--val_bsize', help = 'val batch size per gpu', default = 4, type = int)
    parser.add_argument('--num_workers', help = 'dataloader workers', default = 4, type = int)
    parser.add_argument('--num_epochs', help = 'number of epochs', default = 200, type = int)
    parser.add_argument('--saved_path', help = 'checkpoint save filename', required = True, type = str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, torch.cuda.is_available())
    
    gpus = list(range(args.num_gpus))
    begin_epoch = 0
    num_epochs = args.num_epochs
    num_classes = 1
    output_dir = 'thyroid_large_output_hrnet'
    save_frequency = 10
    print_frequency = 10
    
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs('debug', exist_ok = True)

    alpha = 1
    beta_lm = 0.5
    norm = True
    full_dist = True
    weighted_sample = False
    lr = 1e-4
    radius = 7
    
    criterion = losses.DistLoss(alpha=alpha, beta_lm=beta_lm, norm=norm)
    
    
    model = HighResolutionNet(multi_scale=False, se=True, channel_comp=False, aux=False)
    optimizer = radam.RAdam(model.parameters(), lr=lr)
    model = nn.DataParallel(model, device_ids = gpus).cuda()
    
    train_loader, val_loader = datasets.get_dataloader_tn3k(args.data_root, dist=True, num_workers=args.num_workers, train_bsize=args.train_bsize, val_bsize=args.val_bsize, multi_scale=False, aux=False, full_dist=full_dist, imgz=(512, 512))
    max_gpu_alloc = 0
    best_val_dice = 1000

    begin_epoch = 0
    train_bsize = args.train_bsize * args.num_gpus
    val_bsize = args.val_bsize * args.num_gpus

    saved_path = args.saved_path


    start_time = time.time()

    model.train(True)
    
    for epoch in range(begin_epoch, num_epochs):
        
        print('Epoch {}:\nLoss:'.format(epoch), end = '')
        cur_epoch_loss = np.zeros((1+len(train_loader.dataset)//train_bsize, ))
        time_before_epoch = time.time()
        
        for batch_idx, sample in enumerate(tqdm(train_loader)):
            img, label, dist = sample

            img = img.cuda()
            label = label.cuda()
            dist = dist.cuda()


            pred = model(img)
            
            loss = criterion(pred, label, dist)



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            

            
            cur_epoch_loss[batch_idx] = loss.item()

            torch.cuda.empty_cache()

        print()
        print('train loss:', np.mean(cur_epoch_loss))
        epoch_elapsed_time = time.time() - time_before_epoch

        model.train(False)
   

        cur_val_dice = np.zeros((1+len(val_loader.dataset)//val_bsize, ))

        for batch_idx, sample in enumerate(val_loader):

            img, label, dist = sample

            img = img.cuda()
            label = label.cuda()
            dist = dist.cuda()
            


            pred = model(img)



            val_loss = criterion(pred, label, dist)

            cur_val_dice[batch_idx] = val_loss.item()


        

        max_gpu_alloc = max(max_gpu_alloc, utils.get_gpu_allocated(gpus))
        print('val loss:', cur_val_dice.mean())
        if cur_val_dice.mean() < best_val_dice:
            best_val_dice = cur_val_dice.mean()
            print('find best val dice', best_val_dice)
            checkpoint = {'state_dict' : model.module.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : epoch + 1, 'best_val_dice' : best_val_dice}

            torch.save(checkpoint, os.path.join(output_dir, 'best_' + saved_path))

            del checkpoint
        
        model.train(True)
    end_time = time.time()
    print('avg time:', end_time - start_time)


# --------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Yeshu Li (yli299@uic.edu)
# --------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.nn.modules.loss import _WeightedLoss

import torch
import torch.nn as nn


smooth = 1e-2

def SliceDiceLoss(y_true, y_pred):
    return 1 - slice_dice_coef(y_true, y_pred)

def slice_dice_coef(y_true_f, y_pred_f):
    intersection = (y_true_f * y_pred_f).sum(dim = (2, 3))
    DSC = (2. * intersection + smooth) / (y_true_f.sum(dim = (2, 3)) + y_pred_f.sum(dim = (2, 3)) + smooth)
    return DSC

def local_maximum(x):
    dx1 = x[:, :, 1:-1, 1:-1] - x[:, :, 2:, 1:-1]
    dx2 = x[:, :, 1:-1, 1:-1] - x[:, :, 0:-2, 1:-1]
    dx3 = x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, 2:]
    dx4 = x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, 0:-2]
    dx = torch.cat([dx1, dx2, dx3, dx4], dim=1)
    dx, _ = torch.min(dx, dim=1, keepdim=True)
    dx = dist2seg(dx)
    return dx

class DiceLoss(_WeightedLoss):

    def __init__(self, eps = 0.0001):

        super(DiceLoss, self).__init__()
        self.eps = eps
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        return

    def forward(self, pred, target):  # (B, 1, H, W)

        pred = self.sigmoid(pred)

        loss = SliceDiceLoss(target, pred)
        
        return loss.mean(dim=(0,1))

def dist2seg(x):
    gamma = 1.0
    beta = 2.0
    return nn.ReLU()(1 - beta*nn.ReLU()(gamma - x))

class DistLoss(_WeightedLoss):

    def __init__(self, eps = 0.0001, alpha=1.0, beta_lm=1.0, norm=False):

        super(DistLoss, self).__init__()
        self.eps = eps
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.gamma = 1.1
        self.beta = 1 / (2 * self.gamma - 2)
        self.alpha = alpha
        self.norm = norm
        self.beta_lm = beta_lm

        print(self.norm, self.alpha, self.beta_lm)

        return

    def forward(self, pred, target, dist):  # (B, 1, H, W)

        _, _, h, w = target.size()
        fg_area = target.sum(dim=(2, 3))
        fg_bg_ratio = fg_area / (h * w - fg_area)
        fg_pred = self.relu(pred)
        bg_pred = fg_pred - pred
        fg_dist = self.relu(dist)
        bg_dist = fg_dist - dist
        if self.norm:
            fg_dist_amax = fg_dist.amax(dim=(2,3))
            fg_mse = torch.square((fg_pred - fg_dist)).mean(dim=(2,3)) / (fg_dist_amax ** 2)
            bg_mse = torch.square((bg_pred - bg_dist)).mean(dim=(2,3)) / (fg_dist_amax ** 2)
        else:
            fg_mse = torch.square(fg_pred - fg_dist).mean(dim=(2,3))
            bg_mse = torch.square(bg_pred - bg_dist).mean(dim=(2,3))

        pred_lm = local_maximum(fg_pred)
        true_lm = local_maximum(fg_dist)


        lm_diff = torch.square(pred_lm - true_lm).mean(dim=(2,3))

        seg = self.relu(1 - self.beta * self.relu(self.gamma - fg_pred))
        
        dice = SliceDiceLoss(target, seg)
        
        loss = dice + self.alpha * ((1 - fg_bg_ratio) * fg_mse + fg_bg_ratio * bg_mse) + self.beta_lm * lm_diff
        return loss.mean(dim=(0,1))

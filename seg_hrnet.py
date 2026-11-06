# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace

BN_MOMENTUM = 0.1
ALIGN_CORNERS = None

logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


config = {
  'NAME': 'seg_hrnet',
  'ALIGN_CORNERS': False,
  'EXTRA':{'FINAL_CONV_KERNEL': 1,
    'STAGE1':{'NUM_MODULES': 1,
      'NUM_BRANCHES': 1,
      'BLOCK': 'BOTTLENECK',
      'NUM_BLOCKS': [4],
      'NUM_CHANNELS': [64],
      'FUSE_METHOD': 'SUM'}, 
    'STAGE2':{'NUM_MODULES': 1,
      'NUM_BRANCHES': 2,
      'BLOCK': 'BASIC',
      'NUM_BLOCKS': [4, 4],
      'NUM_CHANNELS': [48, 96],
      'FUSE_METHOD': 'SUM'},
    'STAGE3':{'NUM_MODULES': 4,
      'NUM_BRANCHES': 3,
      'BLOCK': 'BASIC',
      'NUM_BLOCKS': [4, 4, 4],
      'NUM_CHANNELS': [48, 96, 192],
      'FUSE_METHOD': 'SUM'},
    'STAGE4':{'NUM_MODULES': 3,
      'NUM_BRANCHES': 4,
      'BLOCK': 'BASIC',
      'NUM_BLOCKS': [4, 4, 4, 4],
      'NUM_CHANNELS':[48, 96, 192, 384],
      'FUSE_METHOD': 'SUM'}
      }
    }

config_tiny = {
  'NAME': 'seg_hrnet',
  'ALIGN_CORNERS': False,
  'EXTRA':{'FINAL_CONV_KERNEL': 1,
    'STAGE1':{'NUM_MODULES': 1,
      'NUM_BRANCHES': 1,
      'BLOCK': 'BOTTLENECK',
      'NUM_BLOCKS': [4],
      'NUM_CHANNELS': [32],
      'FUSE_METHOD': 'SUM'}, 
    'STAGE2':{'NUM_MODULES': 1,
      'NUM_BRANCHES': 2,
      'BLOCK': 'BASIC',
      'NUM_BLOCKS': [4, 4],
      'NUM_CHANNELS': [24, 48],
      'FUSE_METHOD': 'SUM'},
    'STAGE3':{'NUM_MODULES': 4,
      'NUM_BRANCHES': 3,
      'BLOCK': 'BASIC',
      'NUM_BLOCKS': [4, 4, 4],
      'NUM_CHANNELS': [24, 48, 96],
      'FUSE_METHOD': 'SUM'},
    'STAGE4':{'NUM_MODULES': 3,
      'NUM_BRANCHES': 4,
      'BLOCK': 'BASIC',
      'NUM_BLOCKS': [4, 4, 4, 4],
      'NUM_CHANNELS':[24, 48, 96, 192],
      'FUSE_METHOD': 'SUM'}
      }
    }

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze
        y = self.fc(y).view(b, c, 1, 1) # FC
        return x * y.expand_as(x)
    
class SE_Block_TopK(nn.Module):
    def __init__(self, ch_in, reduction=16, k=48):
        super(SE_Block_TopK, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.k = k
 
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze
        y = self.fc(y).view(b, c, 1, 1) # FC
        _, topk_idx = torch.topk(y, self.k, dim=1, largest=True, sorted=True)  # x (b, c, h, w) idx (b, 48, 1, 1)
        topk_idx = topk_idx.expand(b, self.k, h, w)
        return torch.gather(x, dim=1, index=topk_idx) # output[i][j][k][l] = x[i][index[i][j][k][l]][k][l]  (b, 48, h, w)


class HighResolutionNet(nn.Module):

    def __init__(self, multi_scale, se, channel_comp, aux, **kwargs):
        global ALIGN_CORNERS
        extra = config['EXTRA']
        super(HighResolutionNet, self).__init__()
        ALIGN_CORNERS = config['ALIGN_CORNERS']
        self.multi_scale = multi_scale
        self.channel_comp = channel_comp
        self.aux = aux

        # stem net
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = int(np.sum(pre_stage_channels))

        
        if self.channel_comp:
            setopk_inp_channels = int(np.sum(pre_stage_channels))
            channel_comp_param = [48, 48, 48]
            last_inp_channels = 360 #48 + channel_comp_param[0] + channel_comp_param[1] + channel_comp_param[2]

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=1,
                kernel_size=extra['FINAL_CONV_KERNEL'],
                stride=1,
                padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
        )
        if self.multi_scale:
            self.last_layer1 = nn.Conv2d(
                    in_channels=96,
                    out_channels=4,
                    kernel_size=extra['FINAL_CONV_KERNEL'],
                    stride=1,
                    padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
            self.deconv1_1 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1) # out = (in - 1)*2 - 2 + 4 = 2 * in
            

            self.last_layer2 = nn.Conv2d(
                    in_channels=192,
                    out_channels=16,
                    kernel_size=extra['FINAL_CONV_KERNEL'],
                    stride=1,
                    padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
            self.deconv2_1 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)
            self.deconv2_2 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)

            self.last_layer3 = nn.Conv2d(
                    in_channels=384,
                    out_channels=64,
                    kernel_size=extra['FINAL_CONV_KERNEL'],
                    stride=1,
                    padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)

            self.deconv3_1 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
            self.deconv3_2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)
            self.deconv3_3 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)
            
        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        self.se = se
        
        if self.se:
            if self.channel_comp:
                # self.cc1 = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1)
                # self.cc2 = nn.Conv2d(in_channels=192, out_channels=48, kernel_size=1, stride=1)
                # self.cc3 = nn.Conv2d(in_channels=384, out_channels=48, kernel_size=1, stride=1)
                self.topk_se = SE_Block_TopK(setopk_inp_channels, reduction=4, k=last_inp_channels)
            else:
                self.se_blk = SE_Block(last_inp_channels, reduction=4)
        if self.aux:
            self.global_max_pool = nn.AdaptiveMaxPool2d(1)
            self.aux_layer = nn.Sequential(
            nn.Linear(last_inp_channels, last_inp_channels//2),
            nn.ReLU(inplace=relu_inplace),
            nn.Linear(last_inp_channels//2, 1),
            nn.Sigmoid()
        )
  

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        # print(len(x), x[0].size(1), x[1].size(1), x[2].size(1), x[3].size(1))
        # exit()
        # x0: 48x128, x1: 96x64, x2:192x32, x3:384x16
        if self.multi_scale:
            x_1 = self.last_layer1(x[1])
            x_1 = self.deconv1_1(x_1)

            x_2 = self.last_layer2(x[2])
            x_2 = self.deconv2_1(x_2)
            x_2 = self.deconv2_2(x_2)

            x_3 = self.last_layer3(x[3])
            x_3 = self.deconv3_1(x_3)
            x_3 = self.deconv3_2(x_3)
            x_3 = self.deconv3_3(x_3)

            # x1: 4x64, x2:16x32, x3:64x16

        # if self.channel_comp:
        #     x1 = self.topk_se1(x[1])
        #     x2 = self.topk_se2(x[2])
        #     x3 = self.topk_se3(x[3])

        #     x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        #     x2 = F.interpolate(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        #     x3 = F.interpolate(x3, size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        # else:
        if self.aux:
            x_aux = torch.squeeze(torch.cat([self.global_max_pool(x[0]),self.global_max_pool(x[1]),self.global_max_pool(x[2]),self.global_max_pool(x[3])], 1), (2, 3))
            x_aux = self.aux_layer(x_aux)
            x_aux = torch.squeeze(x_aux, 1)

        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)
        if self.se:
            if self.channel_comp:
                x = self.topk_se(x)
            else:
                x = self.se_blk(x)

        x = self.last_layer(x)

        x = self.deconv1(x)
        x = self.deconv2(x)

        # if self.multi_scale:
        #     x_1 = self.deconv1(x_1)
        #     x_1 = self.deconv2(x_1)
        #     x_2 = self.deconv1(x_2)
        #     x_2 = self.deconv2(x_2)
        #     x_3 = self.deconv1(x_3)
        #     x_3 = self.deconv2(x_3)

        if self.multi_scale:
            return x, x_1, x_2, x_3 # 512, 64, 32, 16
        else:
            if self.aux:
                return x, x_aux
            return x

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()              
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class CFSM(nn.Module):
    def __init__(self, se, channel_comp, k):
        super(CFSM, self).__init__()
        self.se = se
        self.channel_comp = channel_comp
        extra = config['EXTRA']
        if self.se:
            if self.channel_comp:
                # self.cc1 = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1)
                # self.cc2 = nn.Conv2d(in_channels=192, out_channels=48, kernel_size=1, stride=1)
                # self.cc3 = nn.Conv2d(in_channels=384, out_channels=48, kernel_size=1, stride=1)
                self.topk_se = SE_Block_TopK(720, reduction=4, k=k)

                self.last_layer = nn.Sequential(
                    nn.Conv2d(
                        in_channels=k,
                        out_channels=k,
                        kernel_size=1,
                        stride=1,
                        padding=0),
                    BatchNorm2d(k, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=relu_inplace),
                    nn.Conv2d(
                        in_channels=k,
                        out_channels=1,
                        kernel_size=extra['FINAL_CONV_KERNEL'],
                        stride=1,
                        padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
                )
            else:
                self.se_blk = SE_Block(720, reduction=4)

                self.last_layer = nn.Sequential(
                    nn.Conv2d(
                        in_channels=720,
                        out_channels=720,
                        kernel_size=1,
                        stride=1,
                        padding=0),
                    BatchNorm2d(720, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=relu_inplace),
                    nn.Conv2d(
                        in_channels=720,
                        out_channels=1,
                        kernel_size=extra['FINAL_CONV_KERNEL'],
                        stride=1,
                        padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
                )
    
    def forward(self, x):
        if self.se:
            if self.channel_comp:
                x = self.topk_se(x)
            else:
                x = self.se_blk(x)
        x = self.last_layer(x)
        return x

class HighResolutionNet_tiny_se_out(nn.Module):

    def __init__(self, multi_scale, **kwargs):
        global ALIGN_CORNERS
        extra = config_tiny['EXTRA']
        super(HighResolutionNet_tiny_se_out, self).__init__()
        ALIGN_CORNERS = config['ALIGN_CORNERS']
        self.multi_scale = multi_scale

        # stem net
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 32, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=16,
                kernel_size=extra['FINAL_CONV_KERNEL'],
                stride=1,
                padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
        )
        if self.multi_scale:
            self.last_layer1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=48,
                    out_channels=24,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                BatchNorm2d(24, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(
                    in_channels=24,
                    out_channels=1,
                    kernel_size=extra['FINAL_CONV_KERNEL'],
                    stride=1,
                    padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
            )

            self.last_layer2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=96,
                    out_channels=24,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                BatchNorm2d(24, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(
                    in_channels=24,
                    out_channels=1,
                    kernel_size=extra['FINAL_CONV_KERNEL'],
                    stride=1,
                    padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
            )

            self.last_layer3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=192,
                    out_channels=24,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                BatchNorm2d(24, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(
                    in_channels=24,
                    out_channels=1,
                    kernel_size=extra['FINAL_CONV_KERNEL'],
                    stride=1,
                    padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
            )
        self.deconv1 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)

        self.se_blk = SE_Block(last_inp_channels, reduction=4)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        # print(len(x), x[0].size(1), x[1].size(1), x[2].size(1), x[3].size(1))
        # exit()
        # x0: 24x128, x1: 48x64, x2:96x32, x3:192x16
        if self.multi_scale:
            x_1 = self.last_layer1(x[1])

            x_2 = self.last_layer2(x[2])

            x_3 = self.last_layer3(x[3])

        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.se_blk(x)

        x = self.last_layer(x)

        x = self.deconv1(x)
        x = self.deconv2(x)

        if self.multi_scale:
            return x, x_1, x_2, x_3 # 512, 64, 32, 16
        else:
            return x
    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()              
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        

class HighResolutionNetV2(nn.Module):

    def __init__(self, **kwargs):
        global ALIGN_CORNERS
        extra = config['EXTRA']
        super(HighResolutionNetV2, self).__init__()
        ALIGN_CORNERS = config['ALIGN_CORNERS']

        # stem net
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = int(np.sum(pre_stage_channels))

        self.transform_layer = nn.Sequential(nn.ConvTranspose2d(last_inp_channels,
                                                       last_inp_channels//4,
                                                       kernel_size=4,
                                                       stride=2,
                                                       padding=1),
                                            BatchNorm2d(last_inp_channels//4, momentum=BN_MOMENTUM),
                                            nn.ReLU(inplace=relu_inplace),
                                            nn.ConvTranspose2d(last_inp_channels//4,
                                                       last_inp_channels//4,
                                                       kernel_size=4,
                                                       stride=2,
                                                       padding=1),
                                            BatchNorm2d(last_inp_channels//4, momentum=BN_MOMENTUM),
                                            nn.ReLU(inplace=relu_inplace))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels//4,
                out_channels=last_inp_channels//4,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels//4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=last_inp_channels//4,
                out_channels=1,
                kernel_size=extra['FINAL_CONV_KERNEL'],
                stride=1,
                padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
        )
        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        # print(len(x), x[0].size(2), x[0].size(3),x[1].size(2), x[1].size(3),x[2].size(2), x[2].size(3))
        # exit()
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.transform_layer(x)

        x = self.last_layer(x)

        # x = self.deconv1(x)
        # x = self.deconv2(x)

        return x

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()              
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model

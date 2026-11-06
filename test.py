from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import datasets
import numpy as np
import os
import time
import csv
import cv2
from scipy.ndimage import rotate
from seg_hrnet import HighResolutionNet


def parse_args():

    parser = argparse.ArgumentParser(description = 'Test HRNet')
    parser.add_argument('--data_root', help = 'dataset root dir', default = 'data/thyroid_500', type = str)
    parser.add_argument('--wname', help = 'checkpoint weight file name', default = 'xxx.pth.tar', type = str)
    parser.add_argument('--TTA', help = 'test time augmentation', action = 'store_true')
    args = parser.parse_args()

    return args


def list2csv(name, number_list):
    with open(name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for number in number_list:
            writer.writerow([number])


def evaluate_saved_model(data_root, wname, TTA):

    model = HighResolutionNet(multi_scale=False, se=True, channel_comp=False, aux=False)
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    checkpoint = torch.load(os.path.join('thyroid_large_output_hrnet', wname), map_location=torch.device('cpu'), weights_only=False)
    model.module.load_state_dict(checkpoint['state_dict'], strict=False)

    test_img_names, test_seg_names = datasets.get_dataset_2d_img_list_tn3k(data_root, div_name = 'test')
    time_list = []
    model.train(False)
    out_path = wname
    if TTA:
        out_path = out_path + '_TTA'
        wname = wname + '_TTA'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    dice = []
    iou = []

    for i in range(len(test_img_names)):
        img = cv2.imread(test_img_names[i])[:, :, 0] / 255.
        raw_shape = img.shape
        img = cv2.resize(img, (512, 512))
        if TTA:
            img_1 = np.copy(np.flip(img, axis=0))
            img_2 = np.copy(np.flip(img, axis=1))
            img_3 = rotate(img, 90, axes=(0, 1), reshape=False, order=1)
            img_4 = rotate(img, -90, axes=(0, 1), reshape=False, order=1)

        cv2.imwrite(os.path.join(out_path, test_img_names[i].split('\\')[-1]), (img * 255.).astype(np.uint8))
        img = torch.Tensor(img)
        if TTA:
            img_1 = torch.Tensor(img_1)
            img_2 = torch.Tensor(img_2)
            img_3 = torch.Tensor(img_3)
            img_4 = torch.Tensor(img_4)

        label = cv2.imread(test_seg_names[i])[:, :, 0] / 255.

        img = img[None, None, :, :]
        if TTA:
            img_1 = img_1[None, None, :, :]
            img_2 = img_2[None, None, :, :]
            img_3 = img_3[None, None, :, :]
            img_4 = img_4[None, None, :, :]

        with torch.no_grad():
            img = img.cuda()
            if TTA:
                img_1 = img_1.cuda()
                img_2 = img_2.cuda()
                img_3 = img_3.cuda()
                img_4 = img_4.cuda()
            start_time = time.time()

            pred = model(img)
            if TTA:
                pred_1 = model(img_1)
                pred_2 = model(img_2)
                pred_3 = model(img_3)
                pred_4 = model(img_4)

            end_time = time.time()
            time_list.append(end_time - start_time)
            pred = pred.detach().cpu().numpy()[0,0]
            if TTA:
                pred_1 = pred_1.detach().cpu().numpy()[0,0]
                pred_2 = pred_2.detach().cpu().numpy()[0,0]
                pred_3 = pred_3.detach().cpu().numpy()[0,0]
                pred_4 = pred_4.detach().cpu().numpy()[0,0]

            pred[pred <= 0] = 0
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            if TTA:
                pred_1[pred_1 <= 0] = 0
                pred_1[pred_1 >= 0.5] = 1
                pred_1[pred_1 < 0.5] = 0

                pred_2[pred_2 <= 0] = 0
                pred_2[pred_2 >= 0.5] = 1
                pred_2[pred_2 < 0.5] = 0

                pred_3[pred_3 <= 0] = 0
                pred_3[pred_3 >= 0.5] = 1
                pred_3[pred_3 < 0.5] = 0

                pred_4[pred_4 <= 0] = 0
                pred_4[pred_4 >= 0.5] = 1
                pred_4[pred_4 < 0.5] = 0

                pred_1 = np.flip(pred_1, axis=0)
                pred_2 = np.flip(pred_2, axis=1)
                pred_3 = rotate(pred_3, -90, axes=(0, 1), reshape=False, order=1)
                pred_4 = rotate(pred_4, 90, axes=(0, 1), reshape=False, order=1)
                pred = (pred + pred_1 + pred_2 + pred_3 + pred_4)  / 5.
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0

            pred = cv2.resize(pred, (raw_shape[1], raw_shape[0]))
            dice_i = 2*np.sum(pred*label)/(np.sum(pred)+np.sum(label))
            iou_i = np.sum(pred*label)/(np.sum(pred)+np.sum(label)-np.sum(pred*label))

            dice.append(dice_i)
            iou.append(iou_i)

    print('dice:', np.mean(dice), np.std(dice))
    print('iou:', np.mean(iou), np.std(iou))

    list2csv('dice_all_'+wname+'.csv', dice)
    list2csv('iou_all_'+wname+'.csv', iou)

    return 0


if __name__ == '__main__':

    args = parse_args()
    evaluate_saved_model(args.data_root, args.wname, args.TTA)

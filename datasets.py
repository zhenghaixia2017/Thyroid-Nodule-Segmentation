from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torch
import os
import numpy as np
from scipy.ndimage import rotate
import random
import cv2
from scipy.ndimage import distance_transform_edt


def paddedzoom(img, zoomfactor=0.8, img_type='img'):
    """
    Zoom in/out an image while keeping the input image shape.
    i.e., zero pad when factor<1, clip out when factor>1.
    there is another version below (paddedzoom2)
    """
    out = np.zeros_like(img)
    if img_type == 'img':
        zoomed = cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor, interpolation=cv2.INTER_AREA)
    else:
        zoomed = cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor, interpolation=cv2.INTER_NEAREST)

    h, w = img.shape
    zh, zw = zoomed.shape

    if h == zh or w == zw:
        return img

    if zoomfactor < 1:  # zero padded
        out[(h - zh) // 2:-(h - zh) // 2, (w - zw) // 2:-(w - zw) // 2] = zoomed
    else:  # clip out
        out = zoomed[(zh - h) // 2:-(zh - h) // 2, (zw - w) // 2:-(zw - w) // 2]

    return out

def zoomtransform(img, Pxy, zoomfactor=0.8, img_type='img'):
    if zoomfactor == 1:
        return img
    out = np.zeros_like(img)
    if Pxy >= 0.5:
        if img_type == 'img':
            zoomed = cv2.resize(img, None, fx=zoomfactor, fy=1, interpolation=cv2.INTER_AREA)
        else:
            zoomed = cv2.resize(img, None, fx=zoomfactor, fy=1, interpolation=cv2.INTER_NEAREST)
    else:
        if img_type == 'img':
            zoomed = cv2.resize(img, None, fx=1, fy=zoomfactor, interpolation=cv2.INTER_AREA)
        else:
            zoomed = cv2.resize(img, None, fx=1, fy=zoomfactor, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape
    zh, zw = zoomed.shape
    if h == zh:
        if w == zw:
            return img
        else:
            if w > zw:
                out[:, (w - zw) // 2:-(w - zw) // 2] = zoomed
            else:  # clip out
                out = zoomed[:, (zw - w) // 2:-(zw - w) // 2]
        return out
    else:
        if zoomfactor < 1:  # zero padded
            out[(h - zh) // 2:-(h - zh) // 2, :] = zoomed
        else:  # clip out
            out = zoomed[(zh - h) // 2:-(zh - h) // 2, :]
    return out

class MyCustom2DDataset(Dataset):

    def __init__(self, img_list, label_list, is_train, target_size, dist=False, full_dist=False, multi_scale=False, aux=False):

        super(MyCustom2DDataset, self).__init__()
        self.img_list = img_list
        self.label_list = label_list
        self.is_train = is_train
        self.target_size = target_size
        self.count = len(img_list)
        self.dist = dist
        self.full_dist = full_dist
        self.multi_scale = multi_scale
        self.aux = aux
        return

    def transform_volume(self, img, mask, prob = 0.2):

        P = random.uniform(0, 1)
        if P >= 0.5:
            degree = 10 * random.uniform(-1, 1)
            img = rotate(img, degree, axes=(0, 1), reshape=False, order=1)
            mask = rotate(mask, degree, axes=(0, 1), reshape=False, order=0)

        P = random.uniform(0, 1)
        if P >= 0.5:
            P1 = random.uniform(0, 1)
            if P1 >= 0.5:
                degree = 90
                img = rotate(img, degree, axes=(0, 1), reshape=False, order=1)
                mask = rotate(mask, degree, axes=(0, 1), reshape=False, order=0)
            else:
                degree = -90
                img = rotate(img, degree, axes=(0, 1), reshape=False, order=1)
                mask = rotate(mask, degree, axes=(0, 1), reshape=False, order=0)

        P = random.uniform(0, 1)
        if P >= 0.5:
            mvd = np.random.randint(-1 * int(0.1 * img.shape[0]), int(0.1 * img.shape[0]))
            img = np.roll(img, mvd, axis=0)
            mask = np.roll(mask, mvd, axis=0)

        P = random.uniform(0, 1)
        if P >= 0.5:
            mvd = np.random.randint(-1 * int(0.1 * img.shape[1]), int(0.1 * img.shape[1]))
            img = np.roll(img, mvd, axis=1)
            mask = np.roll(mask, mvd, axis=1)

        P = random.uniform(0, 1)
        if P >= 0.5:
            img = np.copy(np.flip(img, axis=0))
            mask = np.copy(np.flip(mask, axis=0))

        P = random.uniform(0, 1)
        if P >= 0.5:
            img = np.copy(np.flip(img, axis=1))
            mask = np.copy(np.flip(mask, axis=1))

        P = random.uniform(0, 1)
        if P >= 0.5:
            P1 = random.uniform(0, 1)
            if P1 >= 0.5:
                zoom_factor = np.random.uniform(0.8, 1.2)
                img = paddedzoom(img, zoomfactor=zoom_factor, img_type='img')
                mask = paddedzoom(mask, zoomfactor=zoom_factor, img_type='mask')
            else:
                zoom_factor = random.uniform(0.8, 1.2)
                Pxy = random.uniform(0, 1)
                img = zoomtransform(img, Pxy, zoomfactor=zoom_factor, img_type='img')
                mask = zoomtransform(mask, Pxy, zoomfactor=zoom_factor, img_type='mask')

        P = random.uniform(0, 1)
        if P >= 0.5:
            img = cv2.resize(img, (self.target_size[0]//2, self.target_size[1]//2), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        
        return img, mask

    def __getitem__(self, index):
        
        
        img = cv2.imread(self.img_list[index])[:, :, 0] / 255.0

        label = cv2.imread(self.label_list[index])[:, :, 0] / 255.

        img = cv2.resize(img, self.target_size)
        label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            img, label = self.transform_volume(img, label)

        if self.dist:

            fg_dist_edt = distance_transform_edt(label)
            bg_dist_edt = distance_transform_edt(1.0 - label)
            fg_dist_edt_max = np.max(fg_dist_edt)
            bg_dist_edt[bg_dist_edt>=fg_dist_edt_max] = fg_dist_edt_max
            dist = fg_dist_edt - bg_dist_edt

        
        img = torch.Tensor(img)
        label = torch.Tensor(label)

        if self.dist:
            dist = torch.Tensor(dist)
        
        img = img[np.newaxis,...]
        label = label[np.newaxis,...]
        if self.dist:
            dist = dist[np.newaxis,...]
        if self.dist:
            return (img, label, dist)
        return (img, label)

    def __len__(self):

        return self.count

def get_dataset_2d_img_list_tn3k(data_root, div_name):

    root_dir = data_root
    
    imgs = []
    segs = []

    for name in os.listdir(os.path.join(root_dir, div_name+'-image')):
        img_file_path = os.path.join(root_dir, div_name+'-image', name)
        seg_file_path = os.path.join(root_dir, div_name+'-mask', name)
        imgs.append(img_file_path)
        segs.append(seg_file_path)
    
    return imgs, segs

def get_dataloader_tn3k(data_root, dist=True, num_workers=4, train_bsize=8, val_bsize=4, multi_scale=False, aux=False, full_dist=False, weighted_sample=False, imgz=(512, 512)):

    trainval_img_names, trainval_seg_names = get_dataset_2d_img_list_tn3k(data_root, div_name = 'trainval')
    
    train_img_names, train_seg_names = trainval_img_names[0:-287], trainval_seg_names[0:-287]

    val_img_names, val_seg_names = trainval_img_names[-287:], trainval_seg_names[-287:]

    train_set = MyCustom2DDataset(train_img_names, train_seg_names, is_train = True, target_size=imgz, dist=dist, full_dist=full_dist, multi_scale=multi_scale, aux=aux, weighted_sample=weighted_sample)
    val_set = MyCustom2DDataset(val_img_names, val_seg_names, is_train = False, target_size=imgz, dist=dist, full_dist=full_dist, multi_scale=multi_scale, aux=aux, weighted_sample=False)

    drop_last=False
    train_loader = DataLoader(train_set, batch_size = train_bsize, shuffle = True, pin_memory = True, num_workers = num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size = val_bsize, shuffle = False, pin_memory = True, num_workers = num_workers, drop_last=drop_last)

    return train_loader, val_loader

import os
from glob import glob
import random
import csv

from PIL import Image
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms


def get_wsi_id_labels(csv_file_path):
    with open(csv_file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        header = None
        wsi_labels = dict()
        for row in spamreader:
            if header is None:
                header = row[0]
            else:
                wsi_id = row[6][1:-1]
                if row[3] == 'NA' or int(row[3]) < 0:
                    continue
                days = int(row[3])
                dead = int(row[4][1:-1])
                wsi_labels[wsi_id] = [days, dead]

    return wsi_labels


class WSI_Dataset(data.Dataset):
    def __init__(self, data_root, csv_file_path, input_nc, transform, mode='train', n_patches=16, interval=750, n_intervals=5):
        self.transform = transform
        self.wsi_path_list = glob('{}/{}/*'.format(data_root, mode))
        self.input_nc = input_nc
        self.n_patches = n_patches
        self.interval = interval
        self.n_intervals = n_intervals
        self.wsi_patches = dict()
        self.wsi_id_no = dict()

        for idx, wsi_path in enumerate(self.wsi_path_list, 0):
            wsi_id = wsi_path.split('/')[-1]
            patch_fn_list = glob('{}/*.png'.format(wsi_path))
            self.wsi_patches[wsi_id] = patch_fn_list
            self.wsi_id_no[idx] = wsi_id
            
        self.wsi_labels = get_wsi_id_labels(csv_file_path)

    def __len__(self):
        return len(self.wsi_path_list)

    def __getitem__(self, index):
        wsi_id = self.wsi_id_no[index]
        patch_fn_list = self.wsi_patches[wsi_id]
        days, obs = self.wsi_labels[wsi_id]
        y = days // self.interval
        if y >= self.n_intervals - 1:
            y = self.n_intervals - 1

        n = len(patch_fn_list)
        if n >= self.n_patches:
            sel_indices = random.sample(range(n), k=self.n_patches)
            n_sel = len(sel_indices)
        else:
            sel_indices = list(range(n))
            n_sel = n
            
        imgs = torch.zeros((self.n_patches, self.input_nc, 224, 224))
        
        for i, idx in enumerate(sel_indices, 0):
            fn = patch_fn_list[idx]
            wsi_id, img_name = fn.split('/')[-2:]
            img_name = img_name.split('.')[0]
            img_pred = np.array(Image.open(fn))
            H, W, C = img_pred.shape
            h_W = W // 2
            img = img_pred[:, :h_W, :]
            pred = img_pred[:, h_W:, :]
            img = self.transform(img)
            pred = self.transform(pred)
            if self.input_nc == 3:
                imgs[i] = img
            elif self.input_nc == 6:
                imgs[i, :3] = img
                imgs[i, 3:] = pred
            else:
                raise NotImplementedError
        return imgs, y, obs


class WSI_Dataset_Eval(data.Dataset):
    def __init__(self, data_root, csv_file_path, input_nc, transform, mode='test', interval=750, n_intervals=5):
        self.transform = transform
        self.wsi_path_list = glob('{}/{}/*'.format(data_root, mode))
        self.input_nc = input_nc
        self.interval = interval
        self.n_intervals = n_intervals
        self.wsi_id_no = dict()
        self.no_wsi_id = dict()
        self.patches_list = []

        for idx, wsi_path in enumerate(self.wsi_path_list, 0):
            wsi_id = wsi_path.split('/')[-1]
            patch_fn_list = glob('{}/*.png'.format(wsi_path))
            self.wsi_id_no[idx] = wsi_id
            self.no_wsi_id[wsi_id] = idx
            self.patches_list += patch_fn_list

        self.wsi_labels = get_wsi_id_labels(csv_file_path)

    def __len__(self):
        return len(self.patches_list)

    def __getitem__(self, index):
        patch_path = self.patches_list[index]
        wsi_id = patch_path.split('/')[-2]
        no = self.no_wsi_id[wsi_id]

        days, obs = self.wsi_labels[wsi_id]
        y = days // self.interval
        if y >= self.n_intervals - 1:
            y = self.n_intervals - 1
        
        img_pred = np.array(Image.open(patch_path))
        H, W, C = img_pred.shape
        h_W = W // 2
        img = img_pred[:, :h_W, :]
        pred = img_pred[:, h_W:, :]
        img = self.transform(img)
        pred = self.transform(pred)
        if self.input_nc == 3:
            img = img
        elif self.input_nc == 6:
            img = torch.cat((img, pred), dim=0)
        else:
            raise NotImplementedError

        return img, y, obs, wsi_id
    



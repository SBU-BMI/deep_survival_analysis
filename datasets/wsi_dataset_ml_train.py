import os
from glob import glob
from PIL import Image
import random

import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_tile_xy_in_fn, get_tile_wh_in_fn, find_str_in_list
from utils import compute_tile_xys, compute_patch_xys
from data_preprocess.data_preprocess import get_wsi_id_labels
    

class Patch_Data(data.Dataset):
    def __init__(self, wsi_root, nu_seg_root, tumor_pred_root, til_pred_root,
                 data_file_path, mask_root, mode='train', scale=1, round_no=0,
                 n_patches=16, interval=750, n_intervals=5, 
                 rgb_only=False, data_part=3, patch_size=(224,224), tile_size=(4000,4000)):
        # wsi_root: string
        # nu_seg_root: string
        # tumor_pred_root: tuple of strings
        # til_pred_root: tuple of strings
        # mask_root: string
        #     |--patch_size_224_224_scale_1
        #         |--fg_masks  # masks for round_no 0
        #         |--disc_masks_round_1  # masks for round_no > 0
        #         |--disc_masks_round_2  # masks for round_no > 0
        #     |--patch_size_224_224_scale_2
        #         |--fg_masks  # masks for round_no 0
        #         |--disc_masks_round_1  # masks for round_no > 0
        #         |--disc_masks_round_2  # masks for round_no > 0
        #     ...
        # patch_size: (pW, pH)
        # tile_size: (tW, tH)

        assert scale >= 1, 'scale should be greater than or equal to 1'
        assert round_no >= 0, 'round_no should be greater than or equal to 0'
        
        self.round_no = round_no
        self.data_file_path = data_file_path
        self.mask_root = mask_root
        self.scale = scale
        self.patch_size = patch_size
        # get dilated patch size using scale
        self.patch_size_d = (patch_size[0] * scale, patch_size[1] * scale)
        self.tile_size = tile_size
        self.rgb_only = rgb_only
        self.data_part = data_part
        self.n_patches = n_patches
        self.interval = interval
        self.n_intervals = n_intervals

        self.wsi_id_labels = get_wsi_id_labels('{}/dataset_for_survival.csv'.format(data_file_path))
        wsi_id_with_labels = tuple(self.wsi_id_labels.keys())
        wsi_path_list = glob('{0}/*'.format(wsi_root))
        wsi_path_list.sort()  # to fix the training set and testing set
        
        # load file names of wsi tiles (patches)
        self.wsi_tile_paths = dict()
        self.wsi_id_no = dict()  # number (no.) start from 0
        self.tile_geo = dict()
        self.total_disc_patches = 0
        num = 0
        
        for wsi_path in wsi_path_list:
            # wsi_id is the wsi file name ended with "DX1" or "DX2"
            wsi_id = wsi_path.split('/')[-1]
            if len(find_str_in_list(wsi_id_with_labels, wsi_id)) == 0:
                continue
            self.wsi_id_no[wsi_id] = num
            num += 1
            
        self.wsi_id_no = self._get_dataset(mode)
        self.no_wsi_id = self.get_no_wsi_id()
        self.nos = list(self.no_wsi_id.keys())
        
        for wsi_id in self.wsi_id_no.keys():
            wsi_tile_list = glob('{0}/{1}/*.png'.format(wsi_root, wsi_id))
            # sort tile names by x and then by y
            # wsi_tile_list = sorted(wsi_tile_list, key=lambda item: get_tile_xy_in_fn(item)) 
            self.wsi_tile_paths[wsi_id] = tuple(wsi_tile_list)
            
            # load disc mask (mask for discriminative patches), a mask must be binary torch tensor
            # disc_mask = self._load_disc_mask(wsi_id, round_no)
            # n_rows, n_cols = disc_mask.shape
            # n_disc_patches = disc_mask.sum().item()
            # self.total_disc_patches += n_disc_patches
            # self.tile_geo[wsi_id] = (n_cols, n_rows, n_disc_patches)

        # load nuclear segmentations
        self.nu_seg_tile_paths = dict()

        for wsi_id in self.wsi_id_no.keys():
            seg_path = glob('{0}/{1}*'.format(nu_seg_root, wsi_id))[0]
            seg_tile_list = glob('{0}/*.png'.format(seg_path))
            # seg_tile_list = sorted(seg_tile_list, key=lambda item: get_tile_xy_in_fn(item))
            # seg_tile_list = tuple(filter(lambda item: get_tile_wh_in_fn(item) == tile_size, seg_tile_list))
            self.nu_seg_tile_paths[wsi_id] = seg_tile_list
            # assert len(self.nu_seg_tile_paths[wsi_id]) == len(self.wsi_tile_paths[wsi_id]), 'wsi and seg not matched!'

        # load tumor predictions
        self.tumor_pred_tile_paths = dict()
        for wsi_id in self.wsi_id_no.keys():
            tumor_tile_list = []
            for tumor_pred_root_i in tumor_pred_root:
                done_fn = '{0}/{1}/done.txt'.format(tumor_pred_root_i, wsi_id)
                if os.path.isfile(done_fn):
                    tumor_tile_list = glob('{0}/{1}/*_1_INTP.png'.format(tumor_pred_root_i, wsi_id))
                    break
            # tumor_tile_list = sorted(tumor_tile_list, key=lambda item: get_tile_xy_in_fn(item))
            self.tumor_pred_tile_paths[wsi_id] = tuple(tumor_tile_list)
            # assert len(self.tumor_pred_tile_paths[wsi_id]) == len(self.wsi_tile_paths[wsi_id]), 'wsi and tumor pred not matched'

        # load til predictions
        self.til_pred_tile_paths = dict()
        for wsi_id in self.wsi_id_no.keys():
            til_tile_list = []
            for til_pred_root_i in til_pred_root:
                done_fn = '{0}/{1}/done.txt'.format(til_pred_root_i, wsi_id)
                if os.path.isfile(done_fn):
                    til_tile_list = glob('{0}/{1}/*_1_INTP.png'.format(til_pred_root_i, wsi_id))
                    break
            # til_tile_list = sorted(til_tile_list, key=lambda item: get_tile_xy_in_fn(item))
            self.til_pred_tile_paths[wsi_id] = tuple(til_tile_list)
            # assert len(self.til_pred_tile_paths[wsi_id]) == len(self.wsi_tile_paths[wsi_id]), 'wsi and til pred not matched'

    def get_no_wsi_id(self):
        return {v: k for k, v in self.wsi_id_no.items()}

    def get_wsi_id_no(self):
        return self.wsi_id_no

    def get_num_cls(self):
        return self.n_intervals

    def _load_disc_mask(self, wsi_id, round_no=0):
        
        """
        if round_no == 0:
            disc_mask = torch.load(
                '{}/patch_size_{}_{}_scale_{}/fg_masks/{}_fg_mask.pth'
                .format(self.mask_root, self.patch_size[0], self.patch_size[1], self.scale, wsi_id)
            )
        else:
            disc_mask = torch.load(
                '{}/patch_size_{}_{}_scale_{}/disc_masks_round_{}/{}_disc_mask.pth'
                .format(self.mask_root, self.patch_size[0], self.patch_size[1], self.scale, round_no, wsi_id)
            )
        """

        disc_mask = torch.load(
            '{}/patch_size_{}_{}_scale_{}/fg_masks/{}_fg_mask.pth'
            .format(self.mask_root, self.patch_size[0], self.patch_size[1], self.scale, wsi_id)
         )
        
        return disc_mask

    def set_scale(self, scale):
        assert scale >= 1, 'scale should be greater than or equal to 1'
        self.scale = scale
        self.patch_size_d = (self.patch_size[0] * scale, self.patch_size[1] * scale)
        
    def set_round_no(self, round_no):
        # if round_no == 0, disck_mask if the foreground mask
        assert round_no >= 0, 'round_no (round number) should be greater than or equal to 0'
        self.total_disc_patches = 0
        self.round_no = round_no
        for wsi_id in self.wsi_id_no.keys():
            disc_mask = self._load_disc_mask(wsi_id, round_no)
            n_rows, n_cols = disc_mask.shape
            n_disc_patches = disc_mask.sum().item()
            self.total_disc_patches += n_disc_patches
            self.tile_geo[wsi_id] = (n_cols, n_rows, n_disc_patches)

    def _get_dataset(self, mode):
        fn = '{}/{}.txt'.format(self.data_file_path, mode)
        wsi_id_no = dict()
        with open(fn) as file:
            for line in file:
                wsi_id = line.rstrip()
                if wsi_id == 'TCGA-OL-A5D8-01Z-00-DX1' or wsi_id == 'TCGA-E9-A1RB-01Z-00-DX1':  # cannot find this file
                    continue
                wsi_id_no[wsi_id] = self.wsi_id_no[wsi_id]
        return wsi_id_no
    
    def __len__(self):
        # return self.total_disc_patches
        return len(self.wsi_id_no)

    def _get_wsi_id_patch_no(self, index):
        patch_no = index + 1  # index start from 0, patch_no start from 1
        current_no = 0
        for key, val in self.tile_geo.items():
            if current_no < patch_no <= current_no + val[2]:
                n_cols, n_rows = val[:2]
                residual_no = patch_no - current_no
                
                disc_mask = self._load_disc_mask(key, self.round_no)
                csum = disc_mask.t().contiguous().view(-1).cumsum(0)
                # might need to debug if csum is int or not
                disc_no = (csum == residual_no).float().argmax().item() + 1  # start from 1                
                col, row = (disc_no - 1) // n_rows + 1, (disc_no - 1) % n_rows + 1  # col, row start from 1
                x_left, x_right = (col - 1) * self.patch_size_d[0] + 1, col * self.patch_size_d[0] # start from 1
                y_top, y_bottom = (row - 1) * self.patch_size_d[1] + 1, row * self.patch_size_d[1] # start from 1
                
                return key, (col, row), (n_cols, n_rows), (x_left, x_right, y_top, y_bottom)
            
            current_no += val[2]

    def _load_cat_data(self, wsi_id, txy, pxy, data_part=3):
        # data part (int)
        # 0: no data, None
        # 1: rgb data only, 3 channels
        # 2: pred data only, 3 channels
        # 3: (rgb, pred) data, 6 channels
        
        subfn = '/{}_{}_{}_{}'.format(txy[0], txy[1], self.tile_size[0], self.tile_size[1])

        wsi_tile = None
        seg_tile = None
        tumor_tile = None
        til_tile = None

        if data_part == 0:
            return None, None, None, None, None
        
        if data_part == 1 or data_part == 3:
            wsi_tile_path = find_str_in_list(self.wsi_tile_paths[wsi_id], subfn)
            try:
                wsi_tile = np.array(Image.open(wsi_tile_path[0]))
                wsi_patch = wsi_tile[pxy[2]-1:pxy[3], pxy[0]-1:pxy[1], :]
            except:
                patch = np.zeros((6, pxy[3] - pxy[2] + 1, pxy[1] - pxy[0] + 1), dtype=np.uint8)
                wsi_tile = np.zeros((self.tile_size[0], self.tile_size[1], 3), dtype=np.uint8)
                seg_tile = np.zeros((self.tile_size[0], self.tile_size[1], 3), dtype=np.uint8)
                tumor_tile = np.zeros((self.tile_size[0], self.tile_size[1], 3), dtype=np.uint8)
                til_tile = np.zeros((self.tile_size[0], self.tile_size[1], 3), dtype=np.uint8) 
                return patch, wsi_tile, seg_tile, tumor_tile, til_tile

            wsi_patch = wsi_patch / 255.0
            wsi_patch = np.transpose(wsi_patch, (2, 0, 1))

            if data_part == 1:
                return wsi_patch, wsi_tile, seg_tile, tumor_tile, til_tile
        
        if data_part == 2 or data_part == 3:
            # need to debug to check the range of segmentation resutls, whether if its in [0,1]
            seg_tile_path = find_str_in_list(self.nu_seg_tile_paths[wsi_id], subfn)
            if len(seg_tile_path) == 0:
                seg_tile = np.zeros((self.tile_size[0], self.tile_size[1], 3)).astype(np.uint32)
            else:
                seg_tile = np.array(Image.open(seg_tile_path[0]))
            seg_patch = seg_tile[pxy[2]-1:pxy[3], pxy[0]-1:pxy[1], 1]
            seg_patch = seg_patch / 255.0
            seg_patch = seg_patch[np.newaxis, ...]

            tumor_tile_path = find_str_in_list(self.tumor_pred_tile_paths[wsi_id], subfn)
            if len(tumor_tile_path) == 0:
                tumor_tile = np.zeros(self.tile_size)
            else:
                tumor_tile = np.array(Image.open(tumor_tile_path[0]))
            tumor_patch = tumor_tile[pxy[2]-1:pxy[3], pxy[0]-1:pxy[1]]
            tumor_patch = tumor_patch / 255.0
            tumor_patch = tumor_patch[np.newaxis, ...]

            til_tile_path = find_str_in_list(self.til_pred_tile_paths[wsi_id], subfn)
            if len(til_tile_path) == 0:
                til_tile = np.zeros(self.tile_size)
            else:
                til_tile = np.array(Image.open(til_tile_path[0]))
            til_patch = til_tile[pxy[2]-1:pxy[3], pxy[0]-1:pxy[1]]
            til_patch = til_patch / 255.0
            til_patch = til_patch[np.newaxis, ...]

            pred_patch = np.concatenate((seg_patch, tumor_patch, til_patch), axis=0)
            
            if data_part == 2:
                return pred_patch, wsi_tile, seg_tile, tumor_tile, til_tile

        if data_part == 3:
            patch = np.concatenate((wsi_patch, pred_patch), axis=0)
            return patch, wsi_tile, seg_tile, tumor_tile, til_tile

    def _get_labels(self, obs, y):
        label = torch.zeros((self.n_intervals, ))
        if obs == 1:
            label[y] = 1
        else:
            if y == self.n_intervals - 1:
                label[y] = 1
            else:
                for i in range(y+1, self.n_intervals):
                    label[i] = 1
        return label

    def __getitem__(self, index_):
        no = self.nos[index_]
        wsi_id = self.no_wsi_id[no]
        num_patches = self.tile_geo[wsi_id][2]
        offset = 0

        days, obs = self.wsi_id_labels[wsi_id]
        y = days // self.interval
        if y >= self.n_intervals - 1:
            y = self.n_intervals - 1

        label = self._get_labels(obs, y)

        if self.data_part == 0:
            return y, obs, wsi_id, label
        
        for key, val in self.tile_geo.items():
            if key != wsi_id:
                offset += val[2]
            else:
                break

        if self.n_patches > num_patches:
            sel_indices = random.sample(range(num_patches), k=num_patches)
        else:
            sel_indices = random.sample(range(num_patches), k=self.n_patches)

        indices = sel_indices
        
        for idx, val in enumerate(indices, 0):
            indices[idx] += offset
        
        if self.rgb_only:
            imgs = torch.zeros((self.n_patches, 3, self.patch_size[0], self.patch_size[1]))
        else:
            if self.data_part == 3:
                imgs = torch.zeros((self.n_patches, 6, self.patch_size[0], self.patch_size[1]))
            else:
                imgs = torch.zeros((self.n_patches, 3, self.patch_size[0], self.patch_size[1]))

        for idx, index in enumerate(indices, 0):
            wsi_id, (col, row), (n_cols, n_rows), pxy = self._get_wsi_id_patch_no(index)
            txy_00, txy_01, txy_10, txy_11 = compute_tile_xys(pxy, self.tile_size)
            cross_status, pxy_00, pxy_01, pxy_10, pxy_11 = compute_patch_xys(pxy, self.patch_size_d, self.tile_size)

            if cross_status == "00":
                patch_00, _, _, _, _ = self._load_cat_data(wsi_id, txy_00, pxy_00, self.data_part)
                patch = patch_00
            elif cross_status == "01":
                patch_00, _, _, _, _ = self._load_cat_data(wsi_id, txy_00, pxy_00, self.data_part)
                patch_01, _, _, _, _ = self._load_cat_data(wsi_id, txy_01, pxy_01, self.data_part)
                patch = np.concatenate((patch_00, patch_01), axis=2)
            elif cross_status == "10":
                patch_00, _, _, _, _ = self._load_cat_data(wsi_id, txy_00, pxy_00, self.data_part)
                patch_10, _, _, _, _ = self._load_cat_data(wsi_id, txy_10, pxy_10, self.data_part)
                patch = np.concatenate((patch_00, patch_10), axis=1)
            elif cross_status == "11":
                patch_00, _, _, _, _ = self._load_cat_data(wsi_id, txy_00, pxy_00, self.data_part)
                patch_01, _, _, _, _ = self._load_cat_data(wsi_id, txy_01, pxy_01, self.data_part)
                patch_10, _, _, _, _ = self._load_cat_data(wsi_id, txy_10, pxy_10, self.data_part)
                patch_11, _, _, _, _ = self._load_cat_data(wsi_id, txy_11, pxy_11, self.data_part)
                try:
                    patch_0 = np.concatenate((patch_00, patch_01), axis=2)
                    patch_1 = np.concatenate((patch_10, patch_11), axis=2)
                    patch = np.concatenate((patch_0, patch_1), axis=1)
                except:
                    import pdb
                    pdb.set_trace()
            else:
                raise NotImplementedError

            try:
                patch = patch[:, ::self.scale, ::self.scale]  # down-size retrieved patch
            except:
                import pdb
                pdb.set_trace()
            patch = (torch.from_numpy(patch).float() - 0.5) / 0.5  # self.transform(patch)
            if self.rgb_only:
                imgs[idx] = patch
            else:
                if self.data_part == 3:
                    imgs[idx, :3] = patch[:3]
                    imgs[idx, 3:] = patch[3:]
                else:
                    imgs[idx] = patch
                        
        return imgs, y, obs, label
        


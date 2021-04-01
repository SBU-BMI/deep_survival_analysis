import os
from glob import glob
from PIL import Image

import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_tile_xy_in_fn, get_tile_wh_in_fn, find_str_in_list
from utils import compute_tile_xys, compute_patch_xys
from data_preprocess.data_preprocess import get_wsi_id_labels
    

class Patch_Data(data.Dataset):
    def __init__(self, wsi_root, nu_seg_root, tumor_pred_root, til_pred_root,
                 label_file, transform, mask_root, scale=1, round_no=0,
                 is_train=True, train_ratio=0.8, rgb_only=False,
                 patch_size=(224,224), tile_size=(4000,4000)):
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
        self.mask_root = mask_root
        self.scale = scale
        self.patch_size = patch_size
        # get dilated patch size using scale
        self.patch_size_d = (patch_size[0] * scale, patch_size[1] * scale)
        self.tile_size = tile_size
        self.transform = transform
        self.rgb_only = rgb_only

        self.wsi_id_labels = get_wsi_id_labels(label_file)
        self.wsi_labels = tuple(self.wsi_id_labels.values())
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
            
        self.wsi_id_no = self._split_dataset(is_train, train_ratio)
        
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
                    tumor_tile_list = glob('{0}/{1}/*_1_INTP.npy'.format(tumor_pred_root_i, wsi_id))
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
                    til_tile_list = glob('{0}/{1}/*_1_INTP.npy'.format(til_pred_root_i, wsi_id))
                    break
            # til_tile_list = sorted(til_tile_list, key=lambda item: get_tile_xy_in_fn(item))
            self.til_pred_tile_paths[wsi_id] = tuple(til_tile_list)
            # assert len(self.til_pred_tile_paths[wsi_id]) == len(self.wsi_tile_paths[wsi_id]), 'wsi and til pred not matched'

    def get_no_wsi_id(self):
        return {v: k for k, v in self.wsi_id_no.items()}

    def get_wsi_id_no(self):
        return self.wsi_id_no

    def get_num_cls(self):
        return max(self.wsi_labels) + 1  # number of classes 

    def _load_disc_mask(self, wsi_id, round_no=0):
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

    def _split_dataset(self, is_train, train_ratio):
        import operator
        wsi_id_no_sel = dict()

        for c in self.wsi_labels:
            # get subdictionary for class c
            wsi_id_no_c = {k: v for k, v in self.wsi_id_no.items() if self.wsi_id_labels[k] == c}
            num_c = len(wsi_id_no_c)
            num_train_c = int(num_c * train_ratio)
            if is_train:
                wsi_id_no_c_sel = dict(sorted(wsi_id_no_c.items(), key=operator.itemgetter(0))[:num_train_c])
            else:
                wsi_id_no_c_sel = dict(sorted(wsi_id_no_c.items(), key=operator.itemgetter(0))[num_train_c:])
            wsi_id_no_sel.update(wsi_id_no_c_sel)

        return wsi_id_no_sel
    
    def __len__(self):
        return self.total_disc_patches

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
                disc_no = (csum == residual_no).argmax().item() + 1  # start from 1                
                col, row = (disc_no - 1) // n_rows + 1, (disc_no - 1) % n_rows + 1  # col, row start from 1
                x_left, x_right = (col - 1) * self.patch_size_d[0] + 1, col * self.patch_size_d[0] # start from 1
                y_top, y_bottom = (row - 1) * self.patch_size_d[1] + 1, row * self.patch_size_d[1] # start from 1
                
                return key, (col, row), (n_cols, n_rows), (x_left, x_right, y_top, y_bottom)
            
            current_no += val[2]

    def _load_cat_data(self, wsi_id, txy, pxy):
        subfn = '/{}_{}_{}_{}'.format(txy[0], txy[1], self.tile_size[0], self.tile_size[1])

        wsi_tile_path = find_str_in_list(self.wsi_tile_paths[wsi_id], subfn)
        wsi_tile = np.array(Image.open(wsi_tile_path[0]))
        wsi_patch = wsi_tile[pxy[2]-1:pxy[3], pxy[0]-1:pxy[1], :]
        wsi_patch = wsi_patch / 255.0
        wsi_patch = np.transpose(wsi_patch, (2, 0, 1))
        patch = wsi_patch

        seg_tile = None
        tumor_tile = None
        til_tile = None
        
        if not self.rgb_only:
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
                tumor_tile = np.load(tumor_tile_path[0])
            tumor_patch = tumor_tile[pxy[2]-1:pxy[3], pxy[0]-1:pxy[1]]
            tumor_patch = tumor_patch[np.newaxis, ...]

            til_tile_path = find_str_in_list(self.til_pred_tile_paths[wsi_id], subfn)
            if len(til_tile_path) == 0:
                til_tile = np.zeros(self.tile_size)
            else:
                til_tile = np.load(til_tile_path[0])
            til_patch = til_tile[pxy[2]-1:pxy[3], pxy[0]-1:pxy[1]]
            til_patch = til_patch[np.newaxis, ...]

            patch = np.concatenate((wsi_patch, seg_patch, tumor_patch, til_patch), axis=0)

        return patch, wsi_tile, seg_tile, tumor_tile, til_tile

    def __getitem__(self, index):
        wsi_id, (col, row), (n_cols, n_rows), pxy = self._get_wsi_id_patch_no(index)
        txy_00, txy_01, txy_10, txy_11 = compute_tile_xys(pxy, self.tile_size)
        cross_status, pxy_00, pxy_01, pxy_10, pxy_11 = compute_patch_xys(pxy, self.patch_size_d, self.tile_size)

        if cross_status == "00":
            patch_00, _, _, _, _ = self._load_cat_data(wsi_id, txy_00, pxy_00)
            patch = patch_00
        elif cross_status == "01":
            patch_00, _, _, _, _ = self._load_cat_data(wsi_id, txy_00, pxy_00)
            patch_01, _, _, _, _ = self._load_cat_data(wsi_id, txy_01, pxy_01)
            patch = np.concatenate((patch_00, patch_01), axis=2)
        elif cross_status == "10":
            patch_00, _, _, _, _ = self._load_cat_data(wsi_id, txy_00, pxy_00)
            patch_10, _, _, _, _ = self._load_cat_data(wsi_id, txy_10, pxy_10)
            patch = np.concatenate((patch_00, patch_10), axis=1)
        elif cross_status == "11":
            patch_00, _, _, _, _ = self._load_cat_data(wsi_id, txy_00, pxy_00)
            patch_01, _, _, _, _ = self._load_cat_data(wsi_id, txy_01, pxy_01)
            patch_10, _, _, _, _ = self._load_cat_data(wsi_id, txy_10, pxy_10)
            patch_11, _, _, _, _ = self._load_cat_data(wsi_id, txy_11, pxy_11)
            patch_0 = np.concatenate((patch_00, patch_01), axis=2)
            patch_1 = np.concatenate((patch_10, patch_11), axis=2)
            patch = np.concatenate((patch_0, patch_1), axis=1)
        else:
            raise NotImplementedError

        patch = patch[:, ::self.scale, ::self.scale]  # down-size retrieved patch
        patch = (torch.from_numpy(patch).float() - 0.5) / 0.5 # self.transform(patch)
        # needs to debug to see label type
        label = torch.tensor(self.wsi_id_labels[wsi_id])
        col = torch.tensor(col)  # index start from 1
        row = torch.tensor(row)  # index start from 1
        wsi_id_no = torch.tensor(self.wsi_id_no[wsi_id])  # index start from 0
        
        return patch, label, col, row, n_cols, n_rows, wsi_id_no
        
    def sequential_loader(self):

        def _reset_cache():
            use_cache = \
                cache_wsi_id == wsi_id and \
                cache_cross_status == cross_status and \
                cache_txy_00 == txy_00 and \
                cache_txy_01 == txy_01 and \
                cache_txy_10 == txy_10 and \
                cache_txy_11 == txy_11

            if not use_cache:
                cache_cross_status = cross_status
                cache_txy_00 = txy_00
                cache_txy_01 = txy_01
                cache_txy_10 = txy_10
                cache_txy_11 = txy_11
                cache_wsi_tile_00 = None
                cache_wsi_tile_01 = None
                cache_wsi_tile_10 = None
                cache_wsi_tile_11 = None
                cache_seg_tile_00 = None
                cache_seg_tile_01 = None
                cache_seg_tile_10 = None
                cache_seg_tile_11 = None
                cache_tumor_tile_00 = None
                cache_tumor_tile_01 = None
                cache_tumor_tile_10 = None
                cache_tumor_tile_11 = None
                cache_til_tile_00 = None
                cache_til_tile_01 = None
                cache_til_tile_10 = None
                cache_til_tile_11 = None

        def _get_cat_cached_patch(quadrant, pxy_00, pxy_01, pxy_10, pxy_11):
            if quadrant == "00":
                wsi_patch = cache_wsi_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                seg_patch = cache_seg_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                tumor_patch = cache_tumor_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                til_patch = cache_til_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
            elif quadrant == "01":
                wsi_patch = cache_wsi_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                seg_patch = cache_seg_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                tumor_patch = cache_tumor_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                til_patch = cache_til_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
            elif quadrant == "10":
                wsi_patch = cache_wsi_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                seg_patch = cache_seg_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                tumor_patch = cache_tumor_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                til_patch = cache_til_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
            elif quadrant == "11":
                wsi_patch = cache_wsi_tile_11[pxy_11[2]-1:pxy_11[3], pxy_11[0]-1:pxy_11[1], :]
                seg_patch = cache_seg_tile_11[pxy_11[2]-1:pxy_11[3], pxy_11[0]-1:pxy_11[1], :]
                tumor_patch = cache_tumor_tile_11[pxy_11[2]-1:pxy_11[3], pxy_11[0]-1:pxy_11[1], :]
                til_patch = cache_til_tile_11[pxy_11[2]-1:pxy_11[3], pxy_11[0]-1:pxy_11[1], :]
            else:
                raise NotImplementedError
            
            patch = np.concatenate((wsi_patch, seg_patch, tumor_patch, til_patch), axis=0)
            
            return patch

        for wsi_id in self.wsi_id_no.keys():
            # needs to debug to see label type
            label = torch.tensor(self.wsi_id_labels[wsi_id])
            col = torch.tensor(col)  # index start from 1
            row = torch.tensor(row)  # index start from 1
            wsi_id_no = torch.tensor(self.wsi_id_no[wsi_id])  # index start from 0

            wsi_tile_list = glob('{0}/{1}/*.png'.format(self.wsi_root, wsi_id))
            # sort tile names by x and then by y
            # wsi_tile_list = sorted(wsi_tile_list, key=lambda item: get_tile_xy_in_fn(item)) 

            # get wsi W and H using the last tile file name
            #  final_tile_basename = wsi_tile_list[-1].split('/')[-1].split('.')[0]
            start_x, start_y = get_max_xy_in_paths(wsi_tile_list)
            tw, th = self.tile_size
            wsi_W, wsi_H = int(start_x) + int(tw) - 1, int(start_y) + int(th) - 1
            # get number of rows and columns for patches (a patch is used to train a CNN)
            n_cols, n_rows = wsi_W // self.patch_size_d[0], wsi_H // self.patch_size_d[1]
            fg_mask = torch.zeros((n_rows, n_cols), dtype=torch.uint8)

            cache_wsi_tile_00 = None
            cache_wsi_tile_01 = None
            cache_wsi_tile_10 = None
            cache_wsi_tile_11 = None
            cache_seg_tile_00 = None
            cache_seg_tile_01 = None
            cache_seg_tile_10 = None
            cache_seg_tile_11 = None
            cache_tumor_tile_00 = None
            cache_tumor_tile_01 = None
            cache_tumor_tile_10 = None
            cache_tumor_tile_11 = None
            cache_til_tile_00 = None
            cache_til_tile_01 = None
            cache_til_tile_10 = None
            cache_til_tile_11 = None

            for col in range(1, n_cols + 1):
                for row in range(1, n_rows + 1):
                    x_left, x_right = (col - 1) * self.patch_size_d[0] + 1, col * self.patch_size_d[0]  # start from 1
                    y_top, y_bottom = (row - 1) * self.patch_size_d[1] + 1, row * self.patch_size_d[1]  # start from 1
                    pxy = (x_left, x_right, y_top, y_bottom)
                    txy_00, txy_01, txy_10, txy_11 = compute_tile_xys(pxy, self.tile_size)
                    cross_status, pxy_00, pxy_01, pxy_10, pxy_11 = compute_patch_xys(pxy, self.patch_size_d, self.tile_size)

                    use_cache = _reset_cache()
                    if cross_status == "00":
                        if use_cache:
                            if self.rgb_only:
                                patch_00 = cache_wsi_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                            else:
                                patch_00 = _get_cat_cached_patch("00", pxy_00, pxy_01, pxy_10, pxy_11)
                        else:
                            patch_00, cache_wsi_tile_00, cache_seg_tile_00, cache_tumor_tile_00, cache_til_tile_00 \
                                = self._load_cat_data(wsi_tile_list, txy_00, pxy_00)
                        patch = patch_00
                    elif cross_status == "01":
                        if use_cache:
                            if self.rgb_only:
                                patch_00 = cache_wsi_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                                patch_01 = cache_wsi_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                            else:
                                patch_00 = _get_cat_cached_patch("00", pxy_00, pxy_01, pxy_10, pxy_11)
                                patch_01 = _get_cat_cached_patch("01", pxy_00, pxy_01, pxy_10, pxy_11)
                        else:
                            patch_00, cache_wsi_tile_00, cache_seg_tile_00, cache_tumor_tile_00, cache_til_tile_00 \
                                = self._load_cat_data(wsi_tile_list, txy_00, pxy_00)
                            patch_01, cache_wsi_tile_01, cache_seg_tile_01, cache_tumor_tile_01, cache_til_tile_01 \
                                = self._load_cat_data(wsi_tile_list, txy_01, pxy_01)
                        patch = np.concatenate((patch_00, patch_01), axis=2)
                    elif cross_status == "10":
                        if use_cache:
                            if self.rgb_only:
                                patch_00 = cache_wsi_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                                patch_10 = cache_wsi_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                            else:
                                patch_00 = _get_cat_cached_patch("00", pxy_00, pxy_01, pxy_10, pxy_11)
                                patch_10 = _get_cat_cached_patch("10", pxy_00, pxy_01, pxy_10, pxy_11)
                        else:
                            patch_00, cache_wsi_tile_00, cache_seg_tile_00, cache_tumor_tile_00, cache_til_tile_00 \
                                = self._load_cat_data(wsi_tile_list, txy_00, pxy_00)
                            patch_10, cache_wsi_tile_10, cache_seg_tile_10, cache_tumor_tile_10, cache_til_tile_10 \
                                = self._load_cat_data(wsi_tile_list, txy_10, pxy_10)
                        patch = np.concatenate((patch_00, patch_10), axis=1)
                    elif cross_status == "11":
                        if use_cache:
                            if self.rgb_only:
                                patch_00 = cache_wsi_tile_00[pxy_00[2]-1:pxy_00[3], pxy_00[0]-1:pxy_00[1], :]
                                patch_01 = cache_wsi_tile_01[pxy_01[2]-1:pxy_01[3], pxy_01[0]-1:pxy_01[1], :]
                                patch_10 = cache_wsi_tile_10[pxy_10[2]-1:pxy_10[3], pxy_10[0]-1:pxy_10[1], :]
                                patch_11 = cache_wsi_tile_11[pxy_11[2]-1:pxy_11[3], pxy_11[0]-1:pxy_11[1], :]
                            else:
                                patch_00 = _get_cat_cached_patch("00", pxy_00, pxy_01, pxy_10, pxy_11)
                                patch_01 = _get_cat_cached_patch("01", pxy_00, pxy_01, pxy_10, pxy_11)
                                patch_10 = _get_cat_cached_patch("10", pxy_00, pxy_01, pxy_10, pxy_11)
                                patch_11 = _get_cat_cached_patch("11", pxy_00, pxy_01, pxy_10, pxy_11)
                        else:
                            patch_00, cache_wsi_tile_00, cache_seg_tile_00, cache_tumor_tile_00, cache_til_tile_00 \
                                = self._load_cat_data(wsi_tile_list, txy_00, pxy_00)
                            patch_01, cache_wsi_tile_01, cache_seg_tile_01, cache_tumor_tile_01, cache_til_tile_01 \
                                = self._load_cat_data(wsi_tile_list, txy_01, pxy_01)
                            patch_10, cache_wsi_tile_10, cache_seg_tile_10, cache_tumor_tile_10, cache_til_tile_10 \
                                = self._load_cat_data(wsi_tile_list, txy_10, pxy_10)
                            patch_11, cache_wsi_tile_11, cache_seg_tile_11, cache_tumor_tile_11, cache_til_tile_11 \
                                = self._load_cat_data(wsi_tile_list, txy_11, pxy_11)
                        patch_0 = np.concatenate((patch_00, patch_01), axis=2)
                        patch_1 = np.concatenate((patch_10, patch_11), axis=2)
                        patch = np.concatenate((patch_0, patch_1), axis=1)
                    else:
                        raise NotImplementedError
                    
                    patch = patch[:, ::self.scale, ::self.scale]  # down-size retrieved patch
                    patch = self.transform(patch)

                    yield patch, label, col, row, n_cols, n_rows, wsi_id_no



import os
import argparse
from glob import glob
from PIL import Image

import numpy as np
import openslide


parser = argparse.ArgumentParser()
parser.add_argument('--wsi-dir', type=str, default='/data03/tcga_data/tumor/brca', help='dir to WSIs')
parser.add_argument('--patch-dir', required=True, help='dir to patches')
parser.add_argument('--patch-pred-dir', required=True, help='dir to predicted patches')
parser.add_argument('--patch_H', type=int, default=224, help='patch height')
parser.add_argument('--patch_W', type=int, default=224, help='patch width')
parser.add_argument('--output-dir', type=str, default='.', help='output dir')

args = parser.parse_args()


# a function to change the size of image
def changeImageSize(maxWidth, maxHeight, image):
    """
        Takes a PIL image and resizes it according to the given
        width and height
    """
    widthRatio = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth = int(widthRatio*image.size[0])
    newHeight = int(heightRatio*image.size[1])
    newImage = image.resize((newWidth, newHeight), Image.BILINEAR)
    
    return newImage


# a function to downsample the WSI image
def downsampleWSI(slide, hm_shape):
    """
        Takes a WSI image and returns a PIL image downsampled 
        using the given heatmap shape
    """
    
    width, height = slide.dimensions
    scale_factor = min((width // hm_shape[1]), (height // hm_shape[0]))
    
    # try to create the level_width and height manually by dividing by scale_factor
    level = slide.get_best_level_for_downsample(scale_factor)
    downsample_factor = int(slide.level_downsamples[level])
    
    level_width, level_height = slide.level_dimensions[level]
    level_width -= (((width % 4000) // downsample_factor))
    level_height -= (((height % 4000) // downsample_factor))
    
    whole_slide_image = slide.read_region((0, 0), level, (level_width, level_height))
    whole_slide_image = whole_slide_image.convert("RGB")
    img = changeImageSize(hm_shape[1], hm_shape[0], whole_slide_image)
    
    return img


patch_H, patch_W = int(args.patch_H), int(args.patch_W)

patch_dir = args.patch_dir
patch_pred_dir = args.patch_pred_dir
output_dir = args.output_dir

patch_list = glob('{}/*.png'.format(patch_dir))
num_coor = len(patch_list)

coor = np.zeros((num_coor, 2))

for idx, img_path in enumerate(patch_list, 0):
    x, y = img_path.split('/')[-1].split('_')[1:3]
    coor[idx, 0] = int(x)
    coor[idx, 1] = int(y)

x_max, y_max = np.max(coor, axis=0)

x_max += patch_W - 1
y_max += patch_H - 1
x_max = int(x_max // patch_W)
y_max = int(y_max // patch_H)

img_ds = np.zeros((y_max, x_max, 3), dtype=np.uint8)
nuseg_org = np.zeros((y_max, x_max), dtype=np.uint8)
tumor_org = np.zeros((y_max, x_max), dtype=np.uint8)
til_org = np.zeros((y_max, x_max), dtype=np.uint8)

nuseg_pred = np.zeros((y_max, x_max), dtype=np.uint8)
tumor_pred = np.zeros((y_max, x_max), dtype=np.uint8)
til_pred = np.zeros((y_max, x_max), dtype=np.uint8)

wsi_id = None
for idx, patch_path in enumerate(patch_list, 0):
    fn = patch_path.split('/')[-1].split('.')[0]
    wsi_id, x_start, y_start, _, _ = fn.split('_')
    x_start, y_start = int(x_start), int(y_start)
    col = (x_start - 1) // patch_W
    row = (y_start - 1) // patch_H
    img_label = np.array(Image.open(patch_path))
    H, W, _ = img_label.shape
    half_W = W // 2
    img = img_label[:, :half_W, :]
    img_ds[row, col, 0] = np.mean(img[:, :, 0])
    img_ds[row, col, 1] = np.mean(img[:, :, 1])
    img_ds[row, col, 2] = np.mean(img[:, :, 2])
    label = img_label[:, half_W:, :]
    nuseg_org[row, col] = np.mean(label[:, :, 0])
    tumor_org[row, col] = np.mean(label[:, :, 1])
    til_org[row, col] = np.mean(label[:, :, 2])

    pred_patch_path = '{}/{}_fake_B.png'.format(patch_pred_dir, fn)
    pred_img_label = np.array(Image.open(pred_patch_path))
    pred_H, pred_W, _ = pred_img_label.shape
    pred_half_W = pred_W // 2
    pred_label = pred_img_label[:, pred_half_W:, :]
    nuseg_pred[row, col] = np.mean(pred_label[:, :, 0])
    tumor_pred[row, col] = np.mean(pred_label[:, :, 1])
    til_pred[row, col] = np.mean(pred_label[:, :, 2])

    if idx % 10 == 0:
        print('{}/{} done!'.format(idx, num_coor))

wsi_dir = args.wsi_dir
wsi_path = glob('{}/{}*.svs'.format(wsi_dir, wsi_id))
wsi_path = wsi_path[0]

try:
    slide = openslide.OpenSlide(wsi_path)
except OpenSlideError:
    slide = None
    print("OpenSlideError")
except FileNotFoundError:
    slide = None
    print("FileNotFoundError")

small_wsi = downsampleWSI(slide, (y_max, x_max))
img_ds = small_wsi.convert("RGBA").save('{}/img_org_ds.png'.format(output_dir))

Image.fromarray(nuseg_org).save('{}/nuseg_org_ds.png'.format(output_dir))
Image.fromarray(tumor_org).save('{}/tumor_org_ds.png'.format(output_dir))
Image.fromarray(til_org).save('{}/til_org_ds.png'.format(output_dir))

Image.fromarray(nuseg_pred).save('{}/nuseg_pred_ds.png'.format(output_dir))
Image.fromarray(tumor_pred).save('{}/tumor_pred_ds.png'.format(output_dir))
Image.fromarray(til_pred).save('{}/til_pred_ds.png'.format(output_dir))



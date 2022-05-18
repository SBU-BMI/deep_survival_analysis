import os
import sys
import argparse
from glob import glob

import json
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy.interpolate import interp2d
import openslide

from utils import ensure_dir


def select_preds(pred, x_left, x_right, y_top, y_bottom):
    if x_left <= pred[0] <= x_right and y_top <= pred[1] <= y_bottom:
        return True
    else:
        return False


def ensure_cors(x_cors, y_cors, Z, x_min, x_max, y_min, y_max, x, y, pw, ph):
    if x_min > x:
        x_cors = np.insert(x_cors, 0, x)
        Z = np.insert(Z, 0, Z[:, 0], axis=1)
        x_min = x

    if x_max < x + pw:
        x_cors = np.insert(x_cors, x_cors.shape[0], x + pw)
        Z = np.insert(Z, Z.shape[1], Z[:, -1], axis=1)
        x_max = x + pw

    if y_min > y:
        y_cors = np.insert(y_cors, 0, y)
        Z = np.insert(Z, 0, Z[0, :], axis=0)
        y_min = y

    if y_max < y + ph:
        y_cors = np.insert(y_cors, y_cors.shape[0], y + ph)
        Z = np.insert(Z, Z.shape[0], Z[-1, :], axis=0)
        y_max = y + ph

    return x_cors, y_cors, x_min, x_max, y_min, y_max, Z


def interp(arguments):
    WSI_path, pred_path, output_dir, patch_size = arguments
    # some code is adopted from Le's code
    done_fn = '{}/done.txt'.format(output_dir)
    if os.path.isfile(done_fn):
        return None
    
    margin = 5
    margin_sp = 10
    try:
        oslide = openslide.OpenSlide(WSI_path)
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
            mpp = float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
        elif "XResolution" in oslide.properties:
            mpp = float(oslide.properties["XResolution"]);
        elif "tiff.XResolution" in oslide.properties:
            mpp = float(oslide.properties["tiff.XResolution"]);
        else:
            mpp = 0.250

        width = oslide.dimensions[0]
        height = oslide.dimensions[1]
    except:
        print('Error in {}: exception caught exiting'.format(WSI_path))
        raise Exception('{}: exception caught exiting'.format(WSI_path))

    pw, ph = patch_size

    # Load prediction results
    cor_prob = []

    with open(pred_path) as file:
        for line in file:
            x, y, prob, _ = line.split(" ")
            x, y, prob = int(x), int(y), float(prob)
            cor_prob.append((x, y, prob))

    # cor_prob = sorted(cor_prob, key=lambda item: (item[0], item[1]))
    cor_prob_np = np.array(cor_prob)
    x_cors_all = np.unique(cor_prob_np[:, 0].astype(np.uint32))
    y_cors_all = np.unique(cor_prob_np[:, 1].astype(np.uint32))
    x_3_res = np.argpartition(x_cors_all, 3)
    y_3_res = np.argpartition(y_cors_all, 3)
    x_3_smallest = np.sort(x_cors_all[x_3_res[:3]])
    y_3_smallest = np.sort(y_cors_all[y_3_res[:3]])
    
    n_cors = len(cor_prob)
    spw = x_3_smallest[2] - x_3_smallest[1]
    sph = y_3_smallest[2] - y_3_smallest[1]

    for x in range(1, width, pw):
        for y in range(1, height, ph):
            
             # the segmentation results do not have sizes that are < patch_sizes
            if x + pw > width - margin:
                continue
            if y + ph > height - margin:
                continue

            if pw <= 3 or ph <= 3:
                continue
            
            x_left = max(x - spw - margin_sp, 1)
            x_right = min(x + pw + spw + margin_sp, width)
            y_top = max(y - sph - margin_sp, 1)
            y_bottom = min(y + ph + sph + margin_sp, height)
            selected_cors = filter(lambda item: select_preds(item, x_left, x_right, y_top, y_bottom), cor_prob)
            selected_cors = list(selected_cors)
            selected_cors = sorted(selected_cors, key=lambda item: (item[0], item[1]))
            selected_cors = np.array(selected_cors)
            
            if selected_cors.shape[0] > 0:
                x_cors = np.sort(np.unique(selected_cors[:, 0].astype(np.uint32)))
                y_cors = np.sort(np.unique(selected_cors[:, 1].astype(np.uint32)))
                nx, ny = x_cors.shape[0], y_cors.shape[0]
                if ny * nx == selected_cors.shape[0]:
                    Z = np.reshape(selected_cors[:, 2], (ny, nx), order='F')
                else:
                    Z = np.zeros((ny, nx)).astype(np.float16)
                    d_x = dict(zip(x_cors, range(nx)))
                    d_y = dict(zip(y_cors, range(ny)))
                    for i in range(selected_cors.shape[0]):
                        row = d_y[int(selected_cors[i, 1])]
                        col = d_x[int(selected_cors[i, 0])]
                        Z[row, col] = selected_cors[i, 2]

                x_min, x_max = np.min(x_cors), np.max(x_cors)
                y_min, y_max = np.min(y_cors), np.max(y_cors)
                x_cors, y_cors, x_min, x_max, y_min, y_max, Z = ensure_cors(x_cors, y_cors, Z, x_min, x_max, y_min, y_max, x, y, pw, ph)

                if x_cors.shape[0] < 16 or y_cors.shape[0] < 16:
                    f = interp2d(x_cors, y_cors, Z, kind='linear')
                else:
                    f = interp2d(x_cors, y_cors, Z, kind='cubic')
                x2 = np.linspace(x_min, x_max, x_max - x_min + 1)
                y2 = np.linspace(y_min, y_max, y_max - y_min + 1)
                Z2 = f(x2, y2)
                Z2 = np.clip(Z2, 0, 1)  # clip to [0, 1]
                ind_x_start = int(x - x_min)
                ind_y_start = int(y - y_min)
                Z2_cropped = Z2[ind_y_start : ind_y_start+ph, ind_x_start : ind_x_start+pw].astype(np.float16)
                Z.astype(np.float16)
            else:
                Z = np.zeros((1,1)).astype(np.float16)
                Z2_cropped = np.zeros((ph, pw)).astype(np.float16)

            """
            fn_pre = "{}/{}_{}_{}_{}_{}_1_PREINTP.npy".format(output_dir, x, y, pw, ph, mpp)
            fn = "{}/{}_{}_{}_{}_{}_1_INTP.npy".format(output_dir, x, y, pw, ph, mpp)
            
            np.save(fn_pre, Z)
            np.save(fn, Z2_cropped)
            """
            fn_pre = "{}/{}_{}_{}_{}_{}_1_PREINTP.png".format(output_dir, x, y, pw, ph, mpp)
            fn = "{}/{}_{}_{}_{}_{}_1_INTP.png".format(output_dir, x, y, pw, ph, mpp)

            Image.fromarray((Z * 255).astype(np.uint8)).save(fn_pre)
            Image.fromarray((Z2_cropped * 255).astype(np.uint8)).save(fn)
            
            # print("{}_{}_{}_{}_{}_1_INTP done!".format(x, y, pw, ph, mpp))

    with open(done_fn, 'a') as f:
        f.write(WSI_path)
        f.write('\n width: {}, height: {}\n'.format(width, height))
    # print('{} done!'.format(WSI_path))


def query_fn(wsi_id, path_list):
    for path in path_list:
        basename = path.split('/')[-1]
        if basename.find(wsi_id) != -1:
            return path
    return None    


def interp_main(config, start_idx, end_idx, ncores):
    wsi_root = config['WSIs']['root_path']
    nuclei_seg_root = config['Nuclei_segs']['root_path']
    tumor_preds_root = config['Tumor_preds']['root_path']
    til_preds_root = config['TIL_preds']['root_path']

    # wsi_output_path = config['WSIs']['output_path']
    # nuclei_segs_output_path = config['Nuclei_segs']['output_path']
    tumor_preds_output_path = config['Tumor_preds']['output_path']
    til_preds_output_path = config['TIL_preds']['output_path']
    
    wsi_path_list = glob('{}/*.svs'.format(wsi_root))
    nuclei_segs_path_list = glob('{}/*.svs'.format(nuclei_seg_root))
    tumor_preds_path_list = glob('{}/prediction-*[!.low_res]'.format(tumor_preds_root))
    til_preds_path_list = glob('{}/prediction-*[!.low_res]'.format(til_preds_root))

    tasks = []
    
    for wsi_path in wsi_path_list:
        wsi_fn = wsi_path.split('/')[-1]
        wsi_id = wsi_fn.split('.')[0]
        # not diagnoised
        if wsi_id[-3:-1].lower() != 'dx':
            continue

        nuclei_segs_path = query_fn(wsi_id, nuclei_segs_path_list)
        if nuclei_segs_path is None:
            continue
        
        tumor_path = query_fn(wsi_id, tumor_preds_path_list)
        if tumor_path is None:
            continue

        til_path = query_fn(wsi_id, til_preds_path_list)
        if til_path is None:
            continue

        tumor_preds_output_path_cur = '{}/{}'.format(tumor_preds_output_path, wsi_id)
        ensure_dir(tumor_preds_output_path_cur)
        tasks.append((wsi_path, tumor_path, tumor_preds_output_path_cur, (4000,4000)))

        til_preds_output_path_cur = '{}/{}'.format(til_preds_output_path, wsi_id)
        ensure_dir(til_preds_output_path_cur)
        tasks.append((wsi_path, til_path, til_preds_output_path_cur, (4000,4000)))
        
    total_num = int(len(tasks) / 2)
    print('Total number of WSIs is {}'.format(total_num))
        
    end_idx = min(end_idx, total_num)
    tasks = tasks[start_idx*2:end_idx*2]

    if ncores > 1:
        # multiprocessing
        p = Pool(ncores)
        print('Using {} cores'.format(ncores))
        print('Doing interpolation ...')
        for i, _ in enumerate(p.imap_unordered(interp, tasks), 1):
            sys.stderr.write('\rInterpolation {0:%} done! '.format(i / len(tasks)))
    else:
        for task in tasks:
            interp(task)
    print('\nAll done! \n')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolation')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('--start_idx', type=int, default=0, help='start index')
    parser.add_argument('--end_idx', type=int, default=50000, help='end index, (not included)')
    parser.add_argument('--ncores', type=int, default=10, help='number of cores')
    
    args = parser.parse_args()

    if args.config:
        config = json.load(open(args.config))
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    print(config)

    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)
    ncores = int(args.ncores)

    ensure_dir(config['WSIs']['output_path'])
    ensure_dir(config['Nuclei_segs']['output_path'])
    ensure_dir(config['Tumor_preds']['output_path'])
    ensure_dir(config['TIL_preds']['output_path'])

    interp_main(config, start_idx, end_idx, ncores)

    

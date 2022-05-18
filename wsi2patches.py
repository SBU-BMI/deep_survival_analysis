import os
import sys
import argparse
from glob import glob

import json
import numpy as np
from multiprocessing import Pool
from PIL import Image
import openslide

from utils import ensure_dir


def wsi2patches(arguments):
    WSI_path, output_dir, patch_size = arguments
    # some code is adopted from Le's code
    done_fn = '{}/done.txt'.format(output_dir)
    if os.path.isfile(done_fn):
        return None
    
    margin = 5
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

    for x in range(1, width, pw):
        for y in range(1, height, ph):
            # the segmentation results do not have sizes that are < patch_sizes
            if x + pw > width - margin:
                continue
            if y + ph > height - margin:
                continue

            if pw <= 3 or ph <= 3:
                continue

            try:
                patch = oslide.read_region((x, y), 0, (pw, ph)).convert('RGB')
            except:
                 print('{}: exception caught'.format(slide_name))
                 continue
            
            fn = "{}/{}_{}_{}_{}_{}_1_PATCH.png".format(output_dir, x, y, pw, ph, mpp)
            patch.save(fn)
            # print("{}_{}_{}_{}_{}_1_PATCH done!".format(x, y, pw, ph, mpp))

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
    

def wsi2patches_main(config, start_idx, end_idx, ncores):
    wsi_root = config['WSIs']['root_path']
    nuclei_seg_root = config['Nuclei_segs']['root_path']
    tumor_preds_root = config['Tumor_preds']['root_path']
    til_preds_root = config['TIL_preds']['root_path']

    wsi_output_path = config['WSIs']['output_path']
    # nuclei_segs_output_path = config['Nuclei_segs']['output_path']
    # tumor_preds_output_path = config['Tumor_preds']['output_path']
    # til_preds_output_path = config['TIL_preds']['output_path']
    
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

        wsi_output_path_cur = '{}/{}'.format(wsi_output_path, wsi_id)
        ensure_dir(wsi_output_path_cur)
        tasks.append((wsi_path, wsi_output_path_cur, (4000,4000)))

    total_num = int(len(tasks))
    print('Total number of WSIs is {}'.format(total_num))

    end_idx = min(end_idx, total_num)
    tasks = tasks[start_idx:end_idx]

    if ncores > 1:
        p = Pool(ncores)
        print('Using {} cores'.format(ncores))
        print('Dividing into patches ...')
        for i, _ in enumerate(p.imap_unordered(wsi2patches, tasks), 1):
            sys.stderr.write('\rDividing {0:%} done! '.format(i / len(tasks)))
    else:
        for task in tasks:
            wsi2patches(task)
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

    wsi2patches_main(config, start_idx, end_idx, ncores)

    

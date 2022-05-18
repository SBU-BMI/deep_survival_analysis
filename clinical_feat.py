import os
import os.path as path
from glob import glob
import csv

import numpy as np
from utils import ensure_dir, unique


feat_root = '/gpfs/scratch/huidliu/disk/huidong/BMI_projects/data/brca_data/WSI_patch_data_feat_sel_1024'
brca_info_path = './dataset/brca_info.csv'
csv_file_path = './dataset/dataset_for_survival.csv'
clinic_feat_dir = '/gpfs/scratch/huidliu/disk/huidong/BMI_projects/data/brca_data/clinical_features'


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
                wsi_labels[wsi_id] = [dead, days]

    return wsi_labels


def get_wsi_id_age(csv_file_path):
    with open(csv_file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        header = None
        wsi_age = dict()
        for row in spamreader:
            if header is None:
                header = row[0]
            else:
                wsi_id = row[1]
                age = int(row[3])
                wsi_age[wsi_id] = age

        return wsi_age


def get_wsi_id_feats(csv_file_path):
    with open(csv_file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        header = None
        wsi_data = dict()
        for row in spamreader:
            if header is None:
                header = row[0]
            else:
                wsi_id = row[6][1:-1]
                if row[3] == 'NA' or int(row[3]) < 0:
                    continue
                call = row[2][1:-1]
                tumor_stage = row[5][1:-1]
                percent_pos = float(row[10])
                collapsed_stage = row[13][1:-1]
                days = int(row[3])
                dead = int(row[4][1:-1])
                wsi_data[wsi_id] = [call, tumor_stage, percent_pos, collapsed_stage, days, dead]

    return wsi_data

    
wsi_id_label = get_wsi_id_labels(csv_file_path)
wsi_id_age = get_wsi_id_age(brca_info_path)
wsi_id_feats = get_wsi_id_feats(csv_file_path)

call = []
stage = []
col_stage = []


for key, val in wsi_id_feats.items():
    call.append(val[0])
    stage.append(val[1])
    col_stage.append(val[3])

unique_call = unique(call)
unique_stage = unique(stage)
unique_col_stage = unique(col_stage)
print(unique_call)
print(unique_stage)
print(unique_col_stage)


call2num = {'Her2': 3.0, 'LumB': 2.0, 'LumA': 1.0, 'Basal': 4.0}
stage2num = {'stage iib': 2.2, 'stage ia': 0.8, 'stage iiia': 2.8, 'stage iia': 1.8, 'stage i': 1.0, 'stage iiic': 3.4, 'stage iv': 4.0, 'stage ib': 1.2, 'stage iiib': 3.2, 'stage x': 5.0, 'not reported': 0.0, 'stage iii': 3.0, 'stage ii': 2.0}
col_stage2num = {'stage_ii': 2.0, 'stage_i': 1.0, 'stage_iii': 3.0, 'stage_iv': 4.0, 'stage_x/NR': 5.0}


valid_wsi_ids = {'train': [], 'valid': [], 'test': []}

for dataset in ['train', 'valid', 'test']:
    feat_dir = path.join(feat_root, dataset)
    wsi_path_list = glob('{}/*'.format(feat_dir))

    for wsi_path in wsi_path_list:
        wsi_id = path.basename(wsi_path)
        valid_wsi_ids[dataset].append(wsi_id)


for wsi_id, feat_list in wsi_id_feats.items():
    wsi_id_ = '-'.join(wsi_id.split('-')[:3])
    
    wsi_id_feats[wsi_id][0] = call2num[wsi_id_feats[wsi_id][0]]
    wsi_id_feats[wsi_id][1] = stage2num[wsi_id_feats[wsi_id][1]]
    wsi_id_feats[wsi_id][3] = col_stage2num[wsi_id_feats[wsi_id][3]]

    if wsi_id_ in wsi_id_age:
        age = wsi_id_age[wsi_id_]
    else:
        age = 0
    wsi_id_feats[wsi_id].insert(0, float(age))

print(wsi_id_feats)


set_id_feat = {'train': {}, 'valid': {}, 'test': {}}
for dataset in ['train', 'valid', 'test']:
    for wsi_id in valid_wsi_ids[dataset]:
        set_id_feat[dataset][wsi_id] = np.asarray(wsi_id_feats[wsi_id][:-2], dtype=np.float32)

print(set_id_feat)

for dataset in ['train', 'valid', 'test']:
    dataset_dir = path.join(clinic_feat_dir, dataset)    
    ensure_dir(dataset_dir)
    for wsi_id, feat in set_id_feat[dataset].items():
        wsi_id_dir = path.join(dataset_dir, wsi_id)
        ensure_dir(wsi_id_dir)
        fn = path.join(wsi_id_dir, 'feat.npy')
        np.save(fn, feat)

print('Feature extraction done!')

import os
from glob import glob
import argparse

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import csv
from lifelines import CoxPHFitter


def array2dataframe(A, names):
    assert A.shape[1] == len(names), 'columns of array should match length of names'
    dict_ = dict()
    for idx, name in enumerate(names):
        dict_[name] = A[:, idx]
    return pd.DataFrame(dict_)


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
                if row[3] == 'NA':
                    continue
                days = int(row[3])
                dead = int(row[4][1:-1])
                wsi_labels[wsi_id] = (dead, days)

    return wsi_labels


def load_feat_multi_feat(data_info, mode='train'):
    wsi_id_path_list = glob('{}/*/'.format(data_info[mode][0][0]))
    n_data_info = len(data_info[mode])
    n_wsi = len(wsi_id_path_list)
    wsi_id_list = []
    dim_list = []
    feat = None
    fn = data_info[mode][0][1]

    for idx, wsi_id_path in enumerate(wsi_id_path_list, 0):
        wsi_id = wsi_id_path.split('/')[-2]
        wsi_id_list.append(wsi_id)
        wsi_feat_path = '{}{}'.format(wsi_id_path, fn)
        wsi_feat = np.load(wsi_feat_path)
        if len(dim_list) < n_data_info:
            dim_list.append(wsi_feat.shape[0])

        for i in range(1, n_data_info):
            concate_data_root = data_info[mode][i][0]
            concate_fn = data_info[mode][i][1]
            wsi_concate_feat_path = '{}/{}/{}'.format(concate_data_root, wsi_id, concate_fn)
            wsi_concate_feat = np.load(wsi_concate_feat_path)
            wsi_feat = np.concatenate((wsi_feat, wsi_concate_feat), axis=0)
            if len(dim_list) < n_data_info:
                dim_list.append(wsi_concate_feat.shape[0])

        dim = wsi_feat.shape[0]
        if feat is None:
            feat = np.zeros((n_wsi, dim), dtype=wsi_feat.dtype)
        feat[idx] = wsi_feat

    feat_ = None

    dim_list.insert(0, 0)
    dims = np.cumsum(dim_list)
    
    if mode == 'train':
        for i in range(n_data_info):
            pca_model = data_info['pca_model'][i]

            feat_i = feat[:, dims[i]:dims[i+1]]
                
            if pca_model is not None:
                pca_model.fit(feat_i)
                feat_i = pca_model.transform(feat_i)

            if feat_ is None:
                feat_ = feat_i
            else:
                feat_ = np.concatenate((feat_, feat_i), axis=1)

    else:
        for i in range(n_data_info):
            pca_model = data_info['pca_model'][i]

            feat_i = feat[:, dims[i]:dims[i+1]]
            
            if pca_model is not None:
                feat_i = pca_model.transform(feat_i)

            if feat_ is None:
                feat_ = feat_i
            else:
                feat_ = np.concatenate((feat_, feat_i), axis=1)

    return wsi_id_list, feat_, data_info


def get_pca_model(ratio=0.0):
    if ratio > 0:
        pca_model = PCA(n_components=ratio)
    else:
        pca_model = None
    return pca_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--penalizer', type=float, default=0.01, help='L2 penalizer')

    args = parser.parse_args()
    penalizer = float(args.penalizer)
    survival_info = '/home/huidong/projects/BMI_projects/clinic_prediction_dls/datasets/dataset_for_survival.csv'

    data_info = dict()
    global_pca = 0.0
    data_info['pca_ratio'] = [0.0, 0.9, 0.99] # [0.0, 0.0, 0.0, 0.9]

    data_info['train'] = [['/data02/shared/huidong/BMI_project/survival_pred_cce_dls_w_s1/epoch_1400/train', 'feat_level_out_pred.npy'], 
                          ['/data02/shared/huidong/BMI_project/survival_pred_cce_dls_w_s4/epoch_1600/train', 'feat_level_out_pred.npy'],
                          # ['/data02/shared/huidong/BMI_project/survival_pred_cce_dls_w_s8/epoch_1000/train', 'feat_level_out_pred.npy'],
                          ['/data02/shared/huidong/BMI_project/survival_pred_cce_dls_w_s16/epoch_850/train', 'feat_level_out_pred.npy']]

    data_info['test'] = [['/data02/shared/huidong/BMI_project/survival_pred_cce_dls_w_s1/epoch_1400/test', 'feat_level_out_pred.npy'],
                         ['/data02/shared/huidong/BMI_project/survival_pred_cce_dls_w_s4/epoch_1600/test', 'feat_level_out_pred.npy'],
                         # ['/data02/shared/huidong/BMI_project/survival_pred_cce_dls_w_s8/epoch_1000/test', 'feat_level_out_pred.npy'],
                         ['/data02/shared/huidong/BMI_project/survival_pred_cce_dls_w_s16/epoch_850/test', 'feat_level_out_pred.npy']]

    data_info['pca_model'] = []

    for ratio in data_info['pca_ratio']:
        pca_model = get_pca_model(ratio)
        data_info['pca_model'].append(pca_model)

    if global_pca > 0:
        global_pca_model = PCA(n_components=global_pca)
    else:
        global_pca_model = None

    train_wsi_id_list, train_feat, data_info = load_feat_multi_feat(data_info, mode='train')
    test_wsi_id_list, test_feat, data_info = load_feat_multi_feat(data_info, mode='test')
    
    if global_pca_model is not None:
        global_pca_model.fit(train_feat)
        train_feat = global_pca_model.transform(train_feat)
        test_feat = global_pca_model.transform(test_feat)
    
    n_train, dim = train_feat.shape
    n_test = test_feat.shape[0]
    wsi_labels = get_wsi_id_labels(survival_info)
    train_wsi_labels = np.zeros((n_train, 2), dtype=train_feat.dtype)
    test_wsi_labels = np.zeros((n_test, 2), dtype=test_feat.dtype)

    for idx, wsi_id in enumerate(train_wsi_id_list):
        train_wsi_labels[idx] = wsi_labels[wsi_id]

    for idx, wsi_id in enumerate(test_wsi_id_list):
        test_wsi_labels[idx] = wsi_labels[wsi_id]

    train_data = np.concatenate((train_feat, train_wsi_labels), axis=1)
    test_data = np.concatenate((test_feat, test_wsi_labels), axis=1)
    names = ['name_{}'.format(idx) for idx in range(dim)]
    names += ['censor', 'days']

    train_data_df = array2dataframe(train_data, names)
    test_data_df = array2dataframe(test_data, names)

    ###################### train cox model ################################
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_data_df, duration_col="days", event_col="censor")
    cph.print_summary()

    print('concordance_index for training data')
    print(cph.score(train_data_df, scoring_method="concordance_index"))
    print('concordance_index for testing data')
    print(cph.score(test_data_df, scoring_method="concordance_index"))
    


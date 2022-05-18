import os
from glob import glob
import argparse

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import csv
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import warnings
warnings.filterwarnings('ignore')


def create_bootstrap_data(df, n = 100):
    """
    Given a df, create n (default = 100) bootstraps resamples.
    returns an array with the indexes (samples x n)
    """
    bt_indexes = np.random.choice(df.index, (len(df.index), n), replace = True)
    return bt_indexes


def compute_bootstrapped_cindexes(df, duration_col, event_col, n, penalizer):

    df2 = df.copy()
    df2 = df2.reset_index(drop = True)

    indexes = create_bootstrap_data(df = df2, n = n)
    index_all = set(df2.index)

    cindexes_fitted = []
    cindexes_test_oob = []
    cindexes_test_optimism = []
    cindexes_train_red = []
    cindexes_train_bs = []
    HRs = []

    for i in range(n):

        # Create a model and fit to the bootstrap
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(df = df2.loc[indexes[:,i],:], duration_col = duration_col, 
                event_col = event_col)

        #Append the c-index from the fitted model
        cindexes_fitted.append(cph.concordance_index_)

        #Compute the cindex on different set of samples
        index_i = set(indexes[:,i])
        index_test = index_all.difference(index_i)

        # Test on the out-of-bag samples
        pred = -cph.predict_partial_hazard(df2.loc[index_test,:])
        cindex_test = concordance_index(df2.loc[index_test,:][duration_col], pred, df2.loc[index_test,:][event_col])
        cindexes_test_oob.append(cindex_test)

        # Test on the original sample for optimism
        pred = -cph.predict_partial_hazard(df2)
        cindex_test = concordance_index(df2[duration_col], pred, df2[event_col])
        cindexes_test_optimism.append(cindex_test)

        # Check the c-index on the "set" of training samples removing duplicates
        pred = -cph.predict_partial_hazard(df2.loc[index_i,:])
        cindex_train = concordance_index(df2.loc[index_i,:][duration_col], pred, df2.loc[index_i,:][event_col])
        cindexes_train_red.append(cindex_train)

        # Recompute the c-index on the train sample using concordance_index
        pred = -cph.predict_partial_hazard(df2.loc[indexes[:,i],:])
        cindex_train = concordance_index(df2.loc[indexes[:,i],:][duration_col], pred, df2.loc[indexes[:,i],:][event_col])
        cindexes_train_bs.append(cindex_train)

        hr = cph.hazard_ratios_[cph._compute_p_values().argmin()]
        HRs.append(hr)

    low_idx = int(n * 0.025)
    high_idx = max(int(n * 0.975), n - 1)
    cindexes_test_oob = sorted(cindexes_test_oob)
    HRs = sorted(HRs)
    print('Train c-index (extracted from the model): {:3.3f} +- {:3.3f}'.format(np.mean(cindexes_fitted),np.std(cindexes_fitted)))
    print('Train c-index (recalculated using concordance_index): {:3.3f} +- {:3.3f}'.format(np.mean(cindexes_train_bs),np.std(cindexes_train_bs)))
    print('Train c-index (computed only on set of train samples): {:3.3f} +- {:3.3f}'.format(np.mean(cindexes_train_red),np.std(cindexes_train_red)))
    print('Test c-index (computed on the original data for optimism): {:3.3f} +- {:3.3f}'.format(np.mean(cindexes_test_optimism),np.std(cindexes_test_optimism)))
    print('Test c-index (computed only on out-of-bag samples): {:3.3f} +- {:3.3f}'.format(np.mean(cindexes_test_oob),np.std(cindexes_test_oob)))
    print('Test c-index (computed only on out-of-bag samples): {:3.3f} +- {:3.3f}, 95\% confidence interval: ({:3.3f}, {:3.3f})'.format(np.mean(cindexes_test_oob), np.std(cindexes_test_oob), cindexes_test_oob[low_idx], cindexes_test_oob[high_idx]))
    print('Test HR (computed only on out-of-bag samples): {:3.3f} +- {:3.3f}, 95\% confidence interval: ({:3.3f}, {:3.3f})'.format(np.mean(HRs), np.std(HRs), HRs[low_idx], HRs[high_idx]))


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
    parser.add_argument('--penalizer', type=float, default=0.1, help='L2 penalizer')
    parser.add_argument('--n', type=int, default=1000, help='Number of bootstrap samples (default: 1000)')

    args = parser.parse_args()
    penalizer = float(args.penalizer)
    survival_info = './datasets/dataset_for_survival.csv'

    data_info = dict()
    global_pca = 0.0
    data_info['pca_ratio'] = [0.9, 0.9] 


    data_info['train'] = [
                          ['/gpfs/scratch/huidliu/disk/huidong/BMI_projects/survival_pred_cce_dls_w_s8/epoch_1000/train', 'feat_level_out_rgb.npy'],
                          ['/gpfs/scratch/huidliu/disk/huidong/BMI_projects/survival_pred_cce_dls_w_s8/epoch_1000/train', 'feat_level_out_pred.npy'],
    ]

    data_info['test'] = [
                         ['/gpfs/scratch/huidliu/disk/huidong/BMI_projects/survival_pred_cce_dls_w_s8/epoch_1000/test', 'feat_level_out_rgb.npy'],
                         ['/gpfs/scratch/huidliu/disk/huidong/BMI_projects/survival_pred_cce_dls_w_s8/epoch_1000/test', 'feat_level_out_pred.npy'],
    ]

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

    data = np.concatenate((train_data, test_data), axis=0)
    names = ['name_{}'.format(idx) for idx in range(dim)]
    names += ['censor', 'days']

    data_df = array2dataframe(data, names)

    n = args.n 
    duration_col = 'days'
    event_col = 'censor'

    compute_bootstrapped_cindexes(data_df, duration_col, event_col, n, penalizer)
    


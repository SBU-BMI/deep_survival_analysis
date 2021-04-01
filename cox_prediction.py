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


def load_feat(data_root, fn, concate_data_root, concate_fn, pca_model=None, concate_pca_model=None, mode='train'):
    wsi_id_path_list = glob('{}/*/'.format(data_root))
    N = len(wsi_id_path_list)
    wsi_id_list = []
    Feat = None
    dim1 = -1
    dim = -1
    for idx, wsi_id_path in enumerate(wsi_id_path_list, 0):
        wsi_id = wsi_id_path.split('/')[-2]
        wsi_id_list.append(wsi_id)
        wsi_feat_path = '{}{}'.format(wsi_id_path, fn)
        wsi_feat = np.load(wsi_feat_path)
        dim1 = wsi_feat.shape[0]
        if concate_data_root != '' and concate_fn != '':
            wsi_concate_feat_path = '{}/{}/{}'.format(concate_data_root, wsi_id, concate_fn)
            wsi_concate_feat = np.load(wsi_concate_feat_path)
            wsi_feat = np.concatenate((wsi_feat, wsi_concate_feat), axis=0)
        dim = wsi_feat.shape[0]
        if Feat is None:
            Feat = np.zeros((N, dim), dtype=wsi_feat.dtype)
        Feat[idx] = wsi_feat

    Feat1 = Feat[:, :dim1]
    Feat2 = Feat[:, dim1:]
    
    if mode == 'train':
        if pca_model is not None:
            pca_model.fit(Feat1)
            Feat1 = pca_model.transform(Feat1)
        if concate_pca_model is not None:
            concate_pca_model.fit(Feat2)
            Feat2 = concate_pca_model.transform(Feat2)        
    else:
        if pca_model is not None:
            Feat1 = pca_model.transform(Feat1)
        if concate_pca_model is not None:
            Feat2 = concate_pca_model.transform(Feat2)

    Feat = np.concatenate((Feat1, Feat2), axis=1)
    
    return wsi_id_list, Feat, pca_model, concate_pca_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-root', type=str, required=True, help='directory to train data path')
    parser.add_argument('--test-root', type=str, required=True, help='directory to test data path')
    parser.add_argument('--survival-info', type=str, required=True, help='survival info file path')
    parser.add_argument('--fn', type=str, default='wsi_3d_feat.npy', help='feature file name')
    parser.add_argument('--concate-train-root', type=str, default='', help='directory to train data path to concate feature')
    parser.add_argument('--concate-test-root', type=str, default='', help='directory to test data path to concate feature')
    parser.add_argument('--concate-fn', type=str, default='', help='file name to concate feature')
    parser.add_argument('--pca', type=float, default=0.98, help='PCA on data. 0: do not use PCA; (0,1): PCA ratio; >= 1: number of components. (default: 0.98)')
    parser.add_argument('--concate-pca', type=float, default=0.98, help='PCA on concate-data. 0: do not use PCA; (0,1): PCA ratio; >= 1: number of components. (default: 0.98)')
    parser.add_argument('--global-pca', type=float, default=0.98, help='PCA on [data, conate-data]. 0: do not use PCA; (0,1): PCA ratio; >= 1: number of components. (default: 0.98)')
    parser.add_argument('--penalizer', type=float, default=0.01, help='L2 penalizer')

    args = parser.parse_args()

    train_root = args.train_root
    test_root = args.test_root
    survival_info = args.survival_info
    penalizer = float(args.penalizer)
    fn = args.fn
    concate_train_root = args.concate_train_root
    concate_test_root = args.concate_test_root
    concate_fn = args.concate_fn
    pca = args.pca
    concate_pca = args.concate_pca
    global_pca = args.global_pca

    if pca > 0:
        pca_model = PCA(n_components=pca)
    else:
        pca_model = None

    if concate_pca > 0:
        concate_pca_model = PCA(n_components=concate_pca)
    else:
        concate_pca_model = None

    if global_pca > 0:
        global_pca_model = PCA(n_components=global_pca)
    else:
        global_pca_model = None

    train_wsi_id_list, train_Feat, pca_model, concate_pca_model = load_feat(train_root, fn, concate_train_root, concate_fn, pca_model, concate_pca_model, mode='train')
    test_wsi_id_list, test_Feat, pca_model, concate_pca_model = load_feat(test_root, fn, concate_test_root, concate_fn, pca_model, concate_pca_model, mode='test')
    if global_pca_model is not None:
        global_pca_model.fit(train_Feat)
        train_Feat = global_pca_model.transform(train_Feat)
        test_Feat = global_pca_model.transform(test_Feat)
    
    N_train, dim = train_Feat.shape
    N_test = test_Feat.shape[0]
    wsi_labels = get_wsi_id_labels(survival_info)
    train_wsi_labels = np.zeros((N_train, 2), dtype=train_Feat.dtype)
    test_wsi_labels = np.zeros((N_test, 2), dtype=test_Feat.dtype)

    for idx, wsi_id in enumerate(train_wsi_id_list):
        train_wsi_labels[idx] = wsi_labels[wsi_id]

    for idx, wsi_id in enumerate(test_wsi_id_list):
        test_wsi_labels[idx] = wsi_labels[wsi_id]

    train_data = np.concatenate((train_Feat, train_wsi_labels), axis=1)
    test_data = np.concatenate((test_Feat, test_wsi_labels), axis=1)
    names = ['name_{}'.format(idx) for idx in range(dim)]
    names += ['censor', 'days']

    train_data_df = array2dataframe(train_data, names)
    test_data_df = array2dataframe(test_data, names)

    ###################### train cox model ################################
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_data_df, duration_col="days", event_col="censor")
    cph.print_summary()

    print('concordance_index for training using {} data'.format(fn))
    print(cph.score(train_data_df, scoring_method="concordance_index"))
    print('concordance_index for testing using {} data'.format(fn))
    print(cph.score(test_data_df, scoring_method="concordance_index"))
    


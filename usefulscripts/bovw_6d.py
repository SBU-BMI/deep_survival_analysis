import argparse
import os
from glob import glob
import pickle

import numpy as np
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()

parser.add_argument('--train-feat-root', type=str, required=True, help='root of training features')
parser.add_argument('--test-feat-root', type=str, required=True, help='root of testing features')
parser.add_argument('--num-clusters', type=int, default=50, help='number of clusters')
parser.add_argument('--sampling', type=float, default=0.1, help='sampling of training data')
parser.add_argument('--feat-dim', type=int, default=1000, help='dimension of feature') 

args = parser.parse_args()

train_feat_root = args.train_feat_root
test_feat_root = args.test_feat_root
num_clusters = int(args.num_clusters)
sampling = float(args.sampling)
sample_freq = int(1. / sampling)
dim = int(args.feat_dim)

EPS = 1e-9

train_wsi_id_list = glob('{}/*/'.format(train_feat_root))
test_wsi_id_list = glob('{}/*/'.format(test_feat_root))

for idx in range(len(train_wsi_id_list)):
    train_wsi_id_list[idx] = train_wsi_id_list[idx][:-1]

for idx in range(len(test_wsi_id_list)):
    test_wsi_id_list[idx] = test_wsi_id_list[idx][:-1]

train_img_feat_list = []
train_pred_feat_list = []

for wsi_path in train_wsi_id_list:
    train_img_feat = glob('{}/*_img_feat.npy'.format(wsi_path))
    train_pred_feat = glob('{}/*_pred_feat.npy'.format(wsi_path))
    train_img_feat_list += train_img_feat
    train_pred_feat_list += train_pred_feat

N_train = len(train_img_feat_list)

#######################################################################################################
############################# For 6d data #############################################################

def img_feat_path_to_pred_feat_path(img_feat_path):
    fn = img_feat_path.split('/')[-1]
    img_dir = '/'.join(img_feat_path.split('/')[:-1])
    idx = fn.find('_img_feat.npy')
    pred_feat_path = '{}/{}_pred_feat.npy'.format(img_dir, fn[:idx])
    return pred_feat_path

train_img_pred_feat_data = np.ones((N_train, dim*2), dtype=np.float32)

for idx, train_img_feat_path in enumerate(train_img_feat_list, 0):
    train_pred_feat_path = img_feat_path_to_pred_feat_path(train_img_feat_path)
    train_img_feat = np.load(train_img_feat_path)
    train_pred_feat = np.load(train_pred_feat_path)
    train_img_pred_feat_data[idx, :dim] = train_img_feat
    train_img_pred_feat_data[idx, dim:] = train_pred_feat
    if idx % 1000 == 0:
        print('{}/{} done!'.format(idx, N_train))
        
model_path = '{}/kmeans_6d.pkl'.format(train_feat_root)

if False:
    kmeans_6d = KMeans(n_clusters=num_clusters, verbose=1).fit(train_img_pred_feat_data[::sample_freq])
    with open(model_path, 'wb') as output:
        pickle.dump(kmeans_6d, output, pickle.HIGHEST_PROTOCOL)

if True:
    with open(model_path, 'rb') as input:
        kmeans_6d = pickle.load(input)

for wsi_path in train_wsi_id_list:
    train_img_feat = glob('{}/*_img_feat.npy'.format(wsi_path))
    wsi_feat = np.zeros((num_clusters,), dtype=np.float32)
    for train_img_feat_path in train_img_feat:
        train_img_pred_feat_i = np.zeros((1, dim*2), dtype=np.float32)
        train_pred_feat_path = img_feat_path_to_pred_feat_path(train_img_feat_path)
        train_img_feat_i = np.load(train_img_feat_path)
        train_pred_feat_i = np.load(train_pred_feat_path)
        train_img_pred_feat_i[0, :dim] = train_img_feat_i
        train_img_pred_feat_i[0, dim:] = train_pred_feat_i
        cluster_id = kmeans_6d.predict(train_img_pred_feat_i)
        wsi_feat[cluster_id] += 1
    wsi_feat /= (np.sum(wsi_feat) + EPS)
    wsi_feat_path = '{}/wsi_6d_feat.npy'.format(wsi_path)
    np.save(wsi_feat_path, wsi_feat)

for wsi_path in test_wsi_id_list:
    test_img_feat = glob('{}/*_img_feat.npy'.format(wsi_path))
    wsi_feat = np.zeros((num_clusters,), dtype=np.float32)
    for test_img_feat_path in test_img_feat:
        test_img_pred_feat_i = np.zeros((1, dim*2), dtype=np.float32)
        test_pred_feat_path = img_feat_path_to_pred_feat_path(test_img_feat_path)
        test_img_feat_i = np.load(test_img_feat_path)
        test_pred_feat_i = np.load(test_pred_feat_path)
        test_img_pred_feat_i[0, :dim] = test_img_feat_i
        test_img_pred_feat_i[0, dim:] = test_pred_feat_i
        cluster_id = kmeans_6d.predict(test_img_pred_feat_i)
        wsi_feat[cluster_id] += 1
    wsi_feat /= (np.sum(wsi_feat) + EPS)
    wsi_feat_path = '{}/wsi_6d_feat.npy'.format(wsi_path)
    np.save(wsi_feat_path, wsi_feat)


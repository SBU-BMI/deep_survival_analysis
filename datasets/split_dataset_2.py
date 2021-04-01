import os
from glob import glob
import random
import csv

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import torchvision.transforms as transforms


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
                wsi_labels[wsi_id] = [days, dead]

    return wsi_labels


csv_file_path = '/home/huidong/projects/BMI_projects/clinic_prediction_gen_sample_patches/datasets/dataset_for_survival.csv'
train_ratio = 0.5
valid_ratio = 0.25
test_ratio = 0.25
interval = 750
n_intervals = 5
wsi_labels = get_wsi_id_labels(csv_file_path)
wsi_ids = list(wsi_labels.keys())
num_wsi = len(wsi_ids)

X = np.zeros((num_wsi, 2), dtype=np.int32)
y = np.zeros((num_wsi,), dtype=np.uint8)

for index in range(num_wsi):
    wsi_id = wsi_ids[index]
    days, obs = wsi_labels[wsi_id]
    y_ = days // interval
    if y_ >= n_intervals - 1:
        y_ = n_intervals - 1
    X[index, 0] = index
    X[index, 1] = obs
    y[index] = y_


def train_valid_test_split(X, y, train_ratio, valid_ratio, test_ratio):
    X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, shuffle=True, stratify=y)
    valid_ratio_ = valid_ratio / (train_ratio + valid_ratio)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_, y_train_, test_size=valid_ratio_, random_state=42, shuffle=True, stratify=y_train_)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_hist(y, n_intervals):
    counts = np.zeros((n_intervals,), dtype=np.uint32)
    n = y.shape[0]
    for i in range(n):
        counts[y[i]] += 1
    return counts

    
ind_uncensored = X[:, 1] == 1
ind_censored = X[:, 1] == 0
X_uncensored = X[ind_uncensored]
y_uncensored = y[ind_uncensored]
X_censored = X[ind_censored]
y_censored = y[ind_censored]


X_unc_train, y_unc_train, X_unc_valid, y_unc_valid, X_unc_test, y_unc_test = train_valid_test_split(X_uncensored, y_uncensored, train_ratio, valid_ratio, test_ratio)
X_c_train, y_c_train, X_c_valid, y_c_valid, X_c_test, y_c_test = train_valid_test_split(X_censored, y_censored, train_ratio, valid_ratio, test_ratio)

X_train = np.concatenate((X_unc_train, X_c_train), axis=0)
X_valid = np.concatenate((X_unc_valid, X_c_valid), axis=0)
X_test = np.concatenate((X_unc_test, X_c_test), axis=0)

y_train = np.concatenate((y_unc_train, y_c_train), axis=0)
y_valid = np.concatenate((y_unc_valid, y_c_valid), axis=0)
y_test = np.concatenate((y_unc_test, y_c_test), axis=0)

if not os.path.exists('./dataset_info'):
    os.makedirs('./dataset_info')
    
unc_train_hist = get_hist(y_unc_train, n_intervals)
unc_valid_hist = get_hist(y_unc_valid, n_intervals)
unc_test_hist = get_hist(y_unc_test, n_intervals)
c_train_hist = get_hist(y_c_train, n_intervals)
c_valid_hist = get_hist(y_c_valid, n_intervals)
c_test_hist = get_hist(y_c_test, n_intervals)

np.save('./dataset_info/uncensored_train_hist.npy', unc_train_hist)
np.save('./dataset_info/uncensored_valid_hist.npy', unc_valid_hist)
np.save('./dataset_info/uncensored_test_hist.npy', unc_test_hist)
np.save('./dataset_info/censored_train_hist.npy', c_train_hist)
np.save('./dataset_info/censored_valid_hist.npy', c_valid_hist)
np.save('./dataset_info/censored_test_hist.npy', c_test_hist)

train_fn = './train.txt'
valid_fn = './valid.txt'
test_fn = './test.txt'

N = X_train.shape[0]
for i in range(N):
    id_no = X_train[i][0]
    wsi_id = wsi_ids[id_no]
    with open(train_fn, 'a') as f:
        f.write('{}\n'.format(wsi_id))

N = X_valid.shape[0]
for i in range(N):
    id_no = X_valid[i][0]
    wsi_id = wsi_ids[id_no]
    with open(valid_fn, 'a') as f:
        f.write('{}\n'.format(wsi_id))

N = X_test.shape[0]
for i in range(N):
    id_no = X_test[i][0]
    wsi_id = wsi_ids[id_no]
    with open(test_fn, 'a') as f:
        f.write('{}\n'.format(wsi_id))
        


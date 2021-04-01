import os
import argparse
import json
from glob import glob

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

import datasets.wsi_dataset_train as wsi_dataset_train
import datasets.wsi_dataset_eval as wsi_dataset
import models.agg_feat as agg_feat
import loss.censored_crossentropy_loss as cce_loss
from utils import ensure_dir


def load_last_model(model_path, net):
    models = glob('{}/*.pth'.format(model_path))
    model_ids = [(int(f.split('_')[2]), f) for f in [p.split('/')[-1].split('.')[0] for p in models]]
    if not model_ids:
        print('No net loaded!')
        epoch = -1
    else:
        epoch, fn = max(model_ids, key=lambda item: item[0])
        net.load_state_dict(torch.load('{}/{}.pth'.format(
            model_path, fn))
        )
        print('{}.pth for patch classification loaded!'.format(fn))

    return net, epoch


def eval(args, config, device):
    wsi_root = config['tile_process']['WSIs']['output_path']
    nu_seg_root = config['tile_process']['Nuclei_segs']['output_path']
    tumor_pred_root = config['tile_process']['Tumor_preds']['output_path']
    til_pred_root = config['tile_process']['TIL_preds']['output_path']
    label_file = config['tile_process']['label_file']

    data_root = config['dataset']['data_root']
    input_nc = config['dataset']['input_nc']
    data_part = 0  # config['dataset']['data_part']
    data_file_path = config['dataset']['data_file_path']
    n_patches = config['dataset']['n_patches_per_wsi']
    n_patches_wsi = config['dataset']['n_patches_per_wsi_eval']
    interval = config['dataset']['interval']
    n_intervals = config['dataset']['n_intervals']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    mask_root = config['dataset']['mask_root']

    n_epochs = config['train']['n_epochs']
    lr = config['train']['learning_rate']
    output_dir = config['train']['output_dir']
    log_freq = config['train']['log_freq']
    save_freq = config['train']['save_freq']
    valid_freq = config['valid']['valid_freq']

    n_repetitions = n_patches_wsi // n_patches
    mode = args.mode
    feat_level = args.feat_level
    csv_file_path = '{}/dataset_for_survival.csv'.format(data_file_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # data_set = wsi_dataset.WSI_Dataset_Eval(data_root, csv_file_path, input_nc, transform, mode, interval, n_intervals)
    # valid_set = wsi_dataset(data_root, csv_file_path, input_nc, transform, 'valid', n_patches, interval, n_intervals)

    data_set_train = wsi_dataset_train.Patch_Data(
        wsi_root=wsi_root,
        nu_seg_root=nu_seg_root,
        tumor_pred_root=tumor_pred_root,
        til_pred_root=til_pred_root,
        data_file_path=data_file_path,
        mask_root=mask_root,
        mode='train',
        scale=1,
        round_no=0,
        n_patches=4,
        interval=interval,
        n_intervals=n_intervals,
        rgb_only=False,
        data_part=data_part
    )

    data_set_train.set_scale(1)
    data_set_train.set_round_no(0)

    # for debug
    
    data_loader_train = torch.utils.data.DataLoader(
        data_set_train,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    ckpt_dir = '{}/checkpoints'.format(output_dir)

    feat_dir = '{}/feat_dir/epoch_{}/train'.format(output_dir, args.epoch)

    num = len(data_loader_train.dataset)
    res_train = np.zeros((num, 3), dtype=np.uint8)
    confusion_train = np.zeros((n_intervals, n_intervals), dtype=np.int32)
    count = np.zeros((n_intervals,), dtype=np.int32)
    
    for idx, data in enumerate(data_loader_train, 0):
        y, obs, wsi_ids = data
        y_np = y.numpy()[0]
        obs_np = obs.numpy()[0]
        
        feat_dir_wsi_id = '{}/{}/feat_level_out.npy'.format(feat_dir, wsi_ids[0])
        out = np.load(feat_dir_wsi_id)
        pred_y = np.argmax(out)
        
        res_train[idx, 0] = y_np
        res_train[idx, 1] = obs_np
        res_train[idx, 2] = pred_y

        if obs > 0.5:
            confusion_train[y_np, pred_y] += 1
        else:
            if y_np == n_intervals - 1:
                confusion_train[y_np, pred_y] += 1
            else:
                if pred_y > y_np:
                    confusion_train[pred_y, pred_y] += 1
                else:
                    loc = np.random.randint(y_np, n_intervals - 1, size=1)[0] + 1
                    confusion_train[loc, pred_y] += 1
            
        print('{}/{} done!'.format(idx+1, num))

        if obs_np > 0.5:
            count[y_np] += 1
        else:
            if y_np == n_intervals - 1:
                count[y_np] += 1
            else:
                loc = np.random.randint(y_np, n_intervals - 1, size=1)[0] + 1
                count[loc] += 1

    count_t = np.zeros((n_intervals,), dtype=np.int32)            
    for i in range(n_intervals):
        count_t[i] = np.sum(count[:i]) + np.sum(count[i+1:])

    count_total = np.sum(count_t)
    weights = count_t / count_total
    
    print(confusion_train)
    np.save('{}/res_train_epoch_{}.npy'.format(output_dir, args.epoch), res_train)
    np.save('{}/confusion_train_epoch_{}.npy'.format(output_dir, args.epoch), confusion_train)

    feat_dir = '{}/feat_dir/epoch_{}/test'.format(output_dir, args.epoch)
    
    data_set_test = wsi_dataset.Patch_Data_Eval(
        wsi_root=wsi_root,
        nu_seg_root=nu_seg_root,
        tumor_pred_root=tumor_pred_root,
        til_pred_root=til_pred_root,
        data_file_path=data_file_path,
        mask_root=mask_root,
        mode='test',
        scale=1,
        round_no=0,
        n_patches=4,
        n_patches_wsi=4,
        interval=interval,
        n_intervals=n_intervals,
        rgb_only=False,
        data_part=data_part
    )

    data_set_test.set_scale(1)
    data_set_test.set_round_no(0)

    # for debug

    data_loader_test = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    num = len(data_loader_test.dataset)
    res_test = np.zeros((num, 3), dtype=np.uint8)
    confusion_test = np.zeros((n_intervals, n_intervals), dtype=np.int32)

    for idx, data in enumerate(data_loader_test, 0):
        y, obs, wsi_ids = data
        y_np = y.numpy()[0]
        obs_np = obs.numpy()[0]
        pred_y = np.argmax(out)

        feat_dir_wsi_id = '{}/{}/feat_level_out.npy'.format(feat_dir, wsi_ids[0])
        out = np.load(feat_dir_wsi_id)
        res_test[idx, 0] = y_np
        res_test[idx, 1] = obs_np
        res_test[idx, 2] = pred_y

        if obs > 0.5:
            confusion_test[y_np, pred_y] += 1
        else:
            if y_np == n_intervals - 1:
                confusion_test[y_np, pred_y] += 1
            else:
                if pred_y > y_np:
                    confusion_test[pred_y, pred_y] += 1
                else:
                    loc = np.random.randint(y_np, n_intervals - 1, size=1)[0] + 1
                    confusion_test[loc, pred_y] += 1        
        
        print('{}/{} done!'.format(idx+1, num))

    print(confusion_test)
    np.save('{}/res_test_epoch_{}.npy'.format(output_dir, args.epoch), res_test)
    np.save('{}/confusion_test_epoch_{}.npy'.format(output_dir, args.epoch), confusion_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MICNN')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('--mode', default='test', type=str,
                        help='dataset mode: [train | valid | test] (default: test)')
    parser.add_argument('--epoch', default=0, type=int,
                        help='epoch number')
    parser.add_argument('--feat_level', default='out', type=str,
                        help='feature level: [fc | out] (default: out)')
    parser.add_argument('-d', '--gpu_ids', default='0', type=str,
                           help='indices of GPUs to enable (default: 0)')
    parser.add_argument('--aggfeat', required=True, help='avg | cat')
    
    args = parser.parse_args()
    
    if args.config:
        # load config file
        config = json.load(open(args.config))
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda:{0}".format(0) if torch.cuda.is_available() else "cpu")

    eval(args, config, device)
    


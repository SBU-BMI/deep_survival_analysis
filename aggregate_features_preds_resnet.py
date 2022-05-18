import os
import argparse
import json
from glob import glob

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

import datasets.wsi_dataset_eval as wsi_dataset
import models.mobilenet as mobilenet
import models.resnet_aggfeat as resnet
import loss.censored_crossentropy_loss as cce_loss
from utils import ensure_dir


def load_last_model(model_path, net, ch):
    models = glob('{}/*_{}.pth'.format(model_path, ch))
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
    data_part = config['dataset']['data_part']
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
    ch = args.ch
    feat_level = args.feat_level
    csv_file_path = '{}/dataset_for_survival.csv'.format(data_file_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    _6c = input_nc > 3
    rgb_only = not _6c
    # data_set = wsi_dataset.WSI_Dataset_Eval(data_root, csv_file_path, input_nc, transform, mode, interval, n_intervals)
    # valid_set = wsi_dataset(data_root, csv_file_path, input_nc, transform, 'valid', n_patches, interval, n_intervals)

    data_set = wsi_dataset.Patch_Data_Eval(
        wsi_root=wsi_root,
        nu_seg_root=nu_seg_root,
        tumor_pred_root=tumor_pred_root,
        til_pred_root=til_pred_root,
        data_file_path=data_file_path,
        mask_root=mask_root,
        mode=mode,
        scale=1,
        round_no=0,
        n_patches=n_patches,
        n_patches_wsi=n_patches_wsi,
        interval=interval,
        n_intervals=n_intervals,
        rgb_only=rgb_only,
        data_part=data_part
    )

    data_set.set_scale(1)
    data_set.set_round_no(0)

    # for debug
    
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    if ch == 'rgb':
        model = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=3, num_classes=n_intervals)
    elif ch == 'pred':
        model = resnet.resnet50(pretrained=False, in_nc=3, num_classes=n_intervals)
    else:
        raise NotImplementedError
    
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    ckpt_dir = '{}/checkpoints'.format(output_dir)
    model, epoch = load_last_model(ckpt_dir, model, ch)
    model.eval()

    feat_dir = '{}/feat_dir/epoch_{}/{}'.format(output_dir, epoch, mode)
    ensure_dir(feat_dir)

    for idx, data in enumerate(data_loader, 0):
        imgs, y, obs, wsi_ids = data
        imgs, y, obs = imgs[0].to(device), y.to(device), obs.to(device)
        
        if _6c:
            if ch == 'rgb':
                imgs = imgs[:, :3, :, :]
            else:
                imgs = imgs[:, 3:, :, :]
            
        n = imgs.shape[0]

        for i in range(n):
            model.aggregate_features(imgs[i:i+1])

        # print('{}/{} done!'.format(idx, n_repetitions))
        if (idx + 1) % n_repetitions == 0:
            if feat_level == 'out':
                features = model.mean_feature_to_fc()
            else:
                features = model.get_mean_feature()
            features = features[0].data.cpu().numpy()
            feat_dir_wsi_id = '{}/{}'.format(feat_dir, wsi_ids[0])
            ensure_dir(feat_dir_wsi_id)
            fn = '{}/feat_level_{}_{}.npy'.format(feat_dir_wsi_id, feat_level, ch)
            np.save(fn, features)
            model.reset_features()
            print('mode {}, feat_level {}, ch {}, {} {} done!'.format(mode, feat_level, ch, (idx + 1) // n_repetitions, wsi_ids[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MICNN')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('--mode', default='test', type=str,
                        help='dataset mode: [train | valid | test] (default: test)')
    parser.add_argument('--feat_level', default='out', type=str,
                        help='feature level: [fc | out] (default: out)')
    parser.add_argument('--ch', default='rgb', type=str,
                        help='data channel: [rgb | pred] (default: rgb)')
    parser.add_argument('-d', '--gpu_ids', default='0', type=str,
                           help='indices of GPUs to enable (default: 0)')
    
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
    


import os
import argparse
import json
from glob import glob

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

import datasets.wsi_dataset as wsi_dataset
import models.mobilenet as mobilenet
import loss.censored_crossentropy_loss as cce_loss
from utils import ensure_dir


def load_last_model(model_path, net):
    models = glob('{}/*_rgb.pth'.format(model_path))
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
    data_root = config['dataset']['data_root']
    input_nc = config['dataset']['input_nc']
    csv_file_path = config['dataset']['csv_file_path']
    n_patches = config['dataset']['n_patches_per_wsi']
    interval = config['dataset']['interval']
    n_intervals = config['dataset']['n_intervals']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    mode = args.mode

    n_epochs = config['train']['n_epochs']
    lr = config['train']['learning_rate']
    output_dir = config['train']['output_dir']
    log_freq = config['train']['log_freq']
    save_freq = config['train']['save_freq']
    valid_freq = config['valid']['valid_freq']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    test_set = wsi_dataset.WSI_Dataset_Eval(data_root, csv_file_path, input_nc, transform, mode, interval, n_intervals)
    # valid_set = wsi_dataset(data_root, csv_file_path, input_nc, transform, 'valid', n_patches, interval, n_intervals)

    num_workers = 2 # for debug
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    '''
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    '''
    
    model = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=3, num_classes=n_intervals)
    _6c = input_nc > 3
    
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    ckpt_dir = '{}/checkpoints'.format(output_dir)
    model, epoch = load_last_model(ckpt_dir, model)
    model.eval()

    feat_dir = '{}/feat_dir/epoch_{}/{}'.format(output_dir, epoch, mode)
    ensure_dir(feat_dir)

    wsi_id_prev = None

    for idx, data in enumerate(test_loader, 0):
        imgs, y, obs, wsi_ids = data
        imgs, y, obs = imgs.to(device), y.to(device), obs.to(device)
        
        if _6c:
            imgs = imgs[:, :3, :, :]
            
        n = imgs.shape[0]
        for i in range(n):
            wsi_id_cur = wsi_ids[i]
            if wsi_id_cur != wsi_id_prev:
                if wsi_id_prev is not None:
                    features = model.mean_feature_to_fc()
                    features = features[0].data.cpu().numpy()
                    feat_dir_wsi_id = '{}/{}'.format(feat_dir, wsi_id_prev)
                    ensure_dir(feat_dir_wsi_id)
                    fn = '{}/feat_rgb.npy'.format(feat_dir_wsi_id)
                    np.save(fn, features)
                    print('{} saved'.format(fn))
                model.reset_features()
                model.aggregate_features(imgs[i:i+1])
                wsi_id_prev = wsi_id_cur
            else:
                model.aggregate_features(imgs[i:i+1])

    features = model.mean_feature_to_fc()
    features = features[0].data.cpu().numpy()
    feat_dir_wsi_id = '{}/{}'.format(feat_dir, wsi_id_prev)
    ensure_dir(feat_dir_wsi_id)
    fn = '{}/feat_rgb.npy'.format(feat_dir_wsi_id)
    np.save(fn, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MICNN')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('--mode', default='test', type=str,
                        help='dataset mode: [train | valid | test]')
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
    


import os
import argparse
import json
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import datasets.wsi_dataset_ml_train as wsi_dataset
import data_preprocess.data_preprocess as data_preprocess
import models.mobilenet as mobilenet
import loss.censored_crossentropy_loss as cce_loss
from utils import ensure_dir


def load_last_model(model_path, net, net_pred, data_part):
    if data_part == 1 or data_part == 3:
        models = glob('{}/*_rgb.pth'.format(model_path))
        model_ids = [(int(f.split('_')[2]), f) for f in [p.split('/')[-1].split('.')[0] for p in models]]
        if not model_ids:
            print('No net for rgb channels loaded!')
            epoch_rgb = -1
        else:
            epoch_rgb, fn = max(model_ids, key=lambda item: item[0])
            net.load_state_dict(torch.load('{}/{}.pth'.format(
                model_path, fn))
            )

    if data_part == 2 or data_part == 3:
        models = glob('{}/*_pred.pth'.format(model_path))
        model_ids = [(int(f.split('_')[2]), f) for f in [p.split('/')[-1].split('.')[0] for p in models]]
        if not model_ids:
            print('No net for pred channels loaded!')
            epoch_pred = -1
        else:
            epoch_pred, fn = max(model_ids, key=lambda item: item[0])
            net_pred.load_state_dict(torch.load('{}/{}.pth'.format(
                model_path, fn))
            )

    if data_part == 3:
        epoch = min(epoch_rgb, epoch_pred)
    elif data_part == 1:
        epoch = epoch_rgb
    elif data_part == 2:
        epoch = epoch_pred
    else:
        raise NotImplementedError

    return net, net_pred, epoch


def train(args, config, device):
    wsi_root = config['tile_process']['WSIs']['output_path']
    nu_seg_root = config['tile_process']['Nuclei_segs']['output_path']
    tumor_pred_root = config['tile_process']['Tumor_preds']['output_path']
    til_pred_root = config['tile_process']['TIL_preds']['output_path']

    data_root = config['dataset']['data_root']
    input_nc = config['dataset']['input_nc']
    data_part = config['dataset']['data_part']
    data_file_path = config['dataset']['data_file_path']
    n_patches = config['dataset']['n_patches_per_wsi']
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

    label_file = config['tile_process']['label_file']
    patch_size = config['dataset']['patch_size']
    tile_size = config['tile_process']['tile_size']
    max_num_patches = config['dataset']['max_num_patches']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    _6c = input_nc > 3
    rgb_only = not _6c

    if args.fg:
        fg_mask = data_preprocess.FG_Mask(wsi_root, mask_root, label_file, args.scale, patch_size, tile_size, max_num_patches)
        ncores = 10  # number of CPUs to compute the foreground mask
        fg_mask.compute_fg_parallel(ncores)
    
    train_set = wsi_dataset.Patch_Data(
        wsi_root=wsi_root,
        nu_seg_root=nu_seg_root,
        tumor_pred_root=tumor_pred_root,
        til_pred_root=til_pred_root,
        data_file_path=data_file_path,
        mask_root=mask_root,
        mode='train',
        scale=args.scale,
        round_no=0,
        n_patches=n_patches,
        interval=interval,
        n_intervals=n_intervals,
        rgb_only=rgb_only,
        data_part=data_part
    )

    train_set.set_scale(args.scale)
    train_set.set_round_no(0)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,  
        drop_last=False
    )    

    uncensored_train_hist = np.load('{}/dataset_info/uncensored_train_hist.npy'.format(data_file_path))
    censored_train_hist = np.load('{}/dataset_info/censored_train_hist.npy'.format(data_file_path))

    n_uncensored = np.sum(uncensored_train_hist)
    n_censored = np.sum(censored_train_hist)

    w_unc = torch.tensor(1.0).to(device)
    w_c = torch.tensor(args.balanced_weight * n_uncensored / (n_uncensored + n_censored)).to(device)

    if data_part == 1 or data_part == 3:
        model = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=3, num_classes=n_intervals)

    if data_part == 2 or data_part == 3:
        model_pred = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=3, num_classes=n_intervals)
        
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        if data_part == 1 or data_part == 3:
            model = nn.DataParallel(model)
        if data_part == 2 or data_part == 3:
            model_pred = nn.DataParallel(model_pred)

    if data_part == 1 or data_part == 3:
        model = model.to(device)
    if data_part == 2 or data_part == 3:
        model_pred = model_pred.to(device)

    ensure_dir(output_dir)
    log_fn = '{}/log.txt'.format(output_dir)
    ckpt_dir = '{}/checkpoints'.format(output_dir)
    ensure_dir(ckpt_dir)

    if data_part == 3:
        model, model_pred, epoch_prev = load_last_model(ckpt_dir, model, model_pred, data_part)
    elif data_part == 1:
        model, model_pred, epoch_prev = load_last_model(ckpt_dir, model, None, data_part)
    elif data_part == 2:
        model, model_pred, epoch_prev = load_last_model(ckpt_dir, None, model_pred, data_part)
    else:
        raise NotImplementedError

    if data_part == 1 or data_part == 3:
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.RMSprop(model_params, lr=lr)
        # criterion = cce_loss.CensoredCrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()

    if data_part == 2 or data_part == 3:
        model_pred_params = filter(lambda p: p.requires_grad, model_pred.parameters())
        optimizer_pred = optim.RMSprop(model_pred_params, lr=lr)
        # criterion_pred = cce_loss.CensoredCrossEntropyLoss()
        criterion_pred = nn.BCEWithLogitsLoss()

    if data_part == 1 or data_part == 3:
        model.train()
    if data_part == 2 or data_part == 3:
        model_pred.train()
        
    for epoch in range(epoch_prev+1, n_epochs+1):
        if data_part == 1 or data_part == 3:
            running_loss = 0
        if data_part == 2 or data_part == 3:
            running_loss_pred = 0
            
        for idx, data in enumerate(train_loader, 0):
            inputs, y, obs, label = data
            inputs, y, obs, label = inputs[0].to(device), y.to(device), obs.to(device), label.to(device)
            if data_part == 3:
                imgs, preds = inputs[:, :3, :, :], inputs[:, 3:, :, :]
            elif data_part == 1:
                imgs = inputs
            elif data_part == 2:
                preds = inputs

            if data_part == 1 or data_part == 3:
                optimizer.zero_grad()
                output = model(imgs)
                if obs[0].item() == 1:
                    loss = w_unc * criterion(output, label)
                else:
                    loss = w_c * criterion(output, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if data_part == 2 or data_part == 3:
                optimizer_pred.zero_grad()
                output_pred = model_pred(preds)
                if obs[0].item() == 1:
                    loss_pred = w_unc * criterion_pred(output_pred, label)
                else:
                    loss_pred = w_c * criterion_pred(output_pred, label)
                loss_pred.backward()
                optimizer_pred.step()
                running_loss_pred += loss_pred.item()
                
            print('epoch {}, idx {} done!'.format(epoch, idx))

        if data_part == 1 or data_part == 3:
            avg_loss = running_loss / (idx + 1)
        if data_part == 2 or data_part == 3:
            avg_loss_pred = running_loss_pred / (idx + 1)

        if epoch % log_freq == 0:
            if data_part == 3:
                log_str = 'Epoch {:d}/{:d}, loss: {:.6f}, loss_pred: {:.6f}'.format(epoch, n_epochs, avg_loss, avg_loss_pred)
            elif data_part == 1:
                log_str = 'Epoch {:d}/{:d}, loss: {:.6f}'.format(epoch, n_epochs, avg_loss)
            elif data_part == 2:
                log_str = 'Epoch {:d}/{:d}, loss_pred: {:.6f}'.format(epoch, n_epochs, avg_loss_pred)
            else:
                pass
            print(log_str)
            with open(log_fn, 'a') as f:
                f.write('{}\n'.format(log_str))

        if epoch % save_freq == 0:
            if data_part == 1 or data_part == 3:
                torch.save(model.state_dict(), '{}/model_epoch_{}_rgb.pth'.format(ckpt_dir, epoch))
            if data_part == 2 or data_part == 3:
                torch.save(model_pred.state_dict(), '{}/model_epoch_{}_pred.pth'.format(ckpt_dir, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MICNN')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('--fg', action='store_true', default=False,
                        help='compute foreground mask of WSIs (default: False)')
    parser.add_argument('-s', '--scale', default=1, type=int,
                           help='scale (default: 1)')
    parser.add_argument('-w', '--balanced_weight', default=0.5, type=float,
                        help='balanced weight between uncensored and censored data (default: 0.5)')
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

    train(args, config, device)
    


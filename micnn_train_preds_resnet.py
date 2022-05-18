import os
import argparse
import json
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import datasets.wsi_dataset_train as wsi_dataset
import models.mobilenet as mobilenet
import models.resnet_aggfeat as resnet
import loss.censored_crossentropy_loss as cce_loss
from utils import ensure_dir


def load_last_model(model_path, net, net_pred=None):
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
        if net_pred is not None:
            net_pred.load_state_dict(torch.load('{}/model_epoch_{}_pred.pth'.format(
                model_path, epoch))
            )
        print('{}.pth for patch classification loaded!'.format(fn))

    if net_pred is None:
        return net, epoch
    else:
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    _6c = input_nc > 3
    rgb_only = not _6c
    # train_set = wsi_dataset.WSI_Dataset(data_root, csv_file_path, input_nc, transform, 'train', n_patches, interval, n_intervals)
    # valid_set = wsi_dataset(data_root, csv_file_path, input_nc, transform, 'valid', n_patches, interval, n_intervals)
    train_set = wsi_dataset.Patch_Data(
        wsi_root=wsi_root,
        nu_seg_root=nu_seg_root,
        tumor_pred_root=tumor_pred_root,
        til_pred_root=til_pred_root,
        data_file_path=data_file_path,
        mask_root=mask_root,
        mode='train',
        scale=1,
        round_no=0,
        n_patches=n_patches,
        interval=interval,
        n_intervals=n_intervals,
        rgb_only=rgb_only,
        data_part=data_part
    )

    train_set.set_scale(1)
    train_set.set_round_no(0)

    # num_workers = 2  # for debug
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,  # for debug
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
        
    if _6c:
        # model_pred = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=3, num_classes=n_intervals)
        model_pred = resnet.resnet50(pretrained=False, in_nc=3, num_classes=n_intervals)
        
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        if _6c:
            model_pred = nn.DataParallel(model_pred)
    model = model.to(device)
    if _6c:
        model_pred = model_pred.to(device)

    ensure_dir(output_dir)
    log_fn = '{}/log.txt'.format(output_dir)
    ckpt_dir = '{}/checkpoints'.format(output_dir)
    ensure_dir(ckpt_dir)

    if _6c:
        model, model_pred, epoch_prev = load_last_model(ckpt_dir, model, model_pred)
    else:
        model, epoch_prev = load_last_model(ckpt_dir, model)
        
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.RMSprop(model_params, lr=lr)
    criterion = cce_loss.CensoredCrossEntropyLoss()

    if _6c:
        model_pred_params = filter(lambda p: p.requires_grad, model_pred.parameters())
        optimizer_pred = optim.RMSprop(model_pred_params, lr=lr)
        criterion_pred = cce_loss.CensoredCrossEntropyLoss()
    
    model.train()
    if _6c:
        model_pred.train()
        
    for epoch in range(epoch_prev+1, n_epochs+1):
        running_loss = 0
        if _6c:
            running_loss_pred = 0
            
        for idx, data in enumerate(train_loader, 0):
            inputs, y, obs = data
            inputs, y, obs = inputs[0].to(device), y.to(device), obs.to(device)
            if _6c:
                imgs, preds = inputs[:, :3, :, :], inputs[:, 3:, :, :]
            else:
                imgs = inputs
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, y, obs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if _6c:
                optimizer_pred.zero_grad()
                output_pred = model_pred(preds)
                loss_pred = criterion_pred(output_pred, y, obs)
                loss_pred.backward()
                optimizer_pred.step()
                running_loss_pred += loss_pred.item()
                
            print('epoch {}, idx {} done!'.format(epoch, idx))

        avg_loss = running_loss / (idx + 1)
        if _6c:
            avg_loss_pred = running_loss_pred / (idx + 1)

        if epoch % log_freq == 0:
            if _6c:
                log_str = 'Epoch {:d}/{:d}, loss: {:.6f}, loss_pred: {:.6f}'.format(epoch, n_epochs, avg_loss, avg_loss_pred)
            else:
                log_str = 'Epoch {:d}/{:d}, loss: {:.6f}'.format(epoch, n_epochs, avg_loss)
            print(log_str)
            with open(log_fn, 'a') as f:
                f.write('{}\n'.format(log_str))

        if epoch % save_freq == 0:
            torch.save(model.state_dict(), '{}/model_epoch_{}_rgb.pth'.format(ckpt_dir, epoch))
            if _6c:
                torch.save(model_pred.state_dict(), '{}/model_epoch_{}_pred.pth'.format(ckpt_dir, epoch))

        if False:  # epoch % valid_freq == 0:
            model.eval()
            for idx, data in enumerate(valid_loader, 0):
                imgs, y, obs = data
                imgs = imgs[0]
                output = model(imgs)

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MICNN')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
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
    


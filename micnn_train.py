import os
import argparse
import json
from glob import glob

import torch
import torch.optim as optim
import torchvision.transforms as transforms

import datasets.wsi_dataset as wsi_dataset
import models.mobilenet as mobilenet
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


def train(args, config, device):
    data_root = config['dataset']['data_root']
    input_nc = config['dataset']['input_nc']
    csv_file_path = config['dataset']['csv_file_path']
    n_patches = config['dataset']['n_patches_per_wsi']
    interval = config['dataset']['interval']
    n_intervals = config['dataset']['n_intervals']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

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
    
    train_set = wsi_dataset.WSI_Dataset(data_root, csv_file_path, input_nc, transform, 'train', n_patches, interval, n_intervals)
    # valid_set = wsi_dataset(data_root, csv_file_path, input_nc, transform, 'valid', n_patches, interval, n_intervals)

    num_workers = 2  # for debug
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
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
    
    model = mobilenet.mobilenet_v2(pretrained=False, progress=True, input_nc=input_nc, num_classes=n_intervals)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)    

    ensure_dir(output_dir)
    log_fn = '{}/log.txt'.format(output_dir)
    ckpt_dir = '{}/checkpoints'.format(output_dir)
    ensure_dir(ckpt_dir)

    model, epoch_prev = load_last_model(ckpt_dir, model)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.RMSprop(model_params, lr=lr)
    criterion = cce_loss.CensoredCrossEntropyLoss()
    
    model.train()
    for epoch in range(epoch_prev+1, n_epochs+1):
        running_loss = 0
        for idx, data in enumerate(train_loader, 0):
            imgs, y, obs = data
            imgs, y, obs = imgs[0].to(device), y.to(device), obs.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, y, obs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / (idx + 1)

        if epoch % log_freq == 0:
            log_str = 'Epoch {:d}/{:d}, loss: {:.6f}'.format(epoch, n_epochs, avg_loss)
            print(log_str)
            with open(log_fn, 'a') as f:
                f.write('{}\n'.format(log_str))

        if epoch % save_freq == 0:
            torch.save(model.state_dict(), '{}/model_epoch_{}.pth'.format(ckpt_dir, epoch))

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
    


import os
from glob import glob
import argparse
from PIL import Image

import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.models as models


parser = argparse.ArgumentParser()

parser.add_argument('--img-root', required=True, help='image root')
parser.add_argument('--output-dir', required=True, help='output dir')
parser.add_argument('-d', '--cuda_id', default='0', type=str, help='indices of GPUs to enable (default: 0)')

args = parser.parse_args()

if args.cuda_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

device = torch.device("cuda:{0}".format(0) if torch.cuda.is_available() else "cpu")

img_root = args.img_root
output_dir = args.output_dir


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

class Patch_and_Pred(data.Dataset):
    def __init__(self, data_root, transform):
        self.transform = transform
        self.wsi_id_list = glob('{}/*'.format(data_root))
        self.patch_fn_list = []
        for wsi_id in self.wsi_id_list:
            patch_fn_list = glob('{}/*.png'.format(wsi_id))
            self.patch_fn_list += patch_fn_list

    def __len__(self):
        return len(self.patch_fn_list)

    def __getitem__(self, index):
        fn = self.patch_fn_list[index]
        wsi_id, img_name = fn.split('/')[-2:]
        img_name = img_name.split('.')[0]
        img_pred = np.array(Image.open(fn))
        H, W, C = img_pred.shape
        h_W = W // 2
        img = img_pred[:, :h_W, :]
        pred = img_pred[:, h_W:, :]
        img = self.transform(img)
        pred = self.transform(pred)
        return img, pred, wsi_id, img_name
        

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

patch_pred_dataset = Patch_and_Pred(data_root=img_root, transform=transform)
data_loader = torch.utils.data.DataLoader(
    patch_pred_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

vgg = models.vgg16(pretrained=True)
vgg = vgg.to(device)
vgg.eval()
for param in vgg.parameters():
    param.requires_grad = False

N = len(patch_pred_dataset)

for idx, data in enumerate(data_loader, 0):
    img, pred, wsi_id, img_name = data
    img, pred = img.to(device), pred.to(device)
    wsi_id, img_name = wsi_id[0], img_name[0]

    img_feat = vgg(img)
    pred_feat = vgg(pred)
    img_feat = img_feat[0].cpu().numpy()
    pred_feat = pred_feat[0].cpu().numpy()

    out_dir = '{}/{}'.format(output_dir, wsi_id)
    ensure_dir(out_dir)

    img_fn = '{}/{}_img_feat.npy'.format(out_dir, img_name)
    pred_fn = '{}/{}_pred_feat.npy'.format(out_dir, img_name) 
    np.save(img_fn, img_feat)
    np.save(pred_fn, pred_feat)

    if idx % 50 == 0:
        print('{}/{} done!'.format(idx, N))
    

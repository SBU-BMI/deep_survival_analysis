import os
from os import path
from glob import glob

import torch
from torch import nn


def load_last_checkpoint(model, ckpt_dir, column, fn_format=None, strict=True):
    if fn_format is not None:
        checkpoints = glob('{}/{}'.format(ckpt_dir, fn_format))
    else:
        checkpoints = glob('{}/*.pth'.format(ckpt_dir))

    model_ids = [(int(path.splitext(f)[0].split('_')[column]), f) for f in [path.basename(p) for p in checkpoints]]
    if not model_ids:
        start_epoch = -1
    else:
        start_epoch, last_ckpt = max(model_ids, key=(lambda item: item[0]))
        print('Last checkpoint: ', last_ckpt)
        model.load_state_dict(torch.load(path.join(ckpt_dir, last_ckpt), map_location="cpu"), strict=strict)

    return start_epoch, model


def save_checkpoint(model, fn):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), fn)
    else:
        torch.save(model.state_dict(), fn)


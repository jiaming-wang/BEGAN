# -*- coding:utf-8 -*-
'''
 * @Author: wjm
 * @Date: 2019-06-14 11:58:59
 * @Last Modified by:   wjm
 * @Last Modified time: 2019-06-14 11:58:59
 * @Desc:
'''
import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))
    img_bic = img_bic.crop((ty,tx,ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic, info_patch

def get_in_ref_patch(img_in1, img_tar1, img_bic1, img_in2, img_tar2, img_bic2, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in1.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in1 = img_in1.crop((iy,ix,iy + ip, ix + ip))
    img_tar1 = img_tar1.crop((ty,tx,ty + tp, tx + tp))
    img_bic1 = img_bic1.crop((ty,tx,ty + tp, tx + tp))

    img_in2 = img_in2.crop((iy,ix,iy + ip, ix + ip))
    img_tar2 = img_tar2.crop((ty,tx,ty + tp, tx + tp))
    img_bic2 = img_bic2.crop((ty,tx,ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in1, img_tar1, img_bic1, img_in2, img_tar2, img_bic2, info_patch

def get_patch_gan(img_in,  patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size
  
    tp = 1 * patch_size
    ip = tp

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    # (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))


    return img_in

def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, img_bic, info_aug

class DatasetFromGAN_train(data.Dataset):
    def __init__(self, image_dir, patch_size, data_augmentation, transform=None):
        super(DatasetFromGAN_train, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        input = get_patch_gan(input, self.patch_size)

        # if self.data_augmentation:
            # input, target, bicubic, _ = augment(input, target, bicubic)

        if self.transform:
            input = self.transform(input)

        return input, file

    def __len__(self):
        return len(self.image_filenames)

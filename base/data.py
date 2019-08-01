# -*- coding:utf-8 -*-
'''
 * @Author: wjm
 * @Date: 2019-06-14 11:57:14
 * @Last Modified by:   wjm
 * @Last Modified time: 2019-06-14 11:57:14
 * @Desc:
'''
from os.path import join
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from .dataset import DatasetFromGAN_train

def transform():
    return Compose([
        ToTensor(),
    ])

def gan_train(data_dir, patch_size, data_augmentation):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return DatasetFromGAN_train(data_dir, patch_size, data_augmentation,
                             transform=transform)
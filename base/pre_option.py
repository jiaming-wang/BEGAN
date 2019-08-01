# -*- coding:utf-8 -*-
'''
 * @Author: wjm
 * @Date: 2019-06-14 11:18:50
 * @Last Modified by:   wjm
 * @Last Modified time: 2019-06-14 11:18:50
 * @Desc:
'''
import argparse
import os

#Training settings
parser = argparse.ArgumentParser(description='Net')
parser.add_argument('--model_type', type=str, default='Net')

parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size, default=20')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='frequency of model saving')
parser.add_argument('--start_iter', type=int, default=1, help='starting Epoch')
parser.add_argument('--save_best_mode', type=bool, default=False)
parser.add_argument('--gpu_mode', type=bool, default=False)
parser.add_argument('--threads', type=int, default=0,
               help='number of threads for data loader to use, cpu_default=0, Gpu_default=8')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=3, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./dataset', help='path of data')
parser.add_argument('--data_augmentation', type=bool, default=False, help='data augmentation')
parser.add_argument('--hr_train_dataset', type=str, default='./input', help='path of train HR image')
parser.add_argument('--hr_valid_dataset', type=str, default='./input', help='path of vaild HR image')
parser.add_argument('--valid', type=bool, default=False, help='train and val sets together for training')
parser.add_argument('--residual', type=bool, default=False, help='use residual image')
parser.add_argument('--patch_size', type=int, default=32, help='Size of cropped HR image, default=32')
parser.add_argument('--save_folder', default='pre_model', help='Location to save checkpoint models')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--decay', type=str, default='500', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'),
               help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--alpha', type=float, default=0.9, help='RMSprop alpha')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')

# Model settings
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--base_filter', type=int, default=64, help='number of filters')
parser.add_argument('--feat', type=int, default=64, help='feat')

# Loss specifications
parser.add_argument('--loss', type=str, default='MSE',
                    help='loss function configuration (MSE | L1 | VGG22 | VGG54)')

opt = parser.parse_args()

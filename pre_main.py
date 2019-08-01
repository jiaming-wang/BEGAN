# -*- coding:utf-8 -*- 
'''
 * @Author: wjm 
 * @Date: 2019-07-25 11:34:29 
 * @Last Modified by:   wjm 
 * @Last Modified time: 2019-07-25 11:34:29 
 * @Desc: 
'''

import argparse, os, torch
from base.pre_net import pre_GAN

def parse_args():
    parser = argparse.ArgumentParser(description='pre-GAN')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--dataset', type=str, default='dataset/input',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=False)
    parser.add_argument('--benchmark_mode', type=bool, default=False)
    # parser.add_argument('--threads', type=int, default=0,
    #            help='number of threads for data loader to use, cpu_default=0, Gpu_default=8')
    return check_args(parser.parse_args())

def check_args(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args

def main():
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True
    
    gan = pre_GAN(args)

    gan.train()
    print("Training finished!")

    gan.visualize_results(args.epoch)
    print('Testing finished!')

if __name__ == "__main__":
    main()
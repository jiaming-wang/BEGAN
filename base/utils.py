# -*- coding:utf-8 -*- 
'''
 * @Author: wjm 
 * @Date: 2019-06-14 11:37:40 
 * @Last Modified by:   wjm 
 * @Last Modified time: 2019-06-14 11:37:40 
 * @Desc: 
'''
import os, time, datetime, sys, imageio
import torch
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def get_path(subdir):
    return os.path.join(subdir)

def save_config(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    open_type = 'a' if os.path.exists(get_path('./log/config.txt'))else 'w'
    with open(get_path('./log/config.txt'), open_type) as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

def write_log(log, refresh=False):
    print(log)
#     now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    open_type = 'a' if os.path.exists(get_path('./log/log.txt'))else 'w'
    log_file = open(get_path('./log/log.txt'), open_type)
#     log_file.write(now + '\n\n')
    log_file.write(str(log) + '\n')
    if refresh:
        log_file.close()
        log_file = open(get_path('./log/log.txt'), 'a')

def checkpoint(opt, epoch, model):
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
#     model_out_path = opt.save_folder+'/'+opt.model_type+"_epoch_{}.pth".format(epoch)
    model_out_path = opt.save_folder+'/'+opt.model_type+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    log = "Checkpoint saved to {}".format(model_out_path)
    write_log(log)

def checkpoint_best(opt, epoch, model):
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
#     model_out_path = opt.save_folder+'/'+opt.model_type+"_epoch_{}.pth".format(epoch)
    model_out_path = opt.save_folder+'/Best.pth'
    torch.save(model.state_dict(), model_out_path)
    log = "Checkpoint saved to {}".format(model_out_path)
    write_log(log)

def check_opt(opt):
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    if os.listdir(opt.save_folder):
        raise ValueError('The save_folder is not empty!')
    if not os.path.exists(os.path.join(opt.data_dir,opt.hr_train_dataset)):
        raise ValueError('The hr_train_dataset is needed!')
    # if not os.path.exists(os.path.join(opt.data_dir,opt.hr_valid_dataset)):
    #     raise ValueError('The hr_valid_dataset is needed!')     

def save_images(images, size, image_path):
    # image = np.squeeze(merge(images, size))
    image = np.squeeze(images)
    # print(np.shape(image))
    return scipy.misc.imsave(image_path, image)

def save_images1(images, size, image_path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(image_path, image)
    
def generate_animation(path, num):
    image = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        image.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', image, fps=5)

def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
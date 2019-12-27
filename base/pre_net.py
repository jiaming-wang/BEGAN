# -*- coding:utf-8 -*-
'''
 * @Author: wjm
 * @Date: 2019-07-12 14:16:11
 * @Last Modified by:   wjm
 * @Last Modified time: 2019-07-12 14:16:11
 * @Desc:
'''
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import time

from .utils import *
import pickle
from .data import gan_train
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms

class pre_G(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(pre_G, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        # print(input.size())
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class pre_D(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(pre_D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (self.input_size // 2) * (self.input_size // 2), 32),
            nn.Linear(32, 64 * (self.input_size // 2) * (self.input_size // 2)),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            # nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
        
    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = x.view(-1, 64, (self.input_size // 2), (self.input_size // 2))
        x = self.deconv(x)

        return x

class pre_GAN(object):
    def __init__(self, args):
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = 'pre_GAN'
        self.input_size = args.input_size
        self.z_dim = 62
        self.gamma = 1
        self.lambda_ = 0.001
        self.k = 0.0
        self.lr_lower_boundary = 0.00002
        self.data_augmentation = False
        self.channel = 3

        train_set = gan_train(
            self.dataset, 
            self.input_size, 
            self.data_augmentation
            )
        self.data_loader = DataLoader(
            dataset=train_set, 
            num_workers=args.threads, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        ) 
        # transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        # self.data_loader = DataLoader(
        #     datasets.MNIST('data/mnist', train=True, download=False, transform=transform),
        #     batch_size=64, shuffle=True)

        self.G = pre_G(input_dim=self.z_dim, output_dim=self.channel, input_size=self.input_size)
        self.D = pre_D(input_dim=self.channel, output_dim=self.channel, input_size=self.input_size)
        if args.pre_train:
            print(1)
            self.G.load_state_dict(torch.load(args.pre_modelG, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(args.pre_modelD, map_location=lambda storage, loc: storage))

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(args.beta1, args.beta2))
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim)) #noise

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.M = {}
        self.M['pre'] = []
        self.M['pre'].append(1)
        self.M['cur'] = []

        self.y_real, self.y_fake = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)

        if self.gpu_mode:
            self.y_fake, self.y_real = self.y_fake.cuda(), self.y_real.cuda()
        
        self.D.train()
        print('training strat!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iteration, (x_, _) in enumerate(self.data_loader, 1):
            # print(self.data_loader)
            # for iteration, batch in enumerate(self.data_loader, 1):
                # if iteration == self.data_loader.dataset.__len__() // self.batch_size:
                    # break

                z_ = torch.rand((self.batch_size, self.z_dim))
                # x_, name = Variable(batch[0]), batch[1]
                # print(x_.size())
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = torch.mean(torch.abs(D_real - x_))

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(torch.abs(D_fake - G_))

                D_loss = D_real_loss - self.k * D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(torch.abs(D_fake - G_))

                G_loss = D_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                # convergence metric
                temp_M = D_real_loss + torch.abs(self.gamma * D_real_loss - G_loss)

                # operation for updating k
                temp_k = self.k + self.lambda_ * (self.gamma * D_real_loss - G_loss)
                temp_k = temp_k.item()

                self.k = min(max(temp_k, 0), 1)
                self.M['cur'] = temp_M.item()

                if ((iteration + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, M: %.8f, k: %.8f" %
                          ((epoch + 1), (iteration + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(), self.M['cur'], self.k))

            if np.mean(self.M['pre']) < np.mean(self.M['cur']):
                pre_lr = self.G_optimizer.param_groups[0]['lr']
                self.G_optimizer.param_groups[0]['lr'] = max(self.G_optimizer.param_groups[0]['lr'] / 2.0,
                                                             self.lr_lower_boundary)
                self.D_optimizer.param_groups[0]['lr'] = max(self.D_optimizer.param_groups[0]['lr'] / 2.0,
                                                             self.lr_lower_boundary)
                print('M_pre: ' + str(np.mean(self.M['pre'])) + ', M_cur: ' + str(
                    np.mean(self.M['cur'])) + ', lr: ' + str(pre_lr) + ' --> ' + str(
                    self.G_optimizer.param_groups[0]['lr']))
            else:
                print('M_pre: ' + str(np.mean(self.M['pre'])) + ', M_cur: ' + str(np.mean(self.M['cur'])))
                self.M['pre'] = self.M['cur']

                self.M['cur'] = []

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        
    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.sqrt(tot_num_samples))

        if fix:
            self.sample_z_ = self.sample_z_.cuda()
            samples = self.G(self.sample_z_)
        else:
            sample_z_ = torch.rand((1, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()
            
            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)
        # print(np.shape(samples))
        # samples = (samples + 1) / 2
        # save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #                   self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
        # save_images(samples, [64, 64, 1],
        #                     self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
        samples = (samples + 1) / 2
        save_images1(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                           self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

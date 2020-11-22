#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   solver.py
@Time    :   2020/11/21 11:46:04
@Author  :   Wang Zhuo 
@Contact :   1048727525@qq.com
'''
import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
import random
import math
import numpy as np
from PIL import Image
import equalize_hist

class solver(object):
    def __init__(self, args):
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        self.expert_net_choice = args.expert_net_choice

        """ Weight """
        self.adv_weight = args.adv_weight
        self.identity_weight = args.identity_weight
        self.perceptual_weight = args.perceptual_weight
        self.histogram_weight = args.histogram_weight
        self.pixel_weight = args.pixel_weight
        self.pixel_loss_interval = args.pixel_loss_interval

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.device = args.device
        self.resume = args.resume
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.s_B_mean = [0., 0.01697547, 0.04589949, 0.06318113, 0.06832961, 0.06642341, 
                        0.06319189, 0.06153597, 0.06103129, 0.06076431, 0.06144873, 0.06210399, 
                        0.06227141, 0.06173987, 0.0589078, 0.05313862, 0.04477818, 0.03399517, 
                        0.02332931, 0.01438512, 0.00798908, 0.00438292, 0.00263466, 0.00156257,
                        0.]
        self.s_B_mean_tensor = torch.tensor(self.s_B_mean, dtype=torch.float32).repeat(1, 1)
        self.s_B_mean_tensor = self.s_B_mean_tensor.to(self.device)
        print()
        print("##### Information #####")
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# expert net : ", self.expert_net_choice)
        print()
        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print()
        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print()
        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# identity_weight : ", self.identity_weight)
        print('# perceptual_weight : ', self.perceptual_weight)
        print('# histogram_weight : ', self.histogram_weight)
        print('# pixel_weight : ', self.pixel_weight)
        print('# pixel_loss_interval : ', self.pixel_loss_interval)

    def build_model(self):
        if self.expert_net_choice == "senet50":
            self.expert_net = se50_net("./other_models/arcface_se50/model_ir_se50.pth")
        else:
            self.expert_net = Mobile_face_net("./other_models/MobileFaceNet/model_mobilefacenet.pth")
        self.expert_net.to(self.device)
        # A:dark face   B:norm face
        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), self.train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), self.train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), self.test_transform)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disLA.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)
    def tranfer_to_histogram(self, d):
        value_list = []
        for v in d:
            _list = get_lum_distribution(cv2.cvtColor(tensor2im(v.unsqueeze(0)), cv2.COLOR_BGR2RGB))
            value_list.append(_list)
        value_tensor = torch.tensor(value_list, dtype=torch.float32)
        return value_tensor.to(self.device)

    def train(self):
        self.genA2B.train(), self.disGA.train(), self.disLA.train()
        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join("results", self.result_dir, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join("results", self.result_dir, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
        print("training start!")
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            s_tensor_B = self.tranfer_to_histogram(real_B)

            # Update D
            self.D_optim.zero_grad()
            
            fake_A2B, _, _, _, _ = self.genA2B(real_A, s_tensor_B, self.device)
            fake_B2B, _, _, _, _ = self.genA2B(real_B, s_tensor_B, self.device)

            real_GB_logit, _ = self.disGA(real_B)
            real_LB_logit, _ = self.disLA(real_B)
            fake_GB_logit, _ = self.disGA(fake_A2B)
            fake_LB_logit, _ = self.disLA(fake_A2B)

            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device))+self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device))+self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))

            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_loss_LB)

            Discriminator_loss = D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            fake_A2B, _, _, _, _ = self.genA2B(real_A, s_tensor_B, self.device)
            fake_B2B, _, _, _, _ = self.genA2B(real_B, s_tensor_B, self.device)

            fake_GB_logit, _ = self.disGA(fake_A2B)
            fake_LB_logit, _ = self.disLA(fake_A2B)
            
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_loss_LB) + self.identity_weight * G_identity_loss_B

            #perceptual_loss
            perceptual_loss_A2B = self.L1_loss(self.expert_net.get_feature(real_A), self.expert_net.get_feature(fake_A2B))
            perceptual_loss = (perceptual_loss_A2B) * self.perceptual_weight

            #histogram loss
            s_tensor_fake_A2B = self.tranfer_to_histogram(fake_A2B)
            s_tensor_B = self.tranfer_to_histogram(real_B)
            histogram_loss = self.MSE_loss(s_tensor_fake_A2B, s_tensor_B)*self.histogram_weight

            #pixel loss
            if step%self.pixel_loss_interval == 0:
                pixel_loss = self.pixel_weight*(self.L1_loss(fake_A2B, real_A) + self.L1_loss(fake_B2B, real_B))
            else:
                pixel_loss = 0

            Generator_loss = G_loss_B + perceptual_loss + histogram_loss + pixel_loss
            
            Generator_loss.backward()
            self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            print("identity loss : %.8f" % (self.identity_weight * G_identity_loss_B))
            print("perceptual loss : %.8f" % (perceptual_loss))
            print("histogram loss : %.8f" % (histogram_loss))
            print("pixel loss : %.8f" % (pixel_loss))
            
            with torch.no_grad():
                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 6, 0, 3))

                    self.genA2B.eval(), self.disGA.eval(), self.disLA.eval()
                    for _ in range(train_sample_num):
                        try:
                            real_A, _ = trainA_iter.next()
                        except:
                            trainA_iter = iter(self.trainA_loader)
                            real_A, _ = trainA_iter.next()

                        try:
                            real_B, _ = trainB_iter.next()
                        except:
                            trainB_iter = iter(self.trainB_loader)
                            real_B, _ = trainB_iter.next()
                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        fake_A2B, fake_A2B_heatmap0, fake_A2B_heatmap1_0, fake_A2B_heatmap1_1, fake_A2B_heatmap2 = self.genA2B(real_A, self.s_B_mean_tensor, self.device)

                        fake_B2B, fake_B2B_heatmap0, fake_B2B_heatmap1_0, fake_B2B_heatmap1_1, fake_B2B_heatmap2 = self.genA2B(real_B, self.s_B_mean_tensor, self.device)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap0[0]), self.img_size),
                                                                cam(tensor2numpy(fake_A2B_heatmap1_0[0]), self.img_size),
                                                                cam(tensor2numpy(fake_A2B_heatmap1_1[0]), self.img_size),
                                                                cam(tensor2numpy(fake_A2B_heatmap2[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))), 0)), 1)


                    for _ in range(test_sample_num):
                        try:
                            real_A, _ = testA_iter.next()
                        except:
                            testA_iter = iter(self.testA_loader)
                            real_A, _ = testA_iter.next()

                        real_A = real_A.to(self.device)

                        fake_A2B, fake_A2B_heatmap0, fake_A2B_heatmap1_0, fake_A2B_heatmap1_1, fake_A2B_heatmap2 = self.genA2B(real_A, self.s_B_mean_tensor, self.device)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap0[0]), self.img_size),
                                                                cam(tensor2numpy(fake_A2B_heatmap1_0[0]), self.img_size),
                                                                cam(tensor2numpy(fake_A2B_heatmap1_1[0]), self.img_size),
                                                                cam(tensor2numpy(fake_A2B_heatmap2[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))), 0)), 1)

                    cv2.imwrite(os.path.join("results", self.result_dir, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    self.genA2B.train(), self.disGA.train(), self.disLA.train()
                
                if step % self.save_freq == 0:
                    self.save(os.path.join("results", self.result_dir, 'model'), step)

                if step % 1000 == 0:
                    params = {}
                    params['genA2B'] = self.genA2B.state_dict()
                    params['disGA'] = self.disGA.state_dict()
                    params['disLA'] = self.disLA.state_dict()
                    torch.save(params, os.path.join("results", self.result_dir, self.dataset + '_params_latest.pt'))
    
    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.disGA.load_state_dict(params['disGA'])
        self.disLA.load_state_dict(params['disLA'])

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disLA'] = self.disLA.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
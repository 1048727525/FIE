#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   networks.py
@Time    :   2020/11/22 19:23:03
@Author  :   Zhuo Wang 
@Contact :   1048727525@qq.com
'''

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils import *
from model import Backbone

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=112):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        DownRes = []
        for i in range(n_blocks):
            DownRes += [ResnetBlock(ngf * mult, use_bias=False)]
        self.MLP = nn.Sequential(nn.Linear(25, 256), nn.Linear(256, ngf * mult*2))

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpRes_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.DownRes = nn.Sequential(*DownRes)
        self.UpBlock = nn.Sequential(*UpBlock)
        self.eliminate_black_func = eliminate_black(torch.device("cuda:0"))

    def forward(self, input, s, device):
        x = self.DownBlock(input)
        heatmap0 = torch.sum(x, dim=1, keepdim=True)

        x = self.DownRes(x)
        heatmap1_0 = torch.sum(x, dim=1, keepdim=True)

        heatmap1_1 = torch.sum(x, dim=1, keepdim=True)

        h = self.MLP(s)
        h = h.view(h.size(0), h.size(1))
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpRes_' + str(i+1))(x, gamma, beta)
        heatmap2 = torch.sum(x, dim=1, keepdim=True)
        out = self.UpBlock(x)
        out = torch.mul(out+1, self.eliminate_black_func(input+1))-1
        return out, heatmap0, heatmap1_0, heatmap1_1, heatmap2


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)
        heatmap = torch.sum(x, dim=1, keepdim=True)
        x = self.pad(x)
        out = self.conv(x)
        return out, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

class se50_net(nn.Module):
    def __init__(self, model_path):
        super(se50_net, self).__init__()
        self.model = Backbone(50, 0.5, "ir_se")
        for p in self.model.parameters():
            p.requires_grad = False
        pre = torch.load(model_path,  map_location="cpu")
        self.model.load_state_dict(pre)
        self.model.eval()
    def get_feature(self, x):
        """

        :param x: Images
        :return: Embeddings of MobileFaceNets
        """
        feature=self.model(x)
        norm = torch.norm(feature, 2, 1, True)
        feature = torch.div(feature, norm)
        return feature
    def get_layers(self, x, num):
        return self.model.get_layers(x, num)

class Mobile_face_net(nn.Module):
    def __init__(self, model_path):
        super(Mobile_face_net, self).__init__()
        self.model = MobileFaceNet(512)
        for p in self.model.parameters():
            p.requires_grad = False
        pre = torch.load(model_path,  map_location="cpu")
        self.model.load_state_dict(pre)
        self.model.eval()
    def get_feature(self, x):
        """

        :param x: Images
        :return: Embeddings of MobileFaceNets
        """
        feature=self.model(x)
        norm = torch.norm(feature, 2, 1, True)
        feature = torch.div(feature, norm)
        return feature
#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/11/21 11:49:16
@Author  :   Wang Zhuo 
@Contact :   1048727525@qq.com
'''

from scipy import misc
import os, cv2, torch
import numpy as np
import torch.nn as nn
import equalize_hist
from torchvision import transforms
import torch.nn.functional as F

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image
    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def get_lum_distribution(img):
    lum_img = equalize_hist.get_luminance(img)
    row, col = lum_img.shape
    res = np.zeros(25)
    for i in range(col):
        for j in range(row):
            if int(lum_img[i][j]//10)<25:
                res[int(lum_img[i][j]//10)] = res[int(lum_img[i][j]//10)] + 1
    res_sum = sum(res)
    return res/res_sum

def get_lum_distribution_eliminate_black(img):
    lum_img = equalize_hist.get_luminance(img)
    row, col = lum_img.shape
    res = np.zeros(25)
    for i in range(col):
        for j in range(row):
            if int(lum_img[i][j]//10)<25 and lum_img[i][j]!=16:
                res[int(lum_img[i][j]//10)] = res[int(lum_img[i][j]//10)] + 1
    res_sum = sum(res)
    return res/res_sum

def tensor2im(input_image, imtype=np.uint8):
    """Converts a tensor array into a numpy image array.

    args:
        input_image (tensor): the input image tensor array
        imtype (type): the desired type of the converted numpy array
    return:
        img: image as imtype
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def get_lum_map(img):
    """Convert a BGR image to illumination map.

    args:
        img: BGR image as numpy.narry
    return:
        img: BGR image of illumination map
    """
    lum_img = equalize_hist.get_luminance(img)
    lum_img = lum_img/np.max(lum_img)
    lum_img = np.uint8(255 * lum_img)
    lum_img = cv2.applyColorMap(lum_img, cv2.COLORMAP_JET)
    return lum_img

def direct_tranfer_lum(img, mean_lum_value):
    lum_img = equalize_hist.get_luminance(img)
    ratio = (mean_lum_value-16)/(np.mean(lum_img)-16)
    img = img*ratio
    w,h,c = img.shape
    for i in range(w):
        for j in range(h):
            for k in range(c):
                if img[i][j][k]>255:
                    img[i][j][k] = 255
    return img

# input:tensor([b, c, w, h])
def F_eliminate_black(img_tensor):
    b, c, w, h = img_tensor.shape
    res_tensor = torch.zeros((w, h))
    sum_res_tensor = torch.sum(img_tensor, dim=(0, 1))
    res_tensor = (sum_res_tensor!=0).unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1).float()
    return res_tensor

class eliminate_black(nn.Module):
    def __init__(self, device):
        super(eliminate_black, self).__init__()
        self.kernel = np.ones((5, 5))
        self.kernel = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(1).to(device)
    def forward(self, img_tensor):
        b, c, w, h = img_tensor.shape
        res_tensor = torch.zeros((w, h))
        sum_res_tensor = torch.sum(img_tensor, dim=(0, 1))
        res_tensor = (sum_res_tensor!=0).unsqueeze(0).unsqueeze(1).float()
        res_tensor = F.conv2d(res_tensor-0.5, weight=self.kernel, stride=1, padding=2, groups=1)
        return (res_tensor>=0).repeat(1, 3, 1, 1).float()


def listdir(path, list_name, type_list):
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name, type_list)  
        elif os.path.splitext(file_path)[1] in type_list:  
            list_name.append(file_path)

def get_mean_histogram_vec(path):
    img_path_list = []
    listdir(path, img_path_list, [".jpg", ".png"])
    sum_lum_np = np.zeros(25)
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        _list = get_lum_distribution(img)
        sum_lum_np = np.array(_list) + sum_lum_np
    return sum_lum_np/sum(sum_lum_np)

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)
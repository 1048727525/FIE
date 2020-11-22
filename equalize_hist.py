#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   equalize_hist.py
@Time    :   2020/11/22 11:21:26
@Author  :   Wang Zhuo 
@Contact :   1048727525@qq.com
'''

import os
import cv2
import numpy as np
import math
def equalize_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    return dst

def equalize_rgb(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result

def equalize_yuv(img):
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channelsYUV = cv2.split(imgYUV)
    channelsYUV[0] = cv2.equalizeHist(channelsYUV[0])
    channels = cv2.merge(channelsYUV)
    result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    return result

def equalize_hsv(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channelsHSV = cv2.split(imgHSV)
    channelsHSV[2] = cv2.equalizeHist(channelsHSV[2])
    channels = cv2.merge(channelsHSV)
    result = cv2.cvtColor(channels, cv2.COLOR_HSV2BGR)
    return result

def equalize_hls(img):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    channelsHLS = cv2.split(imgHLS)
    channelsHLS[1] = cv2.equalizeHist(channelsHLS[1])
    channels = cv2.merge(channelsHLS)
    result = cv2.cvtColor(channels, cv2.COLOR_HLS2BGR)
    return result

def gamma_correction(img):
    src_l = np.mean(get_luminance(img))
    gamma = cal_gamma(src_l)
    table = np.array([((i / 255.0) ** gamma) * 255.0 for i in np.arange(0, 256)]).astype("uint8")
    result = np.empty(img.shape)
    result = cv2.LUT(np.array(img, dtype = np.uint8), table)
    return result

def get_luminance(img):
    img_blur=cv2.blur(img,(3,3))
    ima_r = img_blur[:, :, 2]
    ima_g = img_blur[:, :, 1]
    ima_b = img_blur[:, :, 0]
    ima_y = 0.256789 * ima_r + 0.504129 * ima_g + 0.097906 * ima_b + 16
    return ima_y

def cal_gamma(l):
     x=math.log(l/255)
     y=math.log(101/255)
     gamma = y/x
     return gamma

def main():
    root='dark'
    dst_dir='dark_res'
    img_list = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filepath in filenames:
            img_list.append(os.path.join(dirpath, filepath))
    for i in img_list:
        if(i[-3:]!='jpg'): continue
        img = cv2.imread(i, 1)
        src_l = np.mean(get_luminance(img))
        # dst_rgb = equalize_rgb(img)
        # dst_yuv = equalize_yuv(img)
        # dst_hsv = equalize_hsv(img)
        # dst_hls = equalize_hls(img)
        dst_gamma = gamma_correction(img)
        dst_l = np.mean(get_luminance(dst_gamma))
        #res=cv2.hconcat([img,dst_rgb,dst_yuv,dst_hsv,dst_hls,dst_gamma])
        res=cv2.hconcat([img,dst_gamma])
        cv2.imwrite(i.replace(root,dst_dir),res)
        print(i,src_l,dst_l)

if __name__ == '__main__':
    main()
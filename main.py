#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/11/22 19:22:55
@Author  :   Zhuo Wang 
@Contact :   1048727525@qq.com
'''

from solver import solver
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of Face Illumination Enhancement Model"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')
    parser.add_argument('--iteration', type=int, default=300000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=False, help='The decay_flag')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')

    # Hyperparameter
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--perceptual_weight', type=int, default=100, help='Weight for Perceptual')
    parser.add_argument('--histogram_weight', type=int, default=100, help='Weight for Histogram')
    parser.add_argument('--pixel_weight', type=float, default=0.01, help='Weight for Pixel')
    parser.add_argument('--pixel_loss_interval', type=int, default=5, help='Interval for Pixel Loss Working')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=112, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='YOUR_RESULT_NAME', help='Directory name to save the results')
    parser.add_argument('--device', default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--expert_net_choice', type=str, default='senet50', choices=['senet50', 'moblieface'])
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join("results", args.result_dir, 'model'))
    check_folder(os.path.join("results", args.result_dir, 'img'))
    check_folder(os.path.join("results", args.result_dir, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = solver(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()

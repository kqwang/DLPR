"""
Hologram-phase dataset generation

From: https://github.com/kqwang/DLPR/
@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import os
import random
import cv2
import torch
import scipy.io as scio
import numpy as np
from optparse import OptionParser
from matplotlib import pyplot as plt

from functions.propagation import propagation

' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('--images', dest='images', default='datasets/raw_images/', help='folder of the raw image dataset')
    parser.add_option('--root', dest='root', default="./", help='root directory')
    parser.add_option('--train in', dest='train_in', default='datasets/train_in/', help='folder of the train input')
    parser.add_option('--train gt', dest='train_gt', default='datasets/train_gt/', help='folder of the train ground truth (gt)')
    parser.add_option('--test in', dest='test_in', default='datasets/test_in/', help='folder of the test input')
    parser.add_option('--test gt', dest='test_gt', default='datasets/test_gt/', help='folder of the test ground truth (gt)')
    parser.add_option('--prop dis', dest='prop_dis', default=20, type='float', help='propagation distance, mm')
    parser.add_option('--norm', dest='norm', default=False, type='int', help='Set True to let generated hologram in the range of [0,1]')
    parser.add_option('--phase min', dest='p_min', default=1, type='float', help='min of h, where phase in [0, h]')
    parser.add_option('--phase max', dest='p_max', default=1, type='float', help='max of h, where phase in [0, h]')
    parser.add_option('--dimension', dest='dim', default=256, type='int', help='dimension of the dataset, [dim, dim]')
    (options, args) = parser.parse_args()
    return options

' Run of the training and validation '
def setup_and_run(dir_images, dir_train_in, dir_train_gt, dir_test_in, dir_test_gt, prop_dis, norm, p_min, p_max, dim):

    image_list = sorted(os.listdir(dir_images))  # paths of row images

    # Creat folders for datasets
    os.makedirs(dir_train_in, exist_ok=True)
    os.makedirs(dir_train_gt, exist_ok=True)
    os.makedirs(dir_test_in, exist_ok=True)
    os.makedirs(dir_test_gt, exist_ok=True)

    # generate paired datasets (holograms-phases)
    for ii, img_name in enumerate(image_list, 1):
        img_path = dir_images + img_name  # path of a row image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read image in grayscale mode
        P = cv2.resize(img, (dim, dim))  # get Phase

        # Set Phase to range of [0, h], where h is distributed in [p_min, p_max]
        P = (P - np.min(P)) / (np.max(P) - np.min(P))
        rand = random.uniform(p_min, p_max)
        P = P * rand

        P = torch.from_numpy(P).cuda().unsqueeze(0).unsqueeze(0)  # phase to torch (cuda)
        H = propagation(P, prop_dis=prop_dis, norm=norm, dim=256)  # Get holograms for phase samples
        H = H[0, 0, :, :].cpu().numpy()  # holograms to numpy
        P = P[0, 0, :, :].cpu().numpy()  # phases to numpy

        # save holograms and phases
        if ii <= int(0.9*(len(image_list))):
            scio.savemat(dir_train_in + '%06d.mat' % (ii), {'input': H})
            scio.savemat(dir_train_gt + '%06d.mat' % (ii), {'gt': P})
        else:
            scio.savemat(dir_test_in + '%06d.mat' % (ii), {'input': H})
            scio.savemat(dir_test_gt + '%06d.mat' % (ii), {'gt': P})
        print("Generating dataset, %.2f %%" % (100*ii/len(image_list)))

        # See some generated holograms and phases
        if ii % int(len(image_list)/5) == 0:
            plt.subplot(121)
            plt.imshow(H, cmap="gray")
            plt.title('hologram (input)')
            plt.colorbar(fraction=0.05, pad=0.05)
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(P, cmap="inferno")
            plt.title('phase (gt)')
            plt.colorbar(fraction=0.05, pad=0.05)
            plt.axis('off')
            plt.show()

' Run the application '
if __name__ == "__main__":
    args = get_args()
    setup_and_run(
        dir_images=args.root + args.images,
        dir_train_in=args.root + args.train_in,
        dir_train_gt=args.root + args.train_gt,
        dir_test_in=args.root + args.test_in,
        dir_test_gt=args.root + args.test_gt,
        prop_dis=args.prop_dis,
        norm=args.norm,
        p_min=args.p_min,
        p_max=args.p_max,
        dim=args.dim)
import cv2
import logging
import torch
from os import path as osp
import argparse
import numpy as np

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, imwrite, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

from YUV_RGB import yuv_import
from YUV_RGB import yuv2rgb
from YUV_RGB import rgb2yuv
from YUV_RGB import yuv_save

def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(255.0*255.0 / mse)
    return psnr

parser2 = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser2.add_argument("--raw_path", type=str, default='../org/foreman.yuv')  # path to raw frames
parser2.add_argument("--com_path", type=str, default='../dec/foreman_dec_h265.yuv')  # path to compressed frames
parser2.add_argument("--enh_path", type=str, default='../enh/H265/foreman_enh_mwgan.yuv')  # path to save enhanced frames
parser2.add_argument("--frame_num", type=int, default=300)
parser2.add_argument("--H", type=int, default=288)
parser2.add_argument("--W", type=int, default=352)
parser2.add_argument('-opt', type=str, help='Path to option YAML file.')

args2 = parser2.parse_args()
print('Reading yuv files')
y_org, u_org, v_org = yuv_import(args2.raw_path, args2.H, args2.W, args2.frame_num)
r_org, g_org, b_org = yuv2rgb(y_org, u_org, v_org, args2.H, args2.W, args2.frame_num)

y_dec, u_dec, v_dec = yuv_import(args2.com_path, args2.H, args2.W, args2.frame_num)
r_dec, g_dec, b_dec = yuv2rgb(y_dec, u_dec, v_dec, args2.H, args2.W, args2.frame_num)

imageRGB = np.zeros((args2.H, args2.W, 3), np.uint8, 'C')
print('Converting from png to yuv')
enh_frames = np.zeros((args2.frame_num, args2.H, args2.W, 3), np.uint8, 'C')
for f in range(args2.frame_num):
    title = './results/test_%i.png' % f
    imageRGB = cv2.imread(title)
    enh_frames[f, :, :, 0] = imageRGB[:, :, 2]
    enh_frames[f, :, :, 1] = imageRGB[:, :, 1]
    enh_frames[f, :, :, 2] = imageRGB[:, :, 0]

y_enh, u_enh, v_enh = rgb2yuv(enh_frames[:, :, :, 0], enh_frames[:, :, :, 1], enh_frames[:, :, :, 2],
                              args2.H,args2.W, args2.frame_num)

y_out = np.zeros((args2.frame_num, args2.H, args2.W), np.uint8, 'C')
u_out = np.zeros((args2.frame_num, args2.H//2, args2.W//2), np.uint8, 'C')
v_out = np.zeros((args2.frame_num, args2.H//2, args2.W//2), np.uint8, 'C')

print('Selecting the most similar frames')
for f in range(args2.frame_num):
    PSNRmax = 0
    for i in range(args2.frame_num):
        psnr = cal_psnr(y_dec[f,:,:], y_enh[i,:,:])
        if psnr>PSNRmax:
            PSNRmax=psnr
            y_out[f, :, :] = y_enh[i, :, :]
            u_out[f, :, :] = u_enh[i, :, :]
            v_out[f, :, :] = v_enh[i, :, :]

print('Saving the final yuv video')
yuv_save(args2.enh_path, args2.W, args2.H, args2.frame_num, y_out, u_out, v_out)

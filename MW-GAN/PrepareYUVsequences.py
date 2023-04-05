import argparse
import numpy as np

from basicsr.utils import get_env_info, imwrite, get_root_logger, get_time_str, make_exp_dirs
from YUV_RGB import yuv_import
from YUV_RGB import yuv2rgb

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

print('Converting from yuv to png')
imageRGB = np.zeros((args2.H, args2.W, 3), np.uint8, 'C')
for f in range(args2.frame_num):
    imageRGB[:, :, 0] = b_org[f,:,:]
    imageRGB[:, :, 1] = g_org[f,:,:]
    imageRGB[:, :, 2] = r_org[f,:,:]
    title = './input/raw/test/test%ix%i_%i.png' % (args2.W, args2.H, f)
    imwrite(imageRGB, title)
    imageRGB[:, :, 0] = b_dec[f, :, :]
    imageRGB[:, :, 1] = g_dec[f, :, :]
    imageRGB[:, :, 2] = r_dec[f, :, :]
    title = './input/comp/test/test%ix%i_%i.png' % (args2.W,args2.H,f)
    imwrite(imageRGB, title)
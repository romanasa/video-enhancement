import os
from collections import OrderedDict

import numpy as np
import torch
import yaml
from tqdm import tqdm

import utils
import os.path as op
from net_rfda import RFDA

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw_path", type=str, required=True)
    parser.add_argument("--decoded_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument(
        '--opt_path', type=str, default='RFDA-PyTorch/option.yml',
        help='Path to option YAML file.'
    )
    return parser.parse_args()


args = parse_args()

# Checkpoints dir
ckp_path = f'RFDA-PyTorch/RFDA_CKPS/RFDA_QP{args.qp}.pt'
# raw yuv and lq yuv path
raw_yuv_path = args.raw_path
lq_yuv_path = args.decoded_path

# vname = lq_yuv_path.split("/")[-1].split('.')[0]
# _,wxh,nfs = vname.split('_')
nfs = args.frames
w, h = args.width, args.height

# nfs = min(nfs, 200)

# this is for our another paper
if 'C2C' in ckp_path:
    m_name = 'C2C'
elif 'RF' or 'Final' in ckp_path:
    m_name = 'RF'
else:
    m_name = 'STDF'


def receive_arg():
    """Process all hyper-parameters and experiment settings."""
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log_test.log"
    )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
    )
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
    )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"
        f"{opts_dict['test']['restore_iter']}"
        '.pt'
    )

    return opts_dict


def f2list_valid(f, nf):
    f2head = {
        3: [0, 1, 2],
        4: [0, 2, 3],
        5: [0, 3, 4],
    }
    if (f < 3):  # list(range(iter_frm - radius, iter_frm + radius + 1))
        return list(range(f - 3, f + 4))
    elif (f < 6):
        head = f2head[f]
    else:
        if (f % 4 == 0):
            head = [f - 8, f - 4, f - 1]
        elif (f % 4 == 1):
            head = [f - 9, f - 5, f - 1]
        elif (f % 4 == 2):
            head = [f - 6, f - 2, f - 1]
        elif (f % 4 == 3):
            head = [f - 7, f - 3, f - 1]
    if (f % 4 == 0):
        tail = [f + 1, f + 4, f + 8]
    elif (f % 4 == 1):
        tail = [f + 1, f + 3, f + 7]
    elif (f % 4 == 2):
        tail = [f + 1, f + 2, f + 6]
    elif (f % 4 == 3):
        tail = [f + 1, f + 5, f + 9]
    if (f >= nf - 9):
        tail = set(tail)
        to_del = set([n for n in tail if (n >= nf)])  # 比nf大的删了
        tail -= to_del
        todo = sorted(list(set(list(range(f + 1, f + 4))) - tail))[:3 - len(tail)]  # 使用相邻帧补充
        tail = list(tail) + todo
        tail = sorted(list(tail))
    return np.array(head + [f] + tail)


def main():
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = receive_arg()
    model = RFDA(opts_dict=opts_dict['network'])
    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    # ==========
    # Load entire video
    # ==========
    msg = f'loading raw and low-quality yuv...'
    print(msg)
    raw_y, raw_u, raw_v = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
    )
    lq_y, lq_u, lq_v = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
    )
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    msg = '> yuv loaded.'
    print(msg)

    # ==========
    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========
    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    enhanced_frames = []

    for idx in range(nfs):
        # load lq
        # idx_list = list(range(idx-3,idx+4))
        # idx_list = np.clip(idx_list, 0, nfs-1)
        if 'C2C' in ckp_path:
            idx_list = f2list_valid(idx, nfs)
            idx_list = np.clip(idx_list, 0, nfs - 1)
        else:
            idx_list = list(range(idx - 3, idx + 4))
            idx_list = np.clip(idx_list, 0, nfs - 1)

        input_data = []
        for idx_ in idx_list:
            input_data.append(lq_y[idx_])
        input_data = torch.from_numpy(np.array(input_data))
        input_data = torch.unsqueeze(input_data, 0).cuda()

        # enhance
        with torch.no_grad():
            if idx == 0:
                enhanced_frm, hint = model(input_data)
            else:
                enhanced_frm, hint = model(input_data, hint)

        numpy_frm = enhanced_frm.detach().cpu().numpy().squeeze()
        enhanced_frames.append(np.clip(numpy_frm, 0, 1) * 255)

        # eval
        gt_frm = torch.from_numpy(raw_y[idx]).cuda()
        # print(gt_frm.size(),'vs',input_data.size())
        batch_ori = criterion(input_data[0, 3, ...], gt_frm)
        batch_perf = criterion(enhanced_frm[0, 0, ...], gt_frm)
        ori_psnr_counter.accum(volume=batch_ori)
        enh_psnr_counter.accum(volume=batch_perf)
        msg = str(idx) + " " + str(batch_ori) + " -> " + str(batch_perf) + "\n"
        print(msg)

        # display
        pbar.set_description(
            "[{:.3f}] {:s} -> [{:.3f}] {:s}"
            .format(batch_ori, unit, batch_perf, unit)
        )
        pbar.update()

    pbar.close()
    ori_ = ori_psnr_counter.get_ave()
    enh_ = enh_psnr_counter.get_ave()
    print('ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
        ori_, unit, enh_, unit, (enh_ - ori_), unit
    ))
    print('> done.')

    if args.output_path is not None:
        y_enhanced = np.array(enhanced_frames, dtype=np.uint8)
        print(f'Saving Y-enhanced to {args.output_path}')
        with open(args.output_path, 'wb') as f:
            f.write(y_enhanced.tobytes())


if __name__ == '__main__':
    main()

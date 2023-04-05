import collections
import os
import pickle

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from data import utils as data_utils


def parse_names():
    result = []
    for f_name in sorted(os.listdir(raw_data)):
        file, ext = os.path.splitext(f_name)
        result.append(file)
    return result


def get_enhanced(path, fmt):
    return data_utils.read_yuv(path, (height, width), fmt)


def calc_metrics():
    metrics = collections.defaultdict(dict)
    for name in names:
        raw = data_utils.read_yuv_420(f'{raw_data}/{name}.yuv', (height, width))
        decoded = data_utils.read_yuv_420(f'{decoded_data}/{name}.yuv', (height, width))

        assert len(raw) == len(decoded), f'{name=}, {len(raw)=}, {len(decoded)=}'

        for algo_name, fmt in algos:
            basepath = os.path.join(algo_data, algo_name, name)
            enhanced = get_enhanced(f'{basepath}.yuv', fmt)

            assert len(enhanced) <= len(raw)

            total_metrics = collections.defaultdict(list)
            for ind in range(min(len(enhanced), frame_limit)):
                enhanced_frame = enhanced[ind].copy()
                raw_frame = raw[ind].copy()
                decoded_frame = decoded[ind].copy()

                psnr = peak_signal_noise_ratio(raw_frame, decoded_frame, data_range=255.0)
                enhanced_psnr = peak_signal_noise_ratio(raw_frame, enhanced_frame, data_range=255.0)

                total_metrics['orig_psnr'].append(psnr)
                total_metrics['enhanced_psnr'].append(enhanced_psnr)

            for k, v in total_metrics.items():
                print(name, algo_name, k, np.mean(v))
                metrics[name, algo_name][k] = np.mean(v)
                metrics[name, algo_name][f'{k}_total'] = v
    return metrics


if __name__ == "__main__":
    qp = 42
    width = 352
    height = 288
    frame_limit = 500
    # raw_data = '/root/data/raw/'
    # decoded_data = f'/root/data/decoded_qp_{qp}/'
    mode = 'test'

    raw_data = f'/Users/roman/vkr/data/raw_{mode}/'
    decoded_data = f'/Users/roman/vkr/data/decoded_qp_{qp}/'
    algo_data = f'/Users/roman/vkr/data/results_{qp}'

    names = parse_names()
    # names = ['akiyo']

    algos = [
        ('QECNN', '420'),
        ('QG-ConvLSTM', '400'),
        ('MFQE', '400'),
        ('STDF', '400'),
        ('RFDA', '400'),
        # ('MW-GAN', '420'),
        # ('aggregated', '400'),
        ('aggregated_brisque', '400'),
        ('aggregated_cnniqa', '400'),
        ('CNET', '400'),
        # ('CNET.backup', '400'),
    ]

    # with open('metrics.pickle', 'rb') as f:
    #     metrics = pickle.load(f)
    metrics = calc_metrics()
    with open(f'psnr_metrics_{qp}_{mode}.pickle', 'wb') as f:
        pickle.dump(metrics, f)

    for algo_info in algos:
        print(algo_info[0], end='\t')
    print()

    delta_metric = 'psnr'
    table = []
    for name in names:
        row = []
        for algo_info in algos:
            cur_metrics = metrics[name, algo_info[0]]
            delta = cur_metrics[f'enhanced_{delta_metric}'] - cur_metrics[f'orig_{delta_metric}']
            if delta_metric == 'lpips':
                delta *= -1
            row.append(delta)

        table.append(row)
        ind = np.argmax(row)

        cname = name
        if cname[-4:] == "_cif":
            cname = cname[:-4]
        cname = cname.replace('_', '\\_')

        print(f'{cname:15}', end=' & ')
        for i in range(len(row)):
            if i == ind:
                print(f'\\textbf{{{row[i]:.3f}}}', end='')
            else:
                print(f'{row[i]:.3f}\t', end='')
            if i + 1 < len(row):
                print(' & ', end = '')
        print('\t\\\\\\hline')

    print(np.mean(table, axis=0))

    # for algo_name in algos:
    #     cur_metrics = metrics['foreman', algo_name]
    #     print(algo_name, cur_metrics['enhanced_psnr_total'][:50])
    # for row in table:
    #     name = row[0]
    #     if name[-4:] == "_cif":
    #         name = name[:-4]
    #
    #     if row[1] > row[2]:
    #         print(f'{name:20} & \\textbf{{{row[1]:.3f}}} & {row[2]:.3f} \\\\\\hline')
    #     else:
    #         print(f'{name:20} & {row[1]:.3f} & \\textbf{{{row[2]:.3f}}} \\\\\\hline')

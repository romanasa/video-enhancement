import argparse
import os
import numpy as np

import torch
import yaml

import data.utils
from data.dataset import YuvDataset
from utils import train as train_utils
from skimage import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Run train')
    parser.add_argument('--height', type=int, required=True, help='height')
    parser.add_argument('--width', type=int, required=True, help='width')
    parser.add_argument('--config_path', type=str, required=True, help='config_path')
    parser.add_argument('--raw_path', type=str, required=True, help='raw_path')
    parser.add_argument('--decoded_path', type=str, required=True, help='decoded_path')
    parser.add_argument('--model_path', type=str, required=True, help='model_path')
    parser.add_argument('--output_path', type=str, required=True, help='output_path')

    return parser.parse_args()


def test(config):
    height = args.height
    width = args.width
    raw_path = args.raw_path
    decoded_path = args.decoded_path
    filename = os.path.basename(raw_path)
    model_path = args.model_path
    output_path = args.output_path

    patch_size = config['patch_size']
    stride = config['stride']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = "mps"

    feature_datasets = []
    for info in config['algos'].values():
        feature_datasets.append(
            YuvDataset(os.path.join(info['folder'], filename), (height, width), info['fmt']))

    model_dict = torch.load(model_path, map_location=device)

    model = train_utils.create_model(config['model']).to(device)
    model.load_state_dict(model_dict['model'])

    enhanced = train_utils.enhance(feature_datasets, model, device, patch_size, stride)
    enhanced = np.array(enhanced * 255, dtype=np.uint8)

    raw = data.utils.read_yuv(raw_path, (height, width), '420')
    decoded = data.utils.read_yuv(decoded_path, (height, width), '420')

    dpsnrs = []
    psnrs = []
    for i in range(len(raw)):
        orig_psnr = metrics.peak_signal_noise_ratio(raw[i], decoded[i], data_range=255)
        enh_psnr = metrics.peak_signal_noise_ratio(raw[i], enhanced[i], data_range=255)
        dpsnr = enh_psnr - orig_psnr
        print(f'Frame {i + 1}/{len(raw)}: {dpsnr=:.3f}, {enh_psnr=:.3f}')
        dpsnrs.append(dpsnr)
        psnrs.append(enh_psnr)

    print(f'Mean dpsnr={np.mean(dpsnrs)}')
    print(f'Mean psnr={np.mean(psnrs)}')

    print(f'Saving Y-enhanced to {output_path}')
    with open(output_path, 'wb') as f:
        f.write(enhanced.tobytes())


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path) as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    test(yaml_config)

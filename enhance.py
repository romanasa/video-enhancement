import argparse
import os
import brisque
import numpy as np
import tqdm

from src.data import utils as data_utils
from CNNIQA import test_demo

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Run enhancement for file')
    parser.add_argument('--raw_path', type=str, required=True, help='path to raw file')
    parser.add_argument('--decoded_path', type=str, required=True, help='path to decoded file')
    parser.add_argument('--enhanced_path', type=str, required=True, help='path to enhanced file')
    parser.add_argument('--result_dir', type=str, required=True, help='path to result_dir')
    parser.add_argument('--qp', type=int, required=True, help='output')

    parser.add_argument('--width', type=int, default=352, help='width')
    parser.add_argument('--height', type=int, default=288, help='height')
    return parser.parse_args()


def run(command):
    print(command)
    os.system(command)


def get_enhanced_path(folder):
    result_folder = f'{args.result_dir}/{folder}'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    return f'{result_folder}/{enhanced_basename}'


def run_encoder_and_decoder():
    if os.path.exists(args.decoded_path):
        print('Skip running encoder and decoder')
        return

    x265_path = "/Users/roman/vkr/src/x265_git/build/linux/x265"
    decoder_path = "/Users/roman/vkr/src/HM/bin/TAppDecoderStatic"

    run(f".{x265_path} {args.raw_path} -o {hevc_path} "
        f"--input-res {width}x{height} --preset medium --fps 30 --qp {qp}")
    run(f"{decoder_path} -b {hevc_path} -o {args.decoded_path}")
    run(f'ffprobe -show_entries "frame=pkt_size" -of csv=p=0 >{csv_path} {hevc_path}')


def run_qe_cnn():
    folder = "QECNN"
    enhanced_path = get_enhanced_path(folder)
    if os.path.exists(f'{enhanced_path}.yuv'):
        print('Skip QECNN')
        return

    run(f'python {folder}/QECNNYUV.py --model QECNN --raw_path {args.raw_path} --com_path {args.decoded_path} '
        f'--enh_path {enhanced_path}.yuv --frame_num {frame_cnt} --H {height} --W {width} '
        f'> {enhanced_path}.txt')


def run_qg_lstm():
    folder = "QG-ConvLSTM"
    enhanced_path = get_enhanced_path(folder)
    if os.path.exists(f'{enhanced_path}.yuv'):
        print('Skip qg_lstm')
        return

    if os.path.exists(f'{args.decoded_path}.npy'):
        print('Skip BRISQUE calculation')
    else:
        run(f'python {folder}/brisquequality.py --path {args.decoded_path} --width {width} --height {height} '
            f'--output_path {args.decoded_path}.npy --qp {qp}')

    run(f'python -u {folder}/test_upgraded.py --raw_path {args.raw_path} --decoded_path {args.decoded_path} '
        f'--model_path {folder}/models/QP{qp}_model/model.ckpt-200000 '
        f'--feature_path {args.decoded_path}.npy '
        f'--frames {frame_cnt} --height {height} --width {width} '
        f'--output_path {enhanced_path}.yuv '
        f' >{enhanced_path}.txt')


def run_mfqe():
    folder = "MFQE"
    enhanced_path = get_enhanced_path(folder)
    if os.path.exists(f'{enhanced_path}.yuv'):
        print('Skip mfqe')
        return

    run(f'python -u {folder}/main_test_upgraded.py --raw_path {args.raw_path} --decoded_path {args.decoded_path} '
        f'--frames {frame_cnt} --height {height} --width {width} '
        f'--output_path {enhanced_path}.yuv --qp {args.qp} '
        f' >{enhanced_path}.txt')


def run_mw_gan():
    folder = 'MW-GAN'
    enhanced_path = get_enhanced_path(folder)
    if os.path.exists(f'{enhanced_path}.yuv'):
        print('Skip MW-GAN')
        return

    run(f'python {folder}/PrepareYUVsequences.py --raw_path {args.raw_path} --com_path {args.decoded_path} '
        f'--frame_num {frame_cnt} --H {height} --W {width}')
    run(f'PYTORCH_ENABLE_MPS_FALLBACK=1 python {folder}/MWGANYUV.py -opt {folder}/test_MWGAN_PSNR.yml')
    run(f'python {folder}/StoreFinalYUVsequence.py --raw_path {args.raw_path} --com_path {args.decoded_path} '
        f'--enh_path {enhanced_path}.yuv --frame_num {frame_cnt} --H {height} --W {width} '
        f' >{enhanced_path}.txt')


def run_cnet():
    folder = 'CNET'
    enhanced_path = get_enhanced_path(folder)
    if os.path.exists(f'{enhanced_path}.yuv'):
        print('Skip CNET')
        return
    run(f'python src/test.py --height {height} --width {width} '
        f'--raw_path {args.raw_path} --decoded_path {args.decoded_path} '
        f'--model_path src/best_model_{qp}.pth --output_path {enhanced_path}.yuv '
        f'--config_path src/test_{qp}.yaml')


def run_stdf():
    folder = 'STDF'
    enhanced_path = get_enhanced_path(folder)
    if os.path.exists(f'{enhanced_path}.yuv'):
        print('Skip STDF')
        return
    run(f'python stdf-pytorch/test_one_video.py --height {height} --width {width} '
        f'--raw_path {args.raw_path} --decoded_path {args.decoded_path} '
        f'--frames {frame_cnt} --output_path {enhanced_path}.yuv ')


def run_rfda():
    folder = 'RFDA'
    enhanced_path = get_enhanced_path(folder)
    if os.path.exists(f'{enhanced_path}.yuv'):
        print('Skip RFDA')
        return
    run(f'python RFDA-PyTorch/test_one_video_yuv_RF.py --height {height} --width {width} '
        f'--raw_path {args.raw_path} --decoded_path {args.decoded_path} '
        f'--frames {frame_cnt} --qp {qp} --output_path {enhanced_path}.yuv ')


def run_aggregated():
    infos = [
        ('QECNN', '420'),
        ('QG-ConvLSTM', '400'),
        ('MFQE', '400'),
        ('STDF', '400'),
        ('RFDA', '400'),
    ]

    enhanced_videos = []
    for algo_folder, fmt in infos:
        cur_path = get_enhanced_path(algo_folder)
        cur_yuv = data_utils.read_yuv(f'{cur_path}.yuv', (height, width), fmt)
        enhanced_videos.append(cur_yuv)
        assert len(cur_yuv) == len(enhanced_videos[0]), f'{len(cur_yuv)=}, {len(enhanced_videos[0])=}'

    brisq = brisque.BRISQUE()
    model_file = 'CNNIQA/models/CNNIQA-LIVE'
    frame_cnt = len(enhanced_videos[0])

    for tp in [
        'brisque',
        'cnniqa',
    ]:
        folder = f'aggregated_{tp}'
        enhanced_path = get_enhanced_path(folder)
        if os.path.exists(f'{enhanced_path}.yuv'):
            print(f'Skip {folder}')
            continue

        aggregated_frames = np.zeros((frame_cnt, height, width))
        weight = np.zeros((frame_cnt, height, width))

        if tp == 'brisque':
            patch_size = 352
            stride = 352
        else:
            patch_size = 32
            stride = 32

        for ind in tqdm.tqdm(range(frame_cnt)):
            for i in range(0, max(height - patch_size, 0) + 1, stride):
                for j in range(0, max(width - patch_size, 0) + 1, stride):
                    scores = []
                    frames = []
                    for enhanced in enhanced_videos:
                        patch = enhanced[ind][i: i + patch_size, j: j + patch_size].copy()
                        frames.append(patch.copy())

                        if tp == 'brisque':
                            score = brisq.get_score(patch)
                        else:
                            iqa_scores = test_demo.get_scores(patch, model_file, patch_size)
                            assert len(iqa_scores) == 1
                            score = iqa_scores[0]
                        scores.append(score)

                    scores = np.array(scores).squeeze()
                    total_patch = np.average(frames, weights=scores, axis=0)
                    aggregated_frames[ind, i: i + patch_size, j: j + patch_size] += total_patch
                    weight[ind, i: i + patch_size, j: j + patch_size] += 1

        aggregated_frames = np.array(np.clip(aggregated_frames / weight, 0, 255), dtype=np.uint8)
        print(f'Saving aggregated Y-enhanced to {enhanced_path}.yuv')
        with open(f'{enhanced_path}.yuv', 'wb') as f:
            f.write(aggregated_frames.tobytes())


if __name__ == "__main__":
    args = parse_args()
    width = args.width
    height = args.height
    qp = args.qp

    frames = data_utils.read_yuv_420(args.raw_path, (height, width))
    frame_cnt = len(frames)

    decoded_folder, decoded_filename = os.path.split(args.decoded_path)
    decoded_basename, decoded_ext = os.path.splitext(decoded_filename)

    enhanced_folder, enhanced_filename = os.path.split(args.enhanced_path)
    enhanced_basename, enhanced_ext = os.path.splitext(enhanced_filename)

    csv_path = f'{decoded_folder}/{decoded_basename}.csv'
    hevc_path = f'{decoded_folder}/{decoded_basename}.hevc'

    run_encoder_and_decoder()

    # run_qe_cnn()
    # run_qg_lstm()
    # run_mfqe()
    # run_mw_gan()
    # run_stdf()
    # run_rfda()
    run_aggregated()
    run_cnet()

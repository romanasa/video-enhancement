from skimage.metrics import peak_signal_noise_ratio
from data import utils as data_utils
import numpy as np


class Meter:
    def __init__(self):
        self.value_sum = 0
        self.weight_sum = 0

    def update(self, value, weight):
        self.value_sum += value * weight
        self.weight_sum += weight

    def get_mean(self):
        return self.value_sum / self.weight_sum


def calc_psnrs(tensor_a, tensor_b):
    psnrs = peak_signal_noise_ratio(data_utils.to_numpy(tensor_a.cpu().detach()),
                                    data_utils.to_numpy(tensor_b.cpu().detach()))
    return psnrs


def calc_dpsnr(features, combined, target, return_percent=False):
    combined_psnr = calc_psnrs(target, combined)
    features_psnrs = []
    for ind in range(features.shape[1]):
        features_psnrs.append(calc_psnrs(target, features[:, ind]))
    dpsnr = (combined_psnr - np.max(features_psnrs)).mean()

    if return_percent:
        return dpsnr, dpsnr / np.max(features_psnrs)
    return dpsnr

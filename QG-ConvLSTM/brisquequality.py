import argparse
import math as m
import os

import cv2
import numpy as np
import tqdm
# for gamma function, called
from scipy.special import gamma as tgamma
import yuv


# AGGD fit model, takes input as the MSCN Image / Pair-wise Product
def AGGDfit(structdis):
    # variables to count positive pixels / negative pixels and their squared sum
    poscount = 0
    negcount = 0
    possqsum = 0
    negsqsum = 0
    abssum = 0

    poscount = len(structdis[structdis > 0])  # number of positive pixels
    negcount = len(structdis[structdis < 0])  # number of negative pixels

    # calculate squared sum of positive pixels and negative pixels
    possqsum = np.sum(np.power(structdis[structdis > 0], 2))
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))

    # absolute squared sum
    abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

    # calculate left sigma variance and right sigma variance
    lsigma_best = np.sqrt((negsqsum / negcount))
    rsigma_best = np.sqrt((possqsum / poscount))

    gammahat = lsigma_best / rsigma_best

    # total number of pixels - totalcount
    totalcount = structdis.shape[1] * structdis.shape[0]

    rhat = m.pow(abssum / totalcount, 2) / ((negsqsum + possqsum) / totalcount)
    rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1) / (m.pow(m.pow(gammahat, 2) + 1, 2))

    prevgamma = 0
    prevdiff = 1e10
    sampling = 0.001
    gam = 0.2

    # vectorized function call for best fitting parameters
    vectfunc = np.vectorize(func, otypes=[float], cache=False)

    # calculate best fit params
    gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

    return [lsigma_best, rsigma_best, gamma_best]


def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
    while (gam < 10):
        r_gam = tgamma(2 / gam) * tgamma(2 / gam) / (tgamma(1 / gam) * tgamma(3 / gam))
        diff = abs(r_gam - rhatnorm)
        if (diff > prevdiff): break
        prevdiff = diff
        prevgamma = gam
        gam += sampling
    gamma_best = prevgamma
    return gamma_best


def compute_features(img):
    scalenum = 2
    feat = []
    # make a copy of the image 
    im_original = img.copy()

    # scale the images twice 
    for itr_scale in range(scalenum):
        im = im_original.copy()
        # normalize the image
        im = im / 255.0

        # calculating MSCN coefficients
        mu = cv2.GaussianBlur(im, (7, 7), 1.166)
        mu_sq = mu * mu

        sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166)
        sigma = (sigma - mu_sq) ** 0.5

        # structdis is the MSCN image
        structdis = im - mu
        structdis /= (sigma + 1.0 / 255)

        # calculate best fitted parameters from MSCN image
        best_fit_params = AGGDfit(structdis)
        # unwrap the best fit parameters 
        lsigma_best = best_fit_params[0]
        rsigma_best = best_fit_params[1]
        gamma_best = best_fit_params[2]

        # append the best fit parameters for MSCN image
        feat.append(gamma_best)
        feat.append((lsigma_best * lsigma_best + rsigma_best * rsigma_best) / 2)

        # shifting indices for creating pair-wise products
        shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]  # H V D1 D2

        for itr_shift in range(1, len(shifts) + 1):
            OrigArr = structdis
            reqshift = shifts[itr_shift - 1]  # shifting index

            # create transformation matrix for warpAffine function
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))

            Shifted_new_structdis = ShiftArr
            Shifted_new_structdis = Shifted_new_structdis * structdis
            # shifted_new_structdis is the pairwise product 
            # best fit the pairwise product 
            best_fit_params = AGGDfit(Shifted_new_structdis)
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best = best_fit_params[2]

            constant = m.pow(tgamma(1 / gamma_best), 0.5) / m.pow(tgamma(3 / gamma_best), 0.5)
            meanparam = (rsigma_best - lsigma_best) * (tgamma(2 / gamma_best) / tgamma(1 / gamma_best)) * constant

            # append the best fit calculated parameters            
            feat.append(gamma_best)  # gamma best
            feat.append(meanparam)  # mean shape
            feat.append(m.pow(lsigma_best, 2))  # left variance square
            feat.append(m.pow(rsigma_best, 2))  # right variance square

        # resize the image on next iteration
        im_original = cv2.resize(im_original, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return feat


# function to calculate BRISQUE quality score
# takes input of the image path

def calc_brisque_for_yuv(yuv_path, size):
    frames = yuv.read_yuv_420(yuv_path, size)
    print(yuv_path, size)
    print(frames.shape)

    features = []
    for frame in tqdm.tqdm(frames):
        features.append(compute_features(frame))
    return np.array(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Calc brisque features')
    parser.add_argument('--path', type=str, required=True, help='path to yuv')
    parser.add_argument('--width', type=int, required=True, help='width')
    parser.add_argument('--height', type=int, required=True, help='height')
    parser.add_argument('--output_path', type=str, required=True, help='output')
    parser.add_argument('--qp', type=int, required=True, help='output')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    folder, name = os.path.split(args.path)
    filename, ext = os.path.splitext(name)
    csv_path = f'{folder}/{filename}.csv'
    assert os.path.exists(csv_path), csv_path

    features = calc_brisque_for_yuv(args.path, (args.height, args.width))

    with open(csv_path, 'r') as f:
        bits = np.array([int(x.strip()) * 8 for x in f.readlines()])
    qps = np.ones(len(features)) * args.qp

    features = np.hstack((features, qps.reshape(-1, 1), bits.reshape(-1, 1)))
    np.save(args.output_path, features)

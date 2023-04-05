import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calc_psnr_and_ssims(raw_yuv, compressed_yuv):
    psnrs = []
    ssims = []
    for raw, compressed in zip(raw_yuv, compressed_yuv):
        psnrs.append(peak_signal_noise_ratio(raw, compressed, data_range=255.0))
        ssims.append(structural_similarity(raw, compressed, data_range=255.0))
    return psnrs, ssims


def create_labels(raw_yuv, compressed_yuv):
    assert len(raw_yuv) == len(compressed_yuv)

    dis_max = 4 # > 3! 1001 has no need to modify
    SSIM_ratio_thr = 0.75
    SSIM_thr = 0.6
    radius = 2

    psnrs, ssims = calc_psnr_and_ssims(raw_yuv, compressed_yuv)

    # Cut video into clips (find starting frame of each clip)
    list_index_ClipStart = [0]
    for ite_SSIM in range(len(ssims)):
        index_left = np.clip(ite_SSIM - radius, 0, len(ssims))
        index_right = np.clip(ite_SSIM + radius + 1, 0, len(ssims))
        SSIM_clip = np.append(ssims[index_left: ite_SSIM],
                              ssims[ite_SSIM + 1: index_right])

        if (ssims[ite_SSIM] < SSIM_thr) or (ssims[ite_SSIM] < SSIM_ratio_thr * np.mean(SSIM_clip)):
            list_index_ClipStart.append(ite_SSIM + 1)

    # Make PQF label for each clip according to PSNR
    PQF_label = np.zeros((len(psnrs),), dtype=np.int32)
    num_clip = len(list_index_ClipStart)
    for ite_ClipStart in range(num_clip):
        # extract PSNR clip
        index_start = list_index_ClipStart[ite_ClipStart]
        if ite_ClipStart < num_clip - 1:
            index_NextStart = list_index_ClipStart[ite_ClipStart + 1]
        else:
            index_NextStart = len(psnrs)
        PSNR_clip = psnrs[index_start: index_NextStart]

        # the first and the last frame must be PQF
        PQFLabel_clip = np.zeros((len(PSNR_clip),), dtype=np.int32)
        if len(PSNR_clip) > 0:
            PQFLabel_clip[0] = 1
            PQFLabel_clip[-1] = 1

        # Make label
        for ite_frame in range(1, len(PQFLabel_clip) - 1):
            if (PSNR_clip[ite_frame] > PSNR_clip[ite_frame - 1]) and (PSNR_clip[ite_frame] >= PSNR_clip[ite_frame + 1]):
                PQFLabel_clip[ite_frame] = 1

        PQF_label[index_start: index_NextStart] = PQFLabel_clip

    # Modify PQF distances until success
    while True:
        PQF_index = np.where(PQF_label == 1)[0]
        PQF_distances = PQF_index[1:] - PQF_index[0:-1]
        TooLongDistance_order_list = np.where(PQF_distances > dis_max)[0]

        if len(TooLongDistance_order_list) == 0:  # None
            break

        # reason: monotony
        for ite_order in range(len(TooLongDistance_order_list)):
            TooLongDistance_order = TooLongDistance_order_list[ite_order]
            PQF_index_left = PQF_index[TooLongDistance_order]
            PQF_index_right = PQF_index[TooLongDistance_order + 1]

            if psnrs[PQF_index_left] <= psnrs[PQF_index_right]:
                PQF_label[PQF_index_right - 2] = 1
            else:
                PQF_label[PQF_index_left + 2] = 1
    return PQF_label

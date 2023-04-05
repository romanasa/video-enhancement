import glob, os
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import net_MFCNN_upgraded
import argparse
import yuv
import pqf_labels

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--raw_path", type=str)
parser.add_argument("--decoded_path", type=str)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--frames", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("--width", type=int)
parser.add_argument('--qp', type=int)

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only show error and warning
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # if GPU is not usable, then turn to CPU automatically

BATCH_SIZE = 1
CHANNEL = 1


# search and test all cmp videos

def return_PQFIndices(PQF_label, QP, ApprQP_label):
    """find all PQFs and their pre/sub PQFs pertain to this QP."""
    PQF_indices = [i for i in range(len(PQF_label)) if PQF_label[i] == 1]

    ApprQPLabel_PQF = [ApprQP_label[i] for i in range(len(ApprQP_label)) if i in PQF_indices]

    PQF_order_part = [o for o in range(len(ApprQPLabel_PQF)) if ApprQPLabel_PQF[o] == QP]
    PQFIndex_list_part = [PQF_indices[o] for o in range(len(PQF_indices)) if o in PQF_order_part]

    if len(PQFIndex_list_part) == 0:
        return [], [], []

    num_PQF = len(PQFIndex_list_part)

    CmpPQFIndex_list_part = PQFIndex_list_part.copy()
    PrePQFIndex_list_part = PQFIndex_list_part[0: (num_PQF - 1)]
    SubPQFIndex_list_part = PQFIndex_list_part[1: num_PQF]

    PrePQFIndex_list_part = [PQFIndex_list_part[0]] + PrePQFIndex_list_part
    SubPQFIndex_list_part.append(PQFIndex_list_part[-1])

    return PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part


def return_NPIndices(PQF_label, QP, ApprQP_label):
    """find all non-PQFs and their pre/sub PQFs pertain to this QP."""
    PQFIndex_list = [i for i in range(len(PQF_label)) if PQF_label[i] == 1]

    # find unqualified non-PQFs and their sub PQFs. Pre PQFs are themselves.
    NonPQFIndex_list = [i for i in range(len(PQF_label)) if (PQF_label[i] == 0) and (i < PQFIndex_list[0])]
    PrePQFIndex_list = NonPQFIndex_list.copy()
    SubPQFIndex_list = [PQFIndex_list[0]] * len(NonPQFIndex_list)

    # find qualified non-PQFs and their pre/sub PQFs.
    NonPQFIndex_list_good = [i for i in range(len(PQF_label)) if
                             (PQF_label[i] == 0) and (i > PQFIndex_list[0]) and (i < PQFIndex_list[-1])]
    NonPQFIndex_list += NonPQFIndex_list_good
    num_NonPQF = len(NonPQFIndex_list_good)
    for ite_NonPQF in range(num_NonPQF):

        index_NonPQF = NonPQFIndex_list_good[ite_NonPQF]

        for ite_PQF in range(len(PQFIndex_list) - 1):

            if (PQFIndex_list[ite_PQF] < index_NonPQF) and (PQFIndex_list[ite_PQF + 1] > index_NonPQF):
                PrePQFIndex_list.append(PQFIndex_list[ite_PQF])
                SubPQFIndex_list.append(PQFIndex_list[ite_PQF + 1])
                break

    # find unqualified non-PQFs and their sub PQFs. Sub PQFs are themselves.
    NonPQFIndex_list_bad = [i for i in range(len(PQF_label)) if (PQF_label[i] == 0) and (i > PQFIndex_list[-1])]
    NonPQFIndex_list += NonPQFIndex_list_bad
    PrePQFIndex_list += [PQFIndex_list[-1]] * len(NonPQFIndex_list_bad)
    SubPQFIndex_list += NonPQFIndex_list_bad

    # find non-PQFs pertain to this QP      
    ApprQPLabel_nonPQF = [ApprQP_label[i] for i in range(len(ApprQP_label)) if i in NonPQFIndex_list]

    NonPQF_order_part = [o for o in range(len(ApprQPLabel_nonPQF)) if ApprQPLabel_nonPQF[o] == QP]
    NonPQFIndex_list_part = [NonPQFIndex_list[o] for o in range(len(NonPQFIndex_list)) if o in NonPQF_order_part]

    if len(NonPQFIndex_list_part) == 0:
        return [], [], []

    PrePQFIndex_list_part = [PrePQFIndex_list[o] for o in range(len(PrePQFIndex_list)) if o in NonPQF_order_part]
    SubPQFIndex_list_part = [SubPQFIndex_list[o] for o in range(len(SubPQFIndex_list)) if o in NonPQF_order_part]

    return PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part


def isplane(frame):
    """detect black frames or other plane frames."""
    tmp_array = np.squeeze(frame).reshape([-1])
    if all(tmp_array[1:] == tmp_array[:-1]):  # all values in this frame are equal
        return True
    else:
        return False


def func_enhance(dir_model_pre, QP, PreIndex_list, CmpIndex_list, SubIndex_list):
    """enhance PQFs or non-PQFs
    update dpsnr, dssim and enhanced frames."""
    global sum_dpsnr, sum_dssim
    global enhanced_list

    tf.compat.v1.reset_default_graph()  # reset graph for new video input

    # defind enhancement process
    x1 = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # previous
    x2 = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # current
    x3 = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, height, width, CHANNEL])  # subsequent

    is_training = tf.compat.v1.placeholder_with_default(False, shape=())

    x1to2 = net_MFCNN_upgraded.warp_img(BATCH_SIZE, x2, x1, False)
    x3to2 = net_MFCNN_upgraded.warp_img(BATCH_SIZE, x2, x3, True)

    x2_enhanced = net_MFCNN_upgraded.network(x1to2, x2, x3to2, is_training)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(config=config) as sess:
        # restore model
        model_path = os.path.join(dir_model_pre, "model_step2.ckpt-" + str(QP))
        saver.restore(sess, model_path)

        nfs = len(CmpIndex_list)

        sum_dpsnr_part = 0.0
        sum_dssim_part = 0.0

        for ite_frame in range(nfs):
            # load frames
            pre_frame = compressed_frames[PreIndex_list[ite_frame]][np.newaxis, :, :, np.newaxis] / 255.0
            cmp_frame = compressed_frames[CmpIndex_list[ite_frame]][np.newaxis, :, :, np.newaxis] / 255.0
            sub_frame = compressed_frames[SubIndex_list[ite_frame]][np.newaxis, :, :, np.newaxis] / 255.0

            # if cmp frame is plane?
            if isplane(cmp_frame):
                continue

            # if PQF frames are plane?
            if isplane(pre_frame):
                pre_frame = np.copy(cmp_frame)
            if isplane(sub_frame):
                sub_frame = np.copy(cmp_frame)

            # enhance
            enhanced_frame = sess.run(x2_enhanced,
                                      feed_dict={x1: pre_frame, x2: cmp_frame, x3: sub_frame, is_training: False})

            # record for output video
            enhanced_list[CmpIndex_list[ite_frame]] = np.squeeze(enhanced_frame)

            # evaluate and accumulate dpsnr
            raw_frame = np.squeeze(raw_frames[CmpIndex_list[ite_frame]]) / 255.0
            cmp_frame = np.squeeze(cmp_frame)
            enhanced_frame = np.squeeze(enhanced_frame)

            raw_frame = np.float32(raw_frame)
            cmp_frame = np.float32(cmp_frame)

            psnr_ori = peak_signal_noise_ratio(raw_frame, cmp_frame, data_range=1.0)
            psnr_aft = peak_signal_noise_ratio(raw_frame, enhanced_frame, data_range=1.0)

            ssim_ori = structural_similarity(cmp_frame, raw_frame, data_range=1.0)
            ssim_aft = structural_similarity(enhanced_frame, raw_frame, data_range=1.0)

            sum_dpsnr_part += psnr_aft - psnr_ori
            sum_dssim_part += ssim_aft - ssim_ori

            print("%d | %d at QP = %d" % (ite_frame + 1, nfs, QP), end="\r")
        print(" " * 20, end="\r")

        sum_dpsnr += sum_dpsnr_part
        sum_dssim += sum_dssim_part

        average_dpsnr = sum_dpsnr_part / nfs
        average_dssim = sum_dssim_part / nfs
        print("dPSNR: %.3f - dSSIM: %.3f - nfs: %4d" % (average_dpsnr, average_dssim, nfs), flush=True)


# enhancement video by video
width = args.width
height = args.height

compressed_frames = yuv.read_yuv_420(args.decoded_path, (height, width))
raw_frames = yuv.read_yuv_420(args.raw_path, (height, width))

PQF_label = pqf_labels.create_labels(raw_frames, compressed_frames)
ApprQP_label = [args.qp] * args.frames
enhanced_list = [[] for _ in range(args.frames)]

# record dpsnr and dssim
sum_dpsnr = 0.0
sum_dssim = 0.0

# enhance PQF
print("enhancing PQF...")

PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part = return_PQFIndices(PQF_label, args.qp,
                                                                                        ApprQP_label)
# enhance PQF
dir_model_pre = os.path.join("MFQE/model", "P_enhancement")
func_enhance(dir_model_pre, args.qp, PrePQFIndex_list_part, CmpPQFIndex_list_part, SubPQFIndex_list_part)

# enhance Non-PQF
print("enhancing non-PQFs...")
PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part = return_NPIndices(PQF_label, args.qp,
                                                                                       ApprQP_label)
# enhance non-PQF
dir_model_pre = os.path.join("MFQE/model", "NP_enhancement")
func_enhance(dir_model_pre, args.qp, PrePQFIndex_list_part, NonPQFIndex_list_part, SubPQFIndex_list_part)

# output and record result
average_dpsnr = sum_dpsnr / args.frames
average_dssim = sum_dssim / args.frames

message = "dPSNR: %.3f - dSSIM: %.3f - nfs: %4d - %s" % (average_dpsnr, average_dssim, args.frames, args.decoded_path)
print(message, flush=True)

enhanced_list = np.array(enhanced_list, dtype=np.float32)
y_enhanced = np.array(np.clip(enhanced_list, 0, 1) * 255, dtype=np.uint8)
print(f'Saving Y-enhanced to {args.output_path}')
with open(args.output_path, 'wb') as f:
    f.write(y_enhanced.tobytes())

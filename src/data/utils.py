import cv2
import numpy as np


class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 // 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            # print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        gray = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)
        return ret, gray


def read_yuv_400(path, size):
    with open(path, 'rb') as f:
        raw = f.read()
    yuv = np.frombuffer(raw, dtype=np.uint8)
    yuv = yuv.reshape((-1, size[0], size[1]))
    return yuv


def read_yuv_420(path, size):
    cap = VideoCaptureYUV(path, size)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    return np.array(frames)


def read_yuv(path, size, fmt):
    if fmt == '400':
        frames = read_yuv_400(path, size)
    elif fmt == '420':
        frames = read_yuv_420(path, size)
    else:
        raise ValueError(f'Unsupported{fmt=}')
    return frames


def to_tensor(gray):
    gray = gray.float() / 255.0
    gray = (gray - 0.5) / 0.5
    return gray


def to_numpy(res):
    if not isinstance(res, np.ndarray):
        res = res.numpy()
    res = (res * 0.5) + 0.5
    return res

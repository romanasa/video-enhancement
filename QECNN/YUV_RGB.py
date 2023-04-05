import PIL
import numpy as np
from PIL import Image
from numpy import *

def yuv_import(filename, fw, fh, numfrm):
    fp = open(filename, 'rb')
    d00 = fw // 2
    d01 = fh // 2

    Y = np.zeros((numfrm, fw, fh), np.uint8, 'C')
    U = np.zeros((numfrm, d00, d01), np.uint8, 'C')
    V = np.zeros((numfrm, d00, d01), np.uint8, 'C')

    for i in range(numfrm):
        for m in range(fw):
            for n in range(fh):
                # print m,n
                Y[i, m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                U[i, m, n] = ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                V[i, m, n] = ord(fp.read(1))
    fp.close()
    return (Y, U, V)

def yuv_save(filename, fw, fh, numfrm,Y,U,V):
    fp = open(filename, 'wb')
    for i in range(numfrm):
        Y[i].astype('uint8').tofile(fp)
        U[i].astype('uint8').tofile(fp)
        V[i].astype('uint8').tofile(fp)
    fp.close()

def yuv2rgb(Y, U, V, height, width, frames):
    U_new = np.empty((U.shape[0], height, width))
    for i in range(U.shape[0]):
        U_new[i] = np.array(Image.fromarray(U[i], mode="L").resize(size=(width, height), resample=PIL.Image.BILINEAR))
    V_new = np.empty((V.shape[0], height, width))
    for i in range(V.shape[0]):
        V_new[i] = np.array(Image.fromarray(V[i], mode="L").resize(size=(width, height), resample=PIL.Image.BILINEAR))
    U = U_new
    V = V_new
    Y = Y

    rf = Y + 1.4075 * (V - 128.0)
    gf = Y - 0.3455 * (U - 128.0) - 0.7169 * (V - 128.0)
    bf = Y + 1.7790 * (U - 128.0)

    rf=np.clip(rf, 0.0, 255.0)
    gf=np.clip(gf, 0.0, 255.0)
    bf=np.clip(bf, 0.0, 255.0)

    r = rf
    g = gf
    b = bf

    return r, g, b

def rgb2yuv(R, G, B, height, width, frames):
    Y = np.empty((R.shape[0], height, width))
    U = np.empty((R.shape[0], height, width))
    V = np.empty((R.shape[0], height, width))
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = (B - Y) * 0.5643 + 128.0
    V = (R - Y) * 0.7132 + 128.0

    Y = np.clip(Y, 0, 255.0)
    U = np.clip(U, 0, 255.0)
    V = np.clip(V, 0, 255.0)

    h2 = np.around(height/2)
    h2 = h2.astype(int)
    w2 = np.around(width/ 2)
    w2 = w2.astype(int)
    Cb = np.empty((U.shape[0],h2 , w2))
    Cr = np.empty((V.shape[0], h2, w2))

    for i in range(frames):
        for m in range(h2):
            for n in range(w2):
                Cb[i,m,n]   = (U[i, 2 * m, 2 * n]+U[i, 2 * m+1, 2 * n]+U[i, 2 * m, 2 * n+1]+U[i, 2 * m+1, 2 * n+1])/4
                Cr[i, m, n] = (V[i, 2 * m, 2 * n]+V[i, 2 * m+1, 2 * n]+V[i, 2 * m, 2 * n+1]+V[i, 2 * m+1, 2 * n+1])/4

    return Y, Cb, Cr


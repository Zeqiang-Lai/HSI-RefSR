import numpy as np
import random


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def crop_center(img, cropx, cropy):
    x, y = img.shape[-2], img.shape[-1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[..., startx:startx+cropx, starty:starty+cropy]


def rand_crop(img, cropx, cropy):
    x, y = img.shape[-2], img.shape[-1]
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[..., x1:x1+cropx, y1:y1+cropy]


def mod_crop(img, modulo):
    _, ih, iw = img.shape
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img[:, 0:ih, 0:iw]
    return img


def hwc2chw(img):
    return img.transpose(2, 0, 1)


def chw2hwc(img):
    return img.transpose(1, 2, 0)

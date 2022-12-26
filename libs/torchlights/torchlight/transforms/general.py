import random

import numpy as np

from ._util import LockedIterator
from .functional import chw2hwc, hwc2chw


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        out = data
        for transform in self.transforms:
            out = transform(out)
        return out


class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out


class MinMaxNormalize:
    def __call__(self, array):
        amin = np.min(array)
        amax = np.max(array)
        return (array - amin) / (amax - amin)


class CenterCrop:
    def __init__(self, size):
        self.cropx = size[0]
        self.cropy = size[1]

    def __call__(self, img):
        x, y = img.shape[-2], img.shape[-1]
        startx = x//2-(self.cropx//2)
        starty = y//2-(self.cropy//2)
        return img[..., startx:startx+self.cropx, starty:starty+self.cropy]


class RandCrop:
    def __init__(self, size):
        self.cropx = size[0]
        self.cropy = size[1]

    def __call__(self, img):
        x, y = img.shape[-2], img.shape[-1]
        x1 = random.randint(0, x - self.cropx)
        y1 = random.randint(0, y - self.cropy)
        return img[..., x1:x1+self.cropx, y1:y1+self.cropy]


class HWC2CHW:
    def __call__(self, img):
        return hwc2chw(img)


class CHW2HWC:
    def __call__(self, img):
        return chw2hwc(img)

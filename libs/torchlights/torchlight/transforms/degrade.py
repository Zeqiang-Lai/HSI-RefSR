import os

import numpy as np
from scipy import ndimage
import scipy
import cv2
from scipy.io import loadmat

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1),
                         np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


class AbstractBlur:
    def __call__(self, img):
        img_L = ndimage.filters.convolve(
            img, np.expand_dims(self.kernel, axis=2), mode='nearest')
        return img_L


class GaussianBlur(AbstractBlur):
    """ Expect input'shape = [H,W,C]
    """

    def __init__(self, ksize=8, sigma=3):
        self.kernel = fspecial_gaussian(ksize, sigma)


class UniformBlur(AbstractBlur):
    """ Expect input'shape = [H,W,C]
    """

    def __init__(self, ksize):
        self.kernel = np.ones((ksize, ksize)) / (ksize*ksize)


## -------------------- Downsample -------------------- ##

class KFoldDownsample:
    ''' k-fold downsampler:
        Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

        Expect input'shape = [H,W,C]
    '''

    def __init__(self, sf):
        self.sf = sf

    def __call__(self, img):
        """ input: [H,W,c] """
        st = 0
        return img[st::self.sf, st::self.sf, :]


class AbstractDownsample:
    def __call__(self, img):
        img = self.blur(img)
        img = self.downsampler(img)
        return img


class UniformDownsample(AbstractDownsample):
    """ Expect input'shape = [H,W,C]
    """

    def __init__(self, sf):
        self.sf = sf
        self.blur = UniformBlur(sf)
        self.downsampler = KFoldDownsample(sf)


class GaussianDownsample(AbstractDownsample):
    """ Expect input'shape = [H,W,C]
    """

    def __init__(self, sf, ksize=8, sigma=3):
        self.sf = sf
        self.blur = GaussianBlur(ksize, sigma)
        self.downsampler = KFoldDownsample(sf)



class Resize:
    """ Expect input'shape = [H,W,C]
    """

    def __init__(self, sf, mode='cubic'):
        self.sf = sf
        self.mode = self.mode_map[mode]

    def __call__(self, img):
        return cv2.resize(img, (int(img.shape[1]*self.sf), int(img.shape[0]*self.sf)), interpolation=self.mode)

    mode_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA
    }


Upsample = Resize # backward capability

class HSI2RGB(object):
    def __init__(self, spe=None):
        if spe is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            self.SPE = loadmat(os.path.join(CURRENT_DIR, 'kernels', 'misr_spe_p.mat'))['P']  # (3,31)
        else:
            self.SPE = spe

    def __call__(self, img):
        return img @ self.SPE.transpose()

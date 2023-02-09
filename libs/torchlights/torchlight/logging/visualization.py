from typing import Union
import numpy as np
import torch
import torch.fft as fft


def img2fft_np(img):
    dft = np.fft.fftn(img, axes=(-2, -1))
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(np.abs(dft_shift))
    return magnitude_spectrum


def img2fft_torch(img):
    dft = fft.fftn(img, dim=(-2, -1))
    dft_shift = fft.fftshift(dft, dim=(-2, -1))
    magnitude_spectrum = 20*torch.log(torch.abs(dft_shift))
    return magnitude_spectrum


def img2fft(img: Union[torch.Tensor, np.ndarray]):
    """ img's shape [...,W,H] """

    if isinstance(img, torch.Tensor):
        return img2fft_torch(img)
    elif isinstance(img, np.ndarray):
        return img2fft_np(img)
    else:
        raise ValueError('unsupported data type ' + str(type(img)))

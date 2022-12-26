import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from ._util import enable_batch_input, torch2numpy, bandwise, CHW2HWC

__all__ = [
    'psnr',
    'ssim',
    'sam',
    'mpsnr',
    'mssim'
]

# raw psnr, ssim, sam assume HWC format


@torch2numpy
@enable_batch_input()
@CHW2HWC
def psnr(output, target, data_range=1):
    return peak_signal_noise_ratio(target, output, data_range=data_range)


@torch2numpy
@enable_batch_input()
@CHW2HWC
def ssim(img1, img2, **kwargs):
    return structural_similarity(img1, img2, multichannel=True, **kwargs)


@torch2numpy
@enable_batch_input()
@CHW2HWC
def sam(img1, img2, eps=1e-8):
    """
    Spectral Angle Mapper which defines the spectral similarity between two spectra
    """
    tmp1 = np.sum(img1*img2, axis=2) + eps
    tmp2 = np.sqrt(np.sum(img1**2, axis=2)) + eps
    tmp3 = np.sqrt(np.sum(img2**2, axis=2)) + eps
    tmp4 = tmp1 / tmp2 / tmp3
    angle = np.arccos(tmp4.clip(-1, 1))
    return np.mean(np.real(angle))


@torch2numpy
@enable_batch_input()
@bandwise
def mpsnr(output, target, data_range=1):
    return peak_signal_noise_ratio(target, output, data_range=data_range)


@torch2numpy
@enable_batch_input()
@bandwise
def mssim(img1, img2, **kwargs):
    return structural_similarity(img1, img2, **kwargs)

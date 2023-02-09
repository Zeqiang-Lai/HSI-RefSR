import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt


def imread(path):
    """ a wrapper of imageio.imread, which removes the meta data.
    """
    img = imageio.imread(path)
    img = np.array(img)
    return img


def imwrite(img, path):
    """ a wrapper of imageio.imwrite 
    """
    imageio.imwrite(img, path)


def imshow(img):
    plt.imshow(img)


def hwc2chw(img):
    return img.transpose(2, 0, 1)


def uint2float(img):
    return img.astype('float') / 255


def float2uint(img):
    return (img * 255).astype('uint8')


def rgb2bgr(img):
    """ input [H,W,C] """
    return img[:, :, ::-1]


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.
    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img

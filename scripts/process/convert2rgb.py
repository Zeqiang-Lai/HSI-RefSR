from scipy.io import loadmat, savemat
import os
from imageio import imsave
import numpy as np

names = os.listdir('img1/mat')

for name in sorted(names):
    img1 = loadmat(os.path.join('img1/hsi', name))['hr']
    img2 = loadmat(os.path.join('img2/hsi', name))['hr']

    # normalize
    img1 = img1.astype('float')
    img2 = img2.astype('float')
    img1 = (img1 - np.min(img1)) / (np.max(img1)-np.min(img1))
    img2 = (img2 - np.min(img2)) / (np.max(img2)-np.min(img2))

    # 46 608.09nm (46-15)/2 = 15
    # 33 541.79nm (33-15)/2 = 9
    # 22 486.26nm (22-15)/2 = 3

    rgb1 = img1[:, :, [15, 8, 3]]
    rgb2 = img2[:, :, [15, 8, 3]]
    rgb1 = (rgb1.clip(0,1) * 255).astype('uint8')
    rgb2 = (rgb2.clip(0,1) * 255).astype('uint8')

    print(name, rgb1.shape, rgb2.shape, np.max(rgb1), np.max(rgb2))

    imsave(os.path.join('img1/rgb', name[:-4]+'.png'), rgb1)
    imsave(os.path.join('img2/rgb', name[:-4]+'.png'), rgb2)

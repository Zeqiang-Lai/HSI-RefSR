import numpy as np

srf = [[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019,
        0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007,
        0.012, 0.013, 0.015, 0.016, 0.017, 0.02, 0.013, 0.011, 0.009, 0.005,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
       [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022,
        0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]]

srf = np.array(srf).astype(np.float32)


from scipy.io import loadmat, savemat
import os
from imageio import imsave
import numpy as np

names = os.listdir('dataset/hsi1')

for name in sorted(names):
    img1 = loadmat(os.path.join('dataset/hsi1', name))['hr']
    img2 = loadmat(os.path.join('dataset/hsi2', name))['hr']

    # normalize
    img1 = img1.astype('float')
    img2 = img2.astype('float')
    img1 = (img1 - np.min(img1)) / (np.max(img1)-np.min(img1))
    img2 = (img2 - np.min(img2)) / (np.max(img2)-np.min(img2))

    # 46 608.09nm (46-15)/2 = 15
    # 33 541.79nm (33-15)/2 = 9
    # 22 486.26nm (22-15)/2 = 3

    rgb1 = img1 @ srf.T
    rgb2 = img2 @ srf.T
    rgb1 = (rgb1.clip(0,1) * 255).astype('uint8')
    rgb2 = (rgb2.clip(0,1) * 255).astype('uint8')

    print(name, rgb1.shape, rgb2.shape, np.max(rgb1), np.max(rgb2))

    imsave(os.path.join('dataset/rgb1-srf', name[:-4]+'.png'), rgb1)
    imsave(os.path.join('dataset/rgb2-srf', name[:-4]+'.png'), rgb2)

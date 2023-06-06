from hdf5storage import loadmat
import os
import numpy as np

names = os.listdir('img1/mat')

for name in names:
    img1 = loadmat(os.path.join('img1/mat', name))['tensor']
    img2 = loadmat(os.path.join('img2/mat', name))['tensor']
    print(np.mean(np.abs(img1-img2)))
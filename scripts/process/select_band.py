from scipy.io import loadmat, savemat
import os
import numpy as np

names = os.listdir('img1/mat')

for name in sorted(names):
    img1 = loadmat(os.path.join('img1/mat', name))['tensor']
    img2 = loadmat(os.path.join('img2/mat', name))['tensor']
    
    img1 = img1[:,:,15:15+31*2:2]
    img2 = img2[:,:,15:15+31*2:2]
    
    print(name, img1.shape, img2.shape)
    
    savemat(os.path.join('img1/hsi', name), {'hr': img1})
    savemat(os.path.join('img2/hsi', name), {'hr': img2})
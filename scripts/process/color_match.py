from skimage import exposure

import os
import cv2
from scipy.io import loadmat, savemat

names = os.listdir('cvpr31/img1')

for name in names:
    print(name)
    name, _ = os.path.splitext(name)
    
    rgb1 = cv2.imread(os.path.join('cvpr31/img1', name+'.png'))
    rgb2 = cv2.imread(os.path.join('cvpr31/img2', name+'.png'))
    
    rgb2_cmatch = exposure.match_histograms(rgb2, rgb1, multichannel=True)
    
    cv2.imwrite(os.path.join('cvpr31/img2-cmatch', name+'.png'), rgb2_cmatch)
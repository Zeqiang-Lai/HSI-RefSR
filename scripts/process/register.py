from pathlib import Path
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
from skimage import exposure

from torchlight.utils.registration import Align

al = Align()

root = Path('.')

names = os.listdir(root / 'img1' / 'rgb')

total_psnr = 0
total_img1_psnr = 0

for name in tqdm(['56.png']):
    img1 = cv2.imread(os.path.join(root, 'img1/rgb2', name))
    img2 = cv2.imread(os.path.join(root, 'img2/rgb2', name)) 
    # img1 = exposure.match_histograms(img1, img2, multichannel=True)
    
    try:
        warp_img1, M = al.align_image(img1*2, img2)
        warp_img1 = al.warp_image(img1, img2, M)
        cv2.imwrite(os.path.join('fail', name), warp_img1)
        
        np.save(os.path.join('affine', name[:-4]+'.npy'), M)
    except:
        print(name)
        
    total_psnr += peak_signal_noise_ratio(warp_img1, img2)
    total_img1_psnr += peak_signal_noise_ratio(warp_img1, img1)
    
print(total_psnr / len(names))
print(total_img1_psnr / len(names))
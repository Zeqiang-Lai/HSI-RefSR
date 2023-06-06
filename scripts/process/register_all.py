from pathlib import Path
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

from torchlight.utils.registration import Align

al = Align()

root = Path('.')

names = os.listdir(root / 'affine2')

total_psnr = 0
total_img1_psnr = 0

for name in tqdm(names):
    name, _ = os.path.splitext(name)
    img1 = cv2.imread(os.path.join(root, 'img1/rgb2', name+'.png'))
    img2 = cv2.imread(os.path.join(root, 'img2/rgb2', name+'.png')) 

    M = np.load(os.path.join('affine2', name+'.npy'))
    warp_img1 = al.warp_image(img1, img2, M)
    cv2.imwrite(os.path.join('warp2', name+'.png'), warp_img1)
    
    total_psnr += peak_signal_noise_ratio(warp_img1, img2)
    total_img1_psnr += peak_signal_noise_ratio(warp_img1, img1)
    
print(total_psnr / len(names))
print(total_img1_psnr / len(names))
from pathlib import Path
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import ast
from scipy.io import loadmat, savemat

from torchlight.utils.registration import Align

al = Align()

names = os.listdir('dataset/stat/affine')

total_psnr = 0

def to_image_format(data):
    data = data.astype('float')
    data = (data - data.min()) / (data.max()-data.min())
    data = (data*255).astype('uint8')
    return data

for name in tqdm(names):
    name, _ = os.path.splitext(name)
    img1 = cv2.imread(os.path.join('img1/rgb2', name+'.png'))
    img2 = cv2.imread(os.path.join('img2/rgb2', name+'.png')) 

    M = np.load(os.path.join('dataset/stat/affine', name+'.npy'))
    warp_img1 = al.warp_image(img1, img2, M)
    
    with open(os.path.join('dataset/stat/coords', name+'.txt'), 'r') as f:
        content = f.read().strip()
        coords = ast.literal_eval(content)
    warp_img1 = warp_img1[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0], :]
    
    img2 = img2[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0], :]
    cv2.imwrite(os.path.join('dataset/rgb1', name+'.png'), warp_img1)
    cv2.imwrite(os.path.join('dataset/rgb2', name+'.png'), img2)
    # cv2.imwrite(os.path.join('dataset/cat', name+'.png'), np.hstack([warp_img1, img2]))
    
    # total_psnr += peak_signal_noise_ratio(warp_img1, img2)
    
    # hsi
    hsi1 = loadmat(os.path.join('img1/hsi', name+'.mat'))['hr']
    hsi2 = loadmat(os.path.join('img2/hsi', name+'.mat'))['hr']
    warp_hsi1 = al.warp_image(hsi1, hsi2, M)
    warp_hsi1 = warp_hsi1[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0], :]
    hsi2 = hsi2[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0], :]
    cv2.imwrite(os.path.join('dataset/cat_hsi', name+'.png'),
                np.hstack([to_image_format(warp_hsi1[:,:,0]), to_image_format(hsi2[:,:,0])]))
    
    savemat(os.path.join('dataset/hsi1',name+'.mat'), {'hr': warp_hsi1})
    savemat(os.path.join('dataset/hsi2',name+'.mat'), {'hr': hsi2})
    
print(total_psnr / len(names))
import os
import cv2
from scipy.io import loadmat, savemat

names = os.listdir('dataset/rgb1')


def crop_center(img, croph, cropw):
    h, w = img.shape[0], img.shape[1]
    starth = h//2-(croph//2)
    startw = w//2-(cropw//2)
    return img[starth:starth+croph, startw:startw+cropw, :]

for name in names:
    name, _ = os.path.splitext(name)
    
    rgb1 = cv2.imread(os.path.join('dataset/rgb1', name+'.png'))
    rgb2 = cv2.imread(os.path.join('dataset/rgb2-cmatch', name+'.png'))
    
    hsi1 = loadmat(os.path.join('dataset/hsi1', name+'.mat'))['hr']
    hsi2 = loadmat(os.path.join('dataset/hsi2', name+'.mat'))['hr']
    
    H,W,_ = rgb1.shape
    
    H = H % 64
    W = W % 64
    
    # rgb1 = rgb1[:H,:W,:]
    # rgb2 = rgb2[:H,:W,:]
    
    # hsi1 = hsi1[:H,:W,:]
    # hsi2 = hsi2[:H,:W,:]

    rgb1 = rgb1[H:,W:,:]
    rgb2 = rgb2[H:,W:,:]
    hsi1 = hsi1[H:,W:,:]
    hsi2 = hsi2[H:,W:,:]
    
    
    # rgb1 = crop_center(rgb1, 320, 512)
    # rgb2 = crop_center(rgb2, 320, 512)
    # hsi1 = crop_center(hsi1, 320, 512)
    # hsi2 = crop_center(hsi2, 320, 512)
    
    print(rgb1.shape, hsi1.shape)
    
    rgb1_dir = 'cvpr31-2/img1/HR'
    rgb2_dir = 'cvpr31-2/img2-cmatch/HR'
    hsi1_dir = 'cvpr31-2/img1_hsi/HR'
    hsi2_dir = 'cvpr31-2/img2_hsi/HR'
    os.makedirs(rgb1_dir, exist_ok=True)
    os.makedirs(rgb2_dir, exist_ok=True)
    os.makedirs(hsi1_dir, exist_ok=True)
    os.makedirs(hsi2_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(rgb1_dir, name+'.png'), rgb1)
    cv2.imwrite(os.path.join(rgb2_dir, name+'.png'), rgb2)
    
    savemat(os.path.join(hsi1_dir, name+'.mat'), {'gt': hsi1})
    savemat(os.path.join(hsi2_dir, name+'.mat'), {'gt': hsi2})
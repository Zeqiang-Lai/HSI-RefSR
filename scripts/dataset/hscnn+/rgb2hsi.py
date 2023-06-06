from __future__ import division
import torch
import torch.nn as nn

import os
import numpy as np
from imageio import imread

from resblock import resblock, conv_relu_res_relu_block
from utils import save_matv73, reconstruction


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


model_path = 'res_n16_cleanpng.pkl'
img_path = '/media/exthdd/datasets/refsr/LF_Flowers_Dataset/processed/3_3/HR'
result_path = '/media/exthdd/datasets/refsr/LF_Flowers_Dataset/processed/3_3_hsi/HR'
var_name = 'gt'

os.makedirs(result_path, exist_ok=True)

save_point = torch.load(model_path)
model_param = save_point['state_dict']
model = resblock(conv_relu_res_relu_block, 16, 3, 31)
model = nn.DataParallel(model)
model.load_state_dict(model_param)

model = model.cuda()
model.eval()


def rgb2hsi(rgb):
    rgb = rgb/255
    rgb = np.expand_dims(np.transpose(rgb, [2, 1, 0]), axis=0).copy()

    img_res1 = reconstruction(rgb, model)
    img_res2 = np.flip(reconstruction(np.flip(rgb, 2).copy(), model), 1)
    img_res3 = (img_res1+img_res2)/2
    return img_res3


names = os.listdir(img_path)
for idx, img_name in enumerate(sorted(names)):
    if img_name.startswith('.'):
        continue
    print(idx, len(names), img_name)
    img_path_name = os.path.join(img_path, img_name)
    mat_name = img_name[:-4] + '.mat'
    mat_path = os.path.join(result_path, mat_name)
    
    if os.path.exists(mat_path):
        continue

    input = imread(img_path_name)
    w, h, c = input.shape
    out = np.zeros((w, h, 31))

    for i in range((w+511)//512):
        for j in range((h+511)//512):
            rgb = input[i*512:i*512+512, j*512:j*512+512, :]
            img_res3 = rgb2hsi(rgb)
            out[i*512:i*512+512, j*512:j*512+512, :] = img_res3

    # print(out.shape)
   
    out = minmax_normalize(out)
    save_matv73(mat_path, var_name, out)

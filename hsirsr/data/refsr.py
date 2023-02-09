import os
from pathlib import Path

import imageio
import numpy as np
import torch.utils.data as data
from hdf5storage import loadmat
from matplotlib.image import imread
from torchlight.transforms.functional import minmax_normalize
from torchlight.transforms.stateful import RandCrop

from .transform import SRDegrade


def imread(path):
    img = imageio.imread(path)
    img = np.array(img).astype('float')
    img = img / 255
    return img


def hwc2chw(img):
    return img.transpose(2, 0, 1)



srf = [[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019,
        0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007,
        0.012, 0.013, 0.015, 0.016, 0.017, 0.02, 0.013, 0.011, 0.009, 0.005,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
       [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022,
        0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]]

srf = np.array(srf).astype(np.float32)

class OnDemandDataset(data.Dataset):
    """ Generate LR on demand 
        Require:
            - HSI HR
            - HSI RGB HR
            - Ref RGB HR
    """

    def __init__(self, root, input, ref, names_path, sf, mat_key='gt', crop_size=None, repeat=1):
        root = Path(root)
        self.hsi_dir = root / (input+'_hsi') / 'HR'
        self.hsi_rgb_dir = root / input / 'HR'
        self.rgb_dir = root / ref / 'HR'
        self.mat_key = mat_key
        names_path = root / names_path

        # get the name of all images
        if names_path is None:
            self.names = os.listdir(os.path.join(self.rgb_dir))
        else:
            with open(names_path, 'r') as f:
                names = f.readlines()
                self.names = [n.strip() for n in names]

        self.loadmat = loadmat
        self.imread = imread
        self.degrade = SRDegrade(sf)
        
        self.crop_size = crop_size
        self.repeat = repeat
        self.names = self.names * repeat

    def __len__(self):
        return len(self.names) 

    def __getitem__(self, index):
        name = self.names[index]

        hsi_hr = self.loadmat(str(self.hsi_dir / (name+'.mat')))[self.mat_key]
        hsi_hr = minmax_normalize(hsi_hr.astype('float'))
        hsi_lr = self.degrade(hsi_hr)

        hsi_rgb_hr = self.imread(self.hsi_rgb_dir / (name+'.png'))
        hsi_rgb_lr = self.degrade(hsi_rgb_hr)

        rgb_hr = self.imread(self.rgb_dir / (name+'.png'))
        rgb_lr = self.degrade(rgb_hr)

        output = hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr
        output = tuple(hwc2chw(o) for o in output)
        
        if self.crop_size:
            H = output[0].shape[1]
            W = output[0].shape[2]
            crop_fn = RandCrop((H,W),self.crop_size)
            output =  tuple(crop_fn(o) for o in output)
            
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = output

        hsi_hr = hsi_hr[None]
        hsi_lr = hsi_lr[None]
        return hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr

class SRFDataset(data.Dataset):
    """ Generate LR on demand 
        Require:
            - HSI HR
            - HSI RGB HR
            - Ref RGB HR
    """

    def __init__(self, root, input, ref, names_path, sf, 
                 mat_key='gt', crop_size=None, repeat=1, use_cache=False, sr_input=None):
        root = Path(root)
        self.hsi_inp_dir = root / (input+'_hsi') / 'HR'
        self.hsi_rgb_dir = root / (ref+'_hsi') / 'HR'
        self.mat_key = mat_key
        names_path = root / names_path

        # get the name of all images
        if names_path is None:
            self.names = os.listdir(os.path.join(self.hsi_inp_dir))
        else:
            with open(names_path, 'r') as f:
                names = f.readlines()
                self.names = [n.strip() for n in names]

        self.loadmat = loadmat
        self.imread = imread
        self.degrade = SRDegrade(sf)
        
        self.crop_size = crop_size
        self.repeat = repeat
        self.names = self.names * repeat
        
        self.use_cache = use_cache
        self.cache = {}
        
        # if not none, use results in sr_input as the hsi_lr
        self.sr_input = sr_input
        
    def __len__(self):
        return len(self.names) 

    def get_mat(self, path):
        # if self.use_cache:
        #     if path not in self.cache:
        #         self.cache[path] = self.loadmat(path)
        #     return self.cache[path]
        return self.loadmat(path)
    
    def __getitem__(self, index):
        name = self.names[index]

        data = self.get_mat(str(self.hsi_inp_dir / (name+'.mat')))
        hsi_hr = data[self.mat_key]
        hsi_hr = minmax_normalize(hsi_hr.astype('float'))
        if 'lr' in data.keys():
            hsi_lr = data['lr'].transpose(1,2,0)
        else:
            hsi_lr = self.degrade(hsi_hr)

        hsi_rgb_hr = hsi_hr @ srf.T
        hsi_rgb_lr = hsi_lr @ srf.T 

        rgb_hsi_hr = self.get_mat(str(self.hsi_rgb_dir / (name+'.mat')))[self.mat_key]
        rgb_hsi_hr = minmax_normalize(rgb_hsi_hr.astype('float'))

        rgb_hsi_lr = self.degrade(rgb_hsi_hr)
        rgb_hr = rgb_hsi_hr @ srf.T
        rgb_lr = rgb_hsi_lr @ srf.T

        output = hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr, rgb_hsi_hr
        from skimage import exposure
        hsi_rgb_lr = exposure.match_histograms(hsi_rgb_lr, rgb_lr, channel_axis=2)
        output = tuple(hwc2chw(o) for o in output)
        
        if self.crop_size:
            H = output[0].shape[1]
            W = output[0].shape[2]
            crop_fn = RandCrop((H,W),self.crop_size)
            output =  tuple(crop_fn(o) for o in output)
            
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr, rgb_hsi_hr = output

        hsi_hr = hsi_hr[None]
        hsi_lr = hsi_lr[None]
        rgb_hsi_hr = rgb_hsi_hr[None]
        return hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr

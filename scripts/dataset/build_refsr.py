import os
from pathlib import Path
from torchlight.transforms.degrade import Upsample

import torch
import cv2
from hdf5storage import loadmat, savemat
from torchvision.transforms import Compose
from torchvision.transforms.transforms import ToTensor

from torchlight.transforms import GaussianDownsample
from refsr.model.sisr.qrnn3d import qrnn_16_5

root = Path('/media/exthdd/datasets/refsr/LF_Flowers_Dataset/refsr_hsi/train')

hsi_hr_dir = root / 'hsi_hr'
hsi_rgb_dir = root / 'hsi_rgb'
ref_hr_dir = root / 'ref_hr'

hsi_lr_sr_dir = root / 'hsi_lr_sr'
hsi_rgb_lr_dir = root / 'hsi_rgb_lr'
ref_lr_dir = root / 'ref_lr'

for d in [hsi_lr_sr_dir, hsi_rgb_lr_dir, ref_lr_dir]:
    d.mkdir(exist_ok=True)

names = os.listdir(hsi_rgb_dir)


device = torch.device('cuda:0')
model = qrnn_16_5().to(device)
ckpt_path = '/home/laizeqiang/Desktop/lzq/projects/ref-sr/hsi_ref_sr/saved/sisr_flower_qrnn3d/sf248/ckpt/model-best.pth'
model.load_state_dict(torch.load(ckpt_path)['module']['model'])
        
downsample = Compose([GaussianDownsample(2, ksize=8, sigma=3), Upsample(2)])
downsample_torch = Compose([lambda x: x.transpose(1,2,0),
                            downsample,
                            ToTensor(),
                            lambda x: x.unsqueeze(0).unsqueeze(0).float()])

def process_rgb(name, source, target):
    rgb = cv2.imread(os.path.join(source, name))
    lr = downsample(rgb)
    cv2.imwrite(os.path.join(target, name), lr)
    # print(lr.shape)
    cv2.imwrite('rgb.png', lr)

def process_hsi(name):
    mat = loadmat(os.path.join(hsi_hr_dir, name[:-4]+'.mat'))
    hsi_hr = mat['gt']
    hsi_hr = hsi_hr
    low = downsample_torch(hsi_hr).to(device)
    with torch.no_grad():
        out = model(low)
        
    img = out[0,0,20,:,:].detach().cpu().numpy().clip(0,1) * 255
    img = img.astype('uint8')
    cv2.imwrite('test.png', img)
    
    img = low[0,0,20,:,:].detach().cpu().numpy().clip(0,1) * 255
    img = img.astype('uint8')
    cv2.imwrite('low.png', img)
    
    print(out.shape)
    savemat(os.path.join(hsi_lr_sr_dir, name[:-4]+'.mat'), {'lr': low.squeeze().detach().cpu().numpy(), 
                                                            'sr': out.squeeze().detach().cpu().numpy()})

# hsi_lr, hsi_sr, hsi_rgb, ref_hr, ref_lr
for idx, name in enumerate(names):
    print('{}|{} {}'.format(idx, len(names), name))
    
    process_hsi(name)
    process_rgb(name, hsi_rgb_dir, hsi_rgb_lr_dir)
    process_rgb(name, ref_hr_dir, ref_lr_dir)
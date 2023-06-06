
from pathlib import Path
import os
import cv2
import numpy as np

from torchlight.utils.registration import Align

al = Align()


root = Path('warp2')
save_root = Path('crop')

names = os.listdir(root)
names = sorted(names)

idx = 0
quit = False

while True:
    name = names[idx]
    print(name)
    img_path = root / name
    img = al.read_image(str(img_path))
    pure_name, _ = os.path.splitext(name)

    coords = []
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, ' ', y)
            coords.append((x,y))
        
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    
    while True:
        k = cv2.waitKey(0)
        # print('press ', k)
        if k == 32: # space
            i = img[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0], :]
            cv2.imwrite(os.path.join(save_root, 'img', pure_name+'.png'), i)
            with open(os.path.join(save_root, 'coords', pure_name+'.txt'), 'w') as f:
                f.write(str(coords))
            idx = idx + 1
            idx = min(len(names)-1, idx)
            break
        if k == 81: # left
            idx = idx -1
            idx = max(0, idx)
            break
        if k == 83: # right
            idx = idx + 1
            idx = min(len(names)-1, idx)
            break
        if k == 113: # q
            quit = True
            break
    if quit:
        break
cv2.destroyAllWindows()
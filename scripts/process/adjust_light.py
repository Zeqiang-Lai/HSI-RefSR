import cv2
import os

# name = '51.png'
# path = 'img2/rgb/' + name

# img = cv2.imread(path)
# img = img.astype('float') * 4
# img = img.clip(0,255).astype('uint8')
# cv2.imwrite('img2/rgb2/'+name, img)


import shutil
for name in os.listdir('img2/rgb'):
    if os.path.exists(os.path.join('img2/rgb2', name)):
        continue
    shutil.copy(os.path.join('img2/rgb', name), os.path.join('img2/rgb2', name))
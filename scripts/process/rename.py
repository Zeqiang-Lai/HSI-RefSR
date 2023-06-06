import os
import shutil


path = 'pkg4/img2/mat'

for idx, name in enumerate(sorted(os.listdir(path))):
    fpath = os.path.join(path, name)
    base, ext = os.path.splitext(name)
    npath = os.path.join(path, f'{idx+57}{ext}')
    shutil.move(fpath, npath)
    print(idx, npath)
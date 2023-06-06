import os

import numpy as np
from hdf5storage import loadmat
from numpy.core.fromnumeric import reshape
from torchlight.transforms.functional import minmax_normalize


def create_stats_train(out_name, dataroot, matkey):
    mean = 0
    std = 0
    fns = os.listdir(dataroot)
    for i, fn in enumerate(fns):
        X = loadmat(os.path.join(dataroot, fn))[matkey]
        X = minmax_normalize(X).transpose(2,0,1)
        X = reshape(X, (X.shape[0], X.shape[-1]*X.shape[-2]))
        mean += np.mean(X, axis=1)
        std += np.std(X, axis=1)
        print(f'{i}|{len(fns)}: {fn} - {X.shape}')
    print('done')
    mean = mean.astype(np.float32) / len(fns)
    std = std.astype(np.float32) / len(fns)
    print(mean.shape)
    np.savez(out_name + '_stats.npz', data_mean=mean, data_std=std, dtype='float32')


if __name__ == '__main__':
    root = '/media/exthdd/datasets/refsr/real-fusion/cvpr/img1_hsi/HR'
    create_stats_train('real', root, 'gt')

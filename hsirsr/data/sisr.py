import os
from functools import partial

import numpy as np
import torch
import torch.utils.data as data
from hdf5storage import loadmat
from torchlight.transforms import CenterCrop
from torchlight.transforms.functional import minmax_normalize

from .transform import SRDegrade


class Transform:
    def __init__(self, sf, use_2dconv):
        self.degrade = SRDegrade(sf)
        self.hsi2tensor = partial(hsi2tensor, use_2dconv=use_2dconv)

    def _get_lr_sr_(self, hr):
        tmp = hr.transpose(1, 2, 0)
        lr = self.degrade.down(tmp)
        sr = self.degrade.up(lr)
        lr = lr.transpose(2, 0, 1)
        sr = sr.transpose(2, 0, 1)
        return lr, sr

    def __call__(self, hr):
        hr = hr.astype('float')
        hr = minmax_normalize(hr)
        lr, sr = self._get_lr_sr_(hr)
        lr, sr, hr = map(self.hsi2tensor, (lr, sr, hr))
        return lr, sr, hr


class TrainDataset(data.Dataset):
    def __init__(self, root, sf, use_2dconv=False):
        super().__init__()
        self.dataset = LMDBDataset(root)
        self.tsfm = Transform(sf, use_2dconv)

    def __getitem__(self, index):
        hr = self.dataset.__getitem__(index)
        lr, sr, hr = self.tsfm(hr)
        return lr, sr, hr

    def __len__(self):
        return len(self.dataset)


class TestDataset(data.Dataset):
    def __init__(self, root, sf, size=None, crop_size=None, hr_key='gt', use_2dconv=False, fns=None):
        super().__init__()
        self.dataset = MatDataFromFolder(root, size=size, fns=fns, attach_filename=True)
        self.tsfm = Transform(sf, use_2dconv)
        self.crop = None if crop_size is None else CenterCrop(crop_size)
        self.key = hr_key

    def __getitem__(self, index):
        mat, filename = self.dataset.__getitem__(index)
        hr = mat[self.key].transpose(2,0,1)
        lr, sr, hr = self.tsfm(hr)
        if self.crop:
            lr, sr, hr = map(self.crop, (lr, sr, hr))
        return lr, sr, hr, filename

    def __len__(self):
        return len(self.dataset)


# ---------------------------------------------------------------------------- #
#                                     Utils                                    #
# ---------------------------------------------------------------------------- #


def hsi2tensor(hsi, use_2dconv):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    if use_2dconv:
        img = torch.from_numpy(hsi)
    else:
        img = torch.from_numpy(hsi[None])
    return img.float()


class LMDBDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.length = self.txn.stat()['entries']
        self.repeat = repeat

    def __getitem__(self, index):
        import caffe
        index = index % self.length
        raw_datum = self.txn.get('{:08}'.format(index).encode('ascii'))

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        flat_x = np.fromstring(datum.data, dtype=np.float32)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)

        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class MatDataFromFolder(data.Dataset):
    """Wrap mat data from folder"""

    def __init__(self, dataroot, load=loadmat, suffix='mat',
                 fns=None, size=None, attach_filename=False):
        super(MatDataFromFolder, self).__init__()
        self.load = load
        self.dataroot = dataroot

        if fns:
            with open(fns, 'r') as f:
                self.fns = [l.strip()+'.mat' for l in f.readlines()]
        else:
            self.fns = list(filter(lambda x: x.endswith(suffix), os.listdir(dataroot)))

        if size and size <= len(self.fns):
            self.fns = self.fns[:size]

        self.attach_filename = attach_filename

    def __getitem__(self, index):
        fn = self.fns[index]
        mat = self.load(os.path.join(self.dataroot, fn))
        if self.attach_filename:
            fn, _ = os.path.splitext(fn)
            fn = os.path.basename(fn)
            return mat, fn
        return mat

    def __len__(self):
        return len(self.fns)

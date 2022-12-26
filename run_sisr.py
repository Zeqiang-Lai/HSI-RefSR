import torch.utils.data as data
from torchlight.trainer.config import basic_args
from torchlight.trainer.entry import run

from hsirsr.data.sisr import TestDataset, TrainDataset
from hsirsr.module.sisr import BaseModule


def get_dataloaders(train_root, test_root, sf, batch_size, use_2dconv, key, test_fns, crop_size=None):
    train_dataset = TrainDataset(train_root, sf, use_2dconv)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size, shuffle=True,
                                   num_workers=4, pin_memory=True)

    mat_dataset = TestDataset(test_root, sf, None, crop_size, key, use_2dconv, test_fns)
    test_loader = data.DataLoader(mat_dataset, batch_size=1)

    mat_dataset = TestDataset(test_root, sf, 10, crop_size, key, use_2dconv, test_fns)
    valid_loader = data.DataLoader(mat_dataset, batch_size=1)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    args, cfg = basic_args('Single Hyperspectral Image Super-Resolution')

    train_loader, valid_loader, test_loader = get_dataloaders(**cfg['data'])
    module = BaseModule(**cfg['module'])

    run(args, cfg, module, train_loader, valid_loader, test_loader)

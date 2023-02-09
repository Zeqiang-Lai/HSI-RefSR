import torch.nn as nn
import torch.nn.init as init

from .ops.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

def init_params(net, init_type='kn'):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if init_type == 'kn':
                init.kaiming_normal_(m.weight, mode='fan_out')
            if init_type == 'ku':
                init.kaiming_uniform_(m.weight, mode='fan_out')
            if init_type == 'xn':
                init.xavier_normal_(m.weight)
            if init_type == 'xu':
                init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d)):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

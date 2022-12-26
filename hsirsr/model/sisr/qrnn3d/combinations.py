import torch
import torch.nn as nn
from torch.nn import functional
# from models.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

BatchNorm3d = nn.BatchNorm3d
# BatchNorm3d = SynchronizedBatchNorm3d


class BNReLUConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv3d, self).__init__()
#        self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=False))

    # def forward(self, x):
    #     out = super(BNReLUConv3d, self).forward(x)
    #     print(out.shape)
    #     return out


class BNReLUDeConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUDeConv3d, self).__init__()
#        self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=False))

    # def forward(self, x):
    #     out = super(BNReLUDeConv3d, self).forward(x)
    #     print(out.shape)
    #     return out


class BNReLUUpsampleConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1,2,2), inplace=True):
        super(BNReLUUpsampleConv3d, self).__init__()
#        self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('upsample_conv', UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))


class UpsampleConv3d(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, upsample=None):
        super(UpsampleConv3d, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='trilinear', align_corners=True)
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.conv3d(x_in)
        return out


class BNConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False):
        super(BNConv3d, self).__init__()
#        self.add_module('bn', BatchNorm3d(in_channels))        
#        if s == (1,2,2):
#            self.add_module('maxPooling', nn.MaxPool3d((1,2,2), (1,2,2), 0))
#            s = 1
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=bias))


class BNDeConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False):
        super(BNDeConv3d, self).__init__()
#        self.add_module('bn', BatchNorm3d(in_channels))        
        self.add_module('deconv', nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=bias))


class BNUpsampleConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1,2,2)):
        super(BNUpsampleConv3d, self).__init__()
#        self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('upsample_conv', UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))

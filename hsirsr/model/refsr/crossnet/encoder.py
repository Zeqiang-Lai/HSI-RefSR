import torch.nn as nn

from ..qrnn3d.layer import QRNNConv3D, QRNNDeConv3D, QRNNUpsampleConv3d, BiQRNNDeConv3D, BiQRNNConv3D


def conv_activation(in_ch, out_ch , kernel_size = 3, stride = 1, padding = 1, activation = 'relu', init_type = 'w_init_relu'):
    if activation == 'relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.ReLU(inplace = True))

    elif activation == 'leaky_relu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.LeakyReLU(negative_slope = 0.1 ,inplace = True ))

    elif activation == 'selu':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding),
                nn.SELU(inplace = True))

    elif activation == 'linear':
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding))


class RGBEncoder(nn.Module):

    def __init__(self,in_ch,activation = 'selu', init_type = 'w_init'):
        super(RGBEncoder, self).__init__()

        self.layer_f = conv_activation(in_ch, 64 , kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)

        self.conv1 = conv_activation(64, 64 , kernel_size = 5 ,stride = 1,padding = 2, activation = activation, init_type = init_type)

        self.conv2 = conv_activation(64, 64 , kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)

        self.conv3 = conv_activation(64, 64 , kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)

        self.conv4 = conv_activation(64, 64 , kernel_size = 5 ,stride = 2,padding = 2, activation = activation, init_type = init_type)


    def forward(self,x):

        layer_f = self.layer_f(x)
        conv1 = self.conv1(layer_f)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        return conv1,conv2,conv3,conv4


class HSIEncoder(nn.Module):
    def __init__(self, in_channels, channels, num_layer, sample_idx, QRNNConv3D=QRNNConv3D, bn=False, act='tanh'):
        super(HSIEncoder, self).__init__()
        # Encoder        
        self.feat_extractor = BiQRNNConv3D(in_channels, channels, bn=bn, act=act)
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            if i not in sample_idx:
                encoder_layer = QRNNConv3D(channels, channels, bn=bn, act=act)
            else:
                encoder_layer = QRNNConv3D(channels, 2*channels, k=3, s=(1,2,2), p=1, bn=bn, act=act)
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, reverse=False):
        xs = []

        x = self.feat_extractor(x)
        xs.append(x)
        
        num_layer = len(self.layers)
        for i in range(num_layer):
            x = self.layers[i](x, reverse=reverse)
            reverse = not reverse
            xs.append(x)            
    
        return xs, reverse



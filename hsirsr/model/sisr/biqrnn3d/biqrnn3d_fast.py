import torch
import torch.nn as nn

from .combinations import *


class BiFQRNNREDC3D(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx=None):
        super(BiFQRNNREDC3D, self).__init__()
        assert sample_idx is None or isinstance(sample_idx, list)
        if sample_idx is None:
            sample_idx = []
        self.feature_extractor = BiQRNNConv3D(in_channels, channels)
        self.encoder = BiQRNN3DEncoder(channels, num_half_layer, sample_idx)
        self.decoder = BiQRNN3DDecoder(channels*(2**len(sample_idx)), num_half_layer, sample_idx)
        self.reconstructor = BiQRNNDeConv3D(channels, in_channels, bias=True)

    def forward(self, x):
        xs = [x]
        out = self.feature_extractor(xs[0])
        xs.append(out)
        out = self.encoder(out, xs)
        out = self.decoder(out, xs)
        out = out + xs.pop()
        out = self.reconstructor(out)
        out = out + xs.pop()
        return out


class BiQRNN3DEncoder(nn.Module):
    """Quasi Recurrent 3D Encoder
    Args:
        downsample: downsample times, None denotes no downsample"""

    def __init__(self, channels, num_half_layer, sample_idx):
        super(BiQRNN3DEncoder, self).__init__()
        # Encoder
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = BiQRNNConv3D(channels, channels)
            else:
                encoder_layer = BiQRNNConv3D(channels, 2*channels, k=3, s=(1, 2, 2), p=1)
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer-1):
            x = self.layers[i](x)
            xs.append(x)
        x = self.layers[-1](x)
        return x


class BiQRNN3DDecoder(nn.Module):
    """Quasi Recurrent 3D Decoder
    Args:
        downsample: downsample times, None denotes no downsample"""

    def __init__(self, channels, num_half_layer, sample_idx):
        super(BiQRNN3DDecoder, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                decoder_layer = BiQRNNDeConv3D(channels, channels)
            else:
                decoder_layer = BiQRNNUpsampleConv3d(channels, channels//2)
                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        x = self.layers[0](x)
        for i in range(1, num_half_layer):
            x = x + xs.pop()
            x = self.layers[i](x)
        return x


class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        return Z.tanh(), F.sigmoid()

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, state=None, reverse=False):
        h = None if state is None else state  # unsqueeze dim to feed in _rnn_step
        Z, F = self._conv_step(inputs)
        h_time = []
        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        # return concatenated hidden states
        return torch.cat(h_time, dim=2)


class BiQRNN3DLayer(QRNN3DLayer):
    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F1, F2 = gates.split(split_size=self.hidden_channels, dim=1)
        return Z.tanh(), F1.sigmoid(), F2.sigmoid()

    def forward(self, inputs, state=None):
        h = None if state is None else state
        Z, F1, F2 = self._conv_step(inputs)
        hsl = []
        hsr = []
        zs = Z.split(1, 2)

        for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsl.append(h)

        h = None if state is None else state
        for time, (z, f) in enumerate((zip(
            reversed(zs), reversed(F2.split(1, 2))
        ))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)

        # return concatenated hidden states
        return torch.cat(hsl, dim=2) + torch.cat(hsr, dim=2)


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False):
        super(BiQRNNConv3D, self).__init__(
            in_channels, hidden_channels, BNConv3d(in_channels, hidden_channels*3, k, s, p, bias=bias))


class BiQRNNDeConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False):
        super(BiQRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, BNDeConv3d(in_channels, hidden_channels*3, k, s, p, bias=bias))


class BiQRNNUpsampleConv3d(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1, 2, 2)):
        super(BiQRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels, BNUpsampleConv3d(in_channels, hidden_channels*3, k, s, p, upsample))


class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, BNConv3d(in_channels, hidden_channels*2, k, s, p))


class QRNNDeConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1):
        super(QRNNDeConv3D, self).__init__(
            in_channels, hidden_channels, BNDeConv3d(in_channels, hidden_channels*2, k, s, p))


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1, 2, 2)):
        super(QRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels, BNUpsampleConv3d(in_channels, hidden_channels*2, k, s, p, upsample))

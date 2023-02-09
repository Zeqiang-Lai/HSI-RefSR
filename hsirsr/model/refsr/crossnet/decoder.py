import torch
import torch.nn as nn

from ..qrnn3d.layer import QRNNConv3D, QRNNDeConv3D, QRNNUpsampleConv3d, BiQRNNDeConv3D, BiQRNNConv3D


class HSIDecoder4(nn.Module):
    def __init__(self, channels, QRNNDeConv3D=QRNNDeConv3D, QRNNUpsampleConv3d=QRNNUpsampleConv3d, has_ad=True, bn=False, act='tanh'):
        super(HSIDecoder4, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()

        self.layers.append(QRNNUpsampleConv3d(128+64, 64, bn=bn, act=act))
        self.layers.append(QRNNUpsampleConv3d(64+64+64, 32, bn=bn, act=act))
        self.layers.append(QRNNUpsampleConv3d(32+32+64, 16, bn=bn, act=act))

        self.reconstructor = BiQRNNDeConv3D(16+16+64, 1, bias=True, bn=bn, act=act)

    def forward(self, hsi_feats, ref_feats_warp, reverse=False):
        t = ref_feats_warp[-1]
        BAND = hsi_feats[-1].shape[2]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])

        x = torch.cat([hsi_feats[-1], tmp], dim=1)  # 128+64 -> 64
        x = self.layers[0](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-2]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, hsi_feats[-2], tmp], dim=1)  # 64+64+64 -> 64
        x = self.layers[1](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-3]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, hsi_feats[-3], tmp], dim=1)  # 64+32+64 -> 64
        x = self.layers[2](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-4]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, hsi_feats[-4], tmp], dim=1)
        x = self.reconstructor(x)
        return x


class HSIDecoder8(nn.Module):
    def __init__(self, channels, QRNNDeConv3D=QRNNDeConv3D, QRNNUpsampleConv3d=QRNNUpsampleConv3d, has_ad=True, bn=False, act='tanh'):
        super(HSIDecoder8, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()

        self.layers.append(QRNNUpsampleConv3d(128+64, 64, bn=bn, act=act))
        self.layers.append(QRNNUpsampleConv3d(64+64+64, 32, bn=bn, act=act))
        self.layers.append(QRNNUpsampleConv3d(32+32+64, 16, bn=bn, act=act))

        self.reconstructor = BiQRNNDeConv3D(16+16+64, 1, bias=True, bn=bn, act=act)

    def forward(self, hsi_feats, ref_feats_warp, reverse=False):
        t = ref_feats_warp[-1]
        BAND = hsi_feats[-1].shape[2]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])

        x = torch.cat([hsi_feats[-1], tmp], dim=1)  # 128+64 -> 64
        x = self.layers[0](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-2]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, hsi_feats[-2]/2, tmp], dim=1)  # 64+64+64 -> 64
        x = self.layers[1](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-3]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, hsi_feats[-3]/4, tmp], dim=1)  # 64+32+64 -> 64
        x = self.layers[2](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-4]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, hsi_feats[-4]/8, tmp], dim=1)
        x = self.reconstructor(x)
        return x


class HSIDecoderSISR(nn.Module):
    def __init__(self, channels, QRNNDeConv3D=QRNNDeConv3D, QRNNUpsampleConv3d=QRNNUpsampleConv3d, has_ad=True, bn=False, act='tanh'):
        super(HSIDecoderSISR, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()

        self.layers.append(QRNNUpsampleConv3d(128, 64, bn=bn, act=act))
        self.layers.append(QRNNUpsampleConv3d(64+64, 32, bn=bn, act=act))
        self.layers.append(QRNNUpsampleConv3d(32+32, 16, bn=bn, act=act))

        self.reconstructor = BiQRNNDeConv3D(16+16, 1, bias=True, bn=bn, act=act)

    def forward(self, hsi_feats, reverse=False):

        x = hsi_feats[-1]  # 128
        x = self.layers[0](x, reverse=reverse)
        reverse = not reverse

        x = torch.cat([x, hsi_feats[-2]], dim=1)  # 64+64
        x = self.layers[1](x, reverse=reverse)
        reverse = not reverse

        x = torch.cat([x, hsi_feats[-3]], dim=1)
        x = self.layers[2](x, reverse=reverse)
        reverse = not reverse

        x = torch.cat([x, hsi_feats[-4]], dim=1)
        x = self.reconstructor(x)
        return x


class HSIDecoderSimple(nn.Module):
    def __init__(self, channels, QRNNDeConv3D=QRNNDeConv3D, QRNNUpsampleConv3d=QRNNUpsampleConv3d, has_ad=True, bn=False, act='tanh'):
        super(HSIDecoderSimple, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(QRNNUpsampleConv3d(channels+64, channels//2, bn=bn, act=act))
            channels = channels // 2

        self.reconstructor = BiQRNNDeConv3D(channels+64, 1, bias=True, bn=bn, act=act)

    def forward(self, hsi_feats, ref_feats_warp, reverse=False):
        BAND = hsi_feats[-1].shape[2]
        t = ref_feats_warp[-1]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([hsi_feats[-1], tmp], dim=1)
        x = self.layers[0](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-2]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, tmp], dim=1)
        x = self.layers[1](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-3]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, tmp], dim=1)
        x = self.layers[2](x, reverse=reverse)
        reverse = not reverse

        t = ref_feats_warp[-4]
        tmp = t.unsqueeze(2).expand(t.shape[0], t.shape[1], BAND, t.shape[2], t.shape[3])
        x = torch.cat([x, tmp], dim=1)
        x = self.reconstructor(x)
        return x

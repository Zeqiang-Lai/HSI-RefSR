from functools import partial

import torch.nn as nn
import torch

from .decoder import HSIDecoder4, HSIDecoder8, HSIDecoderSimple, HSIDecoderSISR
from .encoder import HSIEncoder, RGBEncoder
from .mask import MaskPredictor


class CrossNetHSI(nn.Module):
    def __init__(self, reweight=False, use_mask=True, use_pwc=False):
        super().__init__()
        self.flownet1 = self.flow_estimator(use_pwc)
        self.flownet2 = self.flow_estimator(use_pwc)
        self.warp = self.backwarp(use_pwc)
        self.rgb_encoder = RGBEncoder(3)
        self.hsi_encoder = HSIEncoder(1, 16, 3, [0, 1, 2])
        self.decoder = self.hsi_decoder(reweight)

        self.use_mask = use_mask
        if self.use_mask:
            self.mask_predictor = MaskPredictor(64, 32, 32)

    def hsi_decoder(self, reweight):
        if reweight:
            return HSIDecoder8(128)
        else:
            return HSIDecoder4(128)

    def flow_estimator(self, use_pwc):
        if use_pwc:
            from .pwcnet import PWCNet
            net = PWCNet()
            model = 'default'
            url = f'http://content.sniklaus.com/github/pytorch-pwc/network-{model}.pytorch'
            file_name = f'pwc-{model}'
            state_dict = torch.hub.load_state_dict_from_url(url=url, file_name=file_name)
            state_dict = {strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in state_dict.items()}
            net.load_state_dict(state_dict)
            for param in net.parameters():
                param.requires_grad = False
            return net
        else:
            from .flownet import FlowNet
            return FlowNet(3)

    def backwarp(self, use_pwc):
        if use_pwc:
            from .pwcnet import backwarp
            return backwarp
        else:
            from torchlight.nn.ops.warp import flow_warp
            return partial(flow_warp, padding_mode='border')

    def forward(self, hsi_sr, hsi_rgb_sr, ref_hr):
        ref_warp, flow = self.corase_align(hsi_rgb_sr, ref_hr)
        ref_feats_warp, masks = self.feat_align(hsi_rgb_sr, ref_warp)
        hsi_feats, reverse = self.hsi_encoder(hsi_sr)
        out = self.decoder(hsi_feats, ref_feats_warp, reverse)
        return out, ref_warp, flow, masks

    def corase_align(self, x, ref):
        flow = self.flownet1(x, ref)
        flow_12_1 = flow['flow_12_1']  # B, 2, W, H
        ref_warp = self.warp(ref, flow_12_1)
        return ref_warp, flow_12_1

    def feat_align(self, x, ref):
        flow = self.flownet2(x, ref)
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']
        ref_conv1, ref_conv2, ref_conv3, ref_conv4 = self.rgb_encoder(ref)
        warp_21_conv1 = self.warp(ref_conv1, flow_12_1)
        warp_21_conv2 = self.warp(ref_conv2, flow_12_2)
        warp_21_conv3 = self.warp(ref_conv3, flow_12_3)
        warp_21_conv4 = self.warp(ref_conv4, flow_12_4)

        masks = []
        if self.use_mask:
            x_conv1, x_conv2, x_conv3, x_conv4 = self.rgb_encoder(x)
            mask1 = self.mask_predictor(x_conv1, ref_conv1, flow_12_1)
            mask2 = self.mask_predictor(x_conv2, ref_conv2, flow_12_2)
            mask3 = self.mask_predictor(x_conv3, ref_conv3, flow_12_3)
            mask4 = self.mask_predictor(x_conv4, ref_conv4, flow_12_4)
            warp_21_conv1 = warp_21_conv1 * mask1
            warp_21_conv2 = warp_21_conv2 * mask2
            warp_21_conv3 = warp_21_conv3 * mask3
            warp_21_conv4 = warp_21_conv4 * mask4
            masks = [mask1, mask2, mask3, mask4]

        return (warp_21_conv1, warp_21_conv2, warp_21_conv3, warp_21_conv4), masks


class SISRAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = RGBEncoder(3)
        self.hsi_encoder = HSIEncoder(1, 16, 3, [0, 1, 2])
        self.decoder = HSIDecoderSISR(128)

    def forward(self, hsi_sr, hsi_rgb_sr, ref_hr):
        hsi_feats, reverse = self.hsi_encoder(hsi_sr)
        out = self.decoder(hsi_feats, reverse)
        return out


class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = RGBEncoder(3)
        self.hsi_encoder = HSIEncoder(1, 16, 3, [0, 1, 2])
        self.decoder = HSIDecoderSimple(128)

    def forward(self, hsi_sr, hsi_rgb_sr, ref_hr):
        ref_feats_warp = self.rgb_encoder(ref_hr)
        hsi_feats, reverse = self.hsi_encoder(hsi_sr)
        out = self.decoder(hsi_feats, ref_feats_warp, reverse)
        return out

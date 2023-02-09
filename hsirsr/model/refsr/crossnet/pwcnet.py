import torch
import torch.nn as nn
import torch.nn.functional as F

from . import correlation


def backwarp(input, flow):
    B, _, H, W = input.shape

    hor = torch.linspace(-1.0 + (1.0 / W), 1.0 - (1.0 / W), W)
    hor = hor.view(1, 1, 1, -1).expand(-1, -1, H, -1)
    ver = torch.linspace(-1.0 + (1.0 / H), 1.0 - (1.0 / H), H)
    ver = ver.view(1, 1, -1, 1).expand(-1, -1, -1, W)
    grid = torch.cat([hor, ver], 1)
    
    if input.is_cuda:
        grid = grid.cuda()

    flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)

    vgrid = grid + flow
    vgrid = vgrid.permute(0, 2, 3, 1)

    input = torch.cat([input, flow.new_ones([B, 1, H, W])], 1)

    output = F.grid_sample(input=input, grid=vgrid, mode='bilinear', padding_mode='border', align_corners=False)

    mask = output[:, -1:, :, :]
    mask[mask > 0.999] = 1.0
    mask[mask < 1.0] = 0.0

    return output[:, :-1, :, :] * mask


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.netOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netTwo = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netThr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netFou = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netFiv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netSix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, tenInput):
        tenOne = self.netOne(tenInput)
        tenTwo = self.netTwo(tenOne)
        tenThr = self.netThr(tenTwo)
        tenFou = self.netFou(tenThr)
        tenFiv = self.netFiv(tenFou)
        tenSix = self.netSix(tenFiv)

        return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]


class Decoder(nn.Module):
    def __init__(self, intLevel):
        super().__init__()

        intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
        intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]

        if intLevel < 6:
            self.netUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6:
            self.netUpfeat = nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6:
            self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

        self.netOne = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netTwo = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netThr = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netFou = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netFiv = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.netSix = nn.Sequential(
            nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, tenOne, tenTwo, objPrevious):
        tenFlow = None
        tenFeat = None

        if objPrevious is None:
            tenFlow = None
            tenFeat = None

            tenVolume = F.leaky_relu(correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1)

            tenFeat = torch.cat([tenVolume], 1)

        elif objPrevious is not None:
            tenFlow = self.netUpflow(objPrevious['tenFlow'])
            tenFeat = self.netUpfeat(objPrevious['tenFeat'])

            tenVolume = F.leaky_relu(correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(
                tenTwo, tenFlow * self.fltBackwarp)), negative_slope=0.1)

            tenFeat = torch.cat([tenVolume, tenOne, tenFlow, tenFeat], 1)

        tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
        tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

        tenFlow = self.netSix(tenFeat)

        return {
            'tenFlow': tenFlow,
            'tenFeat': tenFeat
        }


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()

        self.netMain = nn.Sequential(
            nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )

    def forward(self, input):
        return self.netMain(input)


class PWCNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

    def forward2(self, img1, img2):
        img1 = self.netExtractor(img1)
        img2 = self.netExtractor(img2)

        objEstimate = self.netSix(img1[-1], img2[-1], None)
        objEstimate = self.netFiv(img1[-2], img2[-2], objEstimate)
        objEstimate = self.netFou(img1[-3], img2[-3], objEstimate)
        objEstimate = self.netThr(img1[-4], img2[-4], objEstimate)
        objEstimate = self.netTwo(img1[-5], img2[-5], objEstimate)

        return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
    
    def forward(self, img1, img2):
        H, W = img1.shape[2], img1.shape[3]
        img1 = self.netExtractor(img1)
        img2 = self.netExtractor(img2)

        flows = []
        
        objEstimate = self.netSix(img1[-1], img2[-1], None)
        objEstimate = self.netFiv(img1[-2], img2[-2], objEstimate)
        flow = objEstimate['tenFlow'] * 2.5
        flows.append(flow)
        objEstimate = self.netFou(img1[-3], img2[-3], objEstimate)
        flow = objEstimate['tenFlow'] * 5
        flows.append(flow)
        objEstimate = self.netThr(img1[-4], img2[-4], objEstimate)
        flow = objEstimate['tenFlow'] * 10
        flows.append(flow)
        objEstimate = self.netTwo(img1[-5], img2[-5], objEstimate)

        flow = (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
        flows.append(flow)

        
        output_dic = {
                    'flow_12_1':F.interpolate(flows[-1], size=(H, W), mode='bilinear', align_corners=False),
                    'flow_12_2':F.interpolate(flows[-1], size=(H//2, W//2), mode='bilinear', align_corners=False),
                    'flow_12_3':F.interpolate(flows[-1], size=(H//4, W//4), mode='bilinear', align_corners=False),
                    'flow_12_4':F.interpolate(flows[-1], size=(H//8, W//8), mode='bilinear', align_corners=False)
                 }

        return output_dic
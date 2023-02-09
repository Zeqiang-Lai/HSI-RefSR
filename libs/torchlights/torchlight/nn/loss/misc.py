import torch
import torch.nn as nn
import torch.functional as F
import torch.fft as fft

from .functional import charbonnier_loss


class SAMLoss(torch.nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, x1, x2, eps=1e-6):
        out = 0
        for i in range(x1.shape[0]):
            X = x1[i].squeeze()
            Y = x2[i].squeeze()
            a = torch.sum(X*Y, axis=0)
            b = torch.sqrt(torch.sum(X**2, axis=0))
            c = torch.sqrt(torch.sum(Y**2, axis=0))
            tmp = (a + eps) / (b + eps) / (c + eps)
            out += torch.mean(torch.arccos(tmp))
        return out / x1.shape[0]


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, reduce='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduce = reduce

    def forward(self, x, y):
        return charbonnier_loss(x, y, self.reduce, self.eps)


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)    # filter
        down = filtered[:, :, ::2, ::2]               # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4                  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class FocalFrequencyLoss(nn.Module):
    """ Paper: Focal Frequency Loss for Image Reconstruction and Synthesis 
        Expect input'shape to be [..., W, H]
    """

    def __init__(self, alpha=1, norm='ortho'):
        super().__init__()
        self.alpha = alpha
        self.norm = norm

    def forward(self, output, target):
        o = fft.fftn(output, dim=(-1, -2), norm=self.norm)
        t = fft.fftn(target, dim=(-1, -2), norm=self.norm)
        d = torch.norm(torch.view_as_real(o - t), p=2, dim=-1)
        w = d.pow(self.alpha)
        w = (w - torch.amin(w, dim=(-1, -2), keepdim=True)) / torch.amax(w, dim=(-1, -2), keepdim=True)
        return torch.mean(w * d.pow(2))


class FFTLoss(nn.Module):
    def __init__(self, rate=0.0):
        super().__init__()
        self.rate = rate

    def forward(self, predict, target):
        assert predict.shape == target.shape
        p_fft = fft.fftn(predict, dim=(-1, -2))
        t_fft = fft.fftn(target, dim=(-1, -2))
        p_fft = fft.fftshift(p_fft, dim=(-1, -2))
        t_fft = fft.fftshift(t_fft, dim=(-1, -2))
        p_fft = p_fft * self.mask(p_fft, self.rate)
        t_fft = t_fft * self.mask(t_fft, self.rate)
        return torch.mean(torch.pow(torch.abs(p_fft-t_fft), 2))

    @staticmethod
    def mask(img, rate):
        mask = torch.ones_like(img)
        rows, cols = img.shape[-2], img.shape[-1]
        mask[:, :, :, int(rows/2-rows*rate):int(rows/2+rows*rate), int(cols/2-cols*rate):int(cols/2+cols*rate)] = 0
        return mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchlight
from torchlight.metrics import mpsnr, mssim, sam
from torchlight.utils.helper import get_obj, to_device

from ..model import refsr as model_zoo
from .util import cal_metric


class BaseModule(torchlight.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.device = torch.device('cuda')
        self.model = get_obj(model, model_zoo).to(self.device)
        self.optimizer = get_obj(optimizer, optim, self.model.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.clip_max_norm = 1e6

    def step(self, data, train, epoch, step):
        if train:
            return self._train(data)
        return self._eval(data)

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])


class CommonModule(BaseModule):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def _train(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = data
        target = hsi_hr
        output, _, _, _ = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=rgb_hr)

        loss = self.criterion(output, target)

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
        self.optimizer.step()

        metrics = {'loss': loss.item(),
                   'psnr': cal_metric(mpsnr, output, target),
                   }

        return self.StepResult(metrics=metrics)

    def _eval(self, data):
        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = data
        target = hsi_hr
        output, warped_rgb, flow, masks = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=rgb_hr)

        metrics = {'psnr': cal_metric(mpsnr, output, target),
                   'ssim': cal_metric(mssim, output, target),
                   'sam': cal_metric(sam, output, target),
                   }
        
        return self.StepResult(metrics=metrics)


class SimpleModule(BaseModule):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)

    def _train(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = data
        target = hsi_hr
        output = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=rgb_hr)

        loss = self.criterion(output, target)

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
        self.optimizer.step()

        metrics = {'loss': loss.item(), 'psnr': cal_metric(mpsnr, output, target)}
        return self.StepResult(metrics=metrics)

    def _eval(self, data):
        data = to_device(data, self.device)
        hsi_hr, hsi_lr, hsi_rgb_hr, hsi_rgb_lr, rgb_hr, rgb_lr = data
        target = hsi_hr
        output = self.model(hsi_sr=hsi_lr, hsi_rgb_sr=hsi_rgb_lr, ref_hr=hsi_rgb_hr)

        metrics = {'psnr': cal_metric(mpsnr, output, target),
                   'ssim': cal_metric(mssim, output, target),
                   'sam': cal_metric(sam, output, target)}
        
        return self.StepResult(metrics=metrics)

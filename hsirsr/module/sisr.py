import torch
import torch.nn as nn
import torchlight
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchlight.metrics import mpsnr, mssim, sam
from torchlight.utils.helper import get_obj

from ..model import sisr as model_zoo


class BaseModule(torchlight.Module):
    def __init__(self, model_cfg, lr, weight_decay=0):
        self.device = torch.device('cuda:0')
        self.model = get_obj(model_cfg, model_zoo).to(self.device)
        self.model_type = model_cfg['type@']
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[45], gamma=0.2)
        self.criterion = nn.MSELoss()

        self.clip_max_norm = 1e6

    def forward(self, lr, sr):
        if self.model_type in ['sspsr']:
            return self.model(lr, sr)
        elif self.model_type in ['mcnet']:
            return self.model(lr)
        else:
            return self.model(sr)

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        lr, sr, hr = data
        lr, sr, hr = lr.to(self.device), sr.to(self.device), hr.to(self.device)

        out = self.forward(lr, sr)
        loss = self.criterion(out, hr)

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
        self.optimizer.step()

        metrics = {'loss': loss.item(), 'psnr': cal_metric(mpsnr, out, hr)}
        return self.StepResult(metrics=metrics)

    def eval(self, data):
        lr, sr, hr, file_name = data
        lr, sr, hr = lr.to(self.device), sr.to(self.device), hr.to(self.device)

        out = self.forward(lr, sr)
        metrics = {'psnr': cal_metric(mpsnr, out, hr),
                   'ssim': cal_metric(mssim, out, hr),
                   'sam': cal_metric(sam, out, hr),
        }
        return self.StepResult(metrics=metrics)

    # ---------------------------------------------------------------------------- #
    #                something you don't need to care in most cases                #
    # ---------------------------------------------------------------------------- #

    def step(self, data, train, epoch, step):
        if train:
            return self.train(data)
        return self.eval(data)

    def on_epoch_end(self, train):
        if train:
            self.scheduler.step()

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])

from abc import ABC, abstractmethod
from typing import NamedTuple


class Module(ABC):
    class StepResult(NamedTuple):
        imgs: dict = {}
        metrics: dict = {}

    @abstractmethod
    def step(self, data, train, epoch, step) -> StepResult:
        """ return a StepResult that contains the imgs and metrics you want to save and log """
        raise NotImplementedError

    def on_epoch_start(self, train):
        pass

    def on_epoch_end(self, train):
        pass

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state):
        raise NotImplementedError


class SMSOModule(Module):
    """ Single Model Single Optimizer Module"""

    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def step(self, data, train, epoch, step):
        if train:
            self.model.train()
            self.optimizer.zero_grad()

        loss, result = self._step(data, train, epoch, step)

        if train:
            loss.backward()
            self.optimizer.step()

        return result

    def _step(self, data, train, epoch, step):
        """ return (loss, self.StepResult) 
        """
        raise NotImplementedError

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])


class SimpleModule(SMSOModule):
    def __init__(self, model, optimizer, device):
        super().__init__(model, optimizer)
        self.device = device
        self.model = self.model.to(device)

    def _step(self, data, train, epoch, step):
        input, target = data
        input, target = input.to(self.device), target.to(self.device)
        output = self.model(input)
        loss = self.criterion(output, target)

        metrics = {'loss': loss.item()}
        return loss, self.StepResult(metrics=metrics)

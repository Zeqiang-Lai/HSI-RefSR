# Torchlight

Torchlight is an ultra light-weight pytorch wrapper for fast prototyping of computer vision models.

[![asciicast](https://asciinema.org/a/441271.svg)](https://asciinema.org/a/441271)

## Installation

- Install via [PyPI](https://pypi.org/project/torchlights/).

```shell
pip install torchlights
```

- Install the latest version from source.

```shell
git clone https://github.com/Zeqiang-Lai/torchlight.git
cd torchlight
pip install .
pip install -e . # editable installation
```

## Features

- Most modules are self-contained.
- Debug Mode.
- User friendly progress bar .
- Save latest checkpoint if interrupted by Ctrl-C.
- Override any option in configuration file with cmd args.

## A Minimal Example

You can also find the following example at [example/mnist/minimal.py](examples/mnist/minimal.py)

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import datasets, transforms
import torchlight
from torchlight.utils.metrics import accuracy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Moudle(torchlight.SMSOModule):
    def __init__(self, lr, device):
        model = Net()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        super().__init__(model, optimizer)
        self.device = device
        self.model.to(device)
        self.criterion = F.nll_loss
        self.metrics = [accuracy]
    
    def _step(self, data, train, epoch, step):
        input, target = data
        input, target = input.to(self.device), target.to(self.device)
        output = self.model(input)
        loss = self.criterion(output, target)
        
        metrics = {'loss': loss.item(), 'accuracy': accuracy(output, target)}
        imgs = {'input': input}

        return loss, self.StepResult(metrics=metrics, imgs=imgs)
    
if __name__ == '__main__':
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='MNIST', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])), batch_size=64, shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='MNIST', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=True, num_workers=4)

    device = torch.device('cuda')
    module = Moudle(lr=0.01, device=device)
    engine = torchlight.Engine(module, save_dir='experiments/simple_l1')
    engine.train(train_loader, valid_loader=test_loader)
```

## Useful Tools

- [kornia](https://github.com/kornia/kornia): Open Source Differentiable Computer Vision Library.
- [accelerate](https://github.com/huggingface/accelerate/): A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.
- [einops](https://github.com/arogozhnikov/einops): Flexible and powerful tensor operations for readable and reliable code.
- [image-similarity-measures](https://github.com/up42/image-similarity-measures): Implementation of eight evaluation metrics to access the similarity between two images. The eight metrics are as follows: RMSE, PSNR, SSIM, ISSM, FSIM, SRE, SAM, and UIQ.
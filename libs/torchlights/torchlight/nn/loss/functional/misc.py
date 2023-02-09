import torch

def charbonnier_loss(x, y, reduce='mean', eps=1e-3):
    diff = x - y
    loss = torch.sqrt((diff * diff) + (eps*eps))
    if reduce == 'mean':
        loss = torch.mean(loss)
    elif reduce == 'batch':
        loss = torch.sum(loss) / loss.shape[0]
    else:
        raise ValueError('Invalid reduce mode, choose from [mean, batch]')
    return loss


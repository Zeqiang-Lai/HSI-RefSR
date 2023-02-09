
import torch


def interp1d(xp, fp, x):
    slopes = (fp[1:]-fp[:-1])/(xp[1:]-xp[:-1])
    locs = torch.searchsorted(xp, x)
    locs = locs.clip(1, len(xp) - 1) - 1
    return slopes[locs] * (x - xp[locs]) + xp[locs]


if __name__ == '__main__':
    xp = torch.arange(0, 1, 0.1)  # X values
    fp = torch.arange(0, 10, 1.0)  # Y values
    x = torch.Tensor([[0.1, 0.05, 0.3], [1.5, -2, 0.5]])  # X values you need to interpolate

    interp1d(xp, fp, x)
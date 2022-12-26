import functools
import numpy as np

FORMAT_HWC = 'HWC'
FORMAT_CHW = 'CHW'
DATA_FORMAT = FORMAT_HWC


def set_data_format(format):
    if format != FORMAT_HWC and format != FORMAT_CHW:
        raise ValueError('Invalid data format, choose from '
                         'torchlight.metrics.HWC or torchlight.metrics.CHW')
    global DATA_FORMAT
    DATA_FORMAT = format


def CHW2HWC(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        if DATA_FORMAT == FORMAT_CHW:
            output = output.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)
        return func(output, target, *args, **kwargs)
    return warpped


def torch2numpy(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        if not isinstance(output, np.ndarray):
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        return func(output, target, *args, **kwargs)
    return warpped


def bandwise(func):
    @functools.wraps(func)
    def warpped(output, target, *args, **kwargs):
        if DATA_FORMAT == FORMAT_CHW:
            C = output.shape[-3]
            total = 0
            for ch in range(C):
                x = output[ch, :, :]
                y = target[ch, :, :]
                total += func(x, y, *args, **kwargs)
            return total / C
        else:
            C = output.shape[-1]
            total = 0
            for ch in range(C):
                x = output[:, :, ch]
                y = target[:, :, ch]
                total += func(x, y, *args, **kwargs)
            return total / C
    return warpped


def enable_batch_input(reduce=True):
    def inner(func):
        @functools.wraps(func)
        def warpped(output, target, *args, **kwargs):
            if len(output.shape) == 4:
                b = output.shape[0]
                out = [func(output[i], target[i]) for i in range(b)]
                if reduce:
                    return sum(out) / len(out)
                return out
            return func(output, target, *args, **kwargs)
        return warpped
    return inner

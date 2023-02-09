import time
import shutil
import os

from numpy import inf


def action_confirm(msg):
    import readchar
    res = ''
    options = ['y', 'n', 'Y', 'N']
    print(msg + " Press y/n")
    while res not in options:
        res = readchar.readchar()
    return res == 'y' or res == 'Y'

def auto_rename(path):
    count = 1
    new_path = path
    while True:
        if not os.path.exists(new_path):
            return new_path
        file_name = os.path.basename(path)
        name, ext = file_name.split('.')
        new_file_name = '{}_{}.{}'.format(name, count, ext)
        new_path = os.path.join(os.path.dirname(path), new_file_name)
        count += 1


def format_num(n, fmt='{0:.3g}'):
    f = fmt.format(n).replace('+0', '+').replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n


def text_divider(sym, text='', ncols=None):
    if ncols is None:
        ncols = shutil.get_terminal_size()[0]
    left = ncols // 2
    right = ncols - left
    divider = sym*(left-len(text)) + text + sym*right
    return divider


class CheckpointCleaner:
    def __init__(self, root, keep='all'):
        """ This class provide the ability to control the storage size of checkpoints. 

            Args:
                root: root dir path of checkpoints
                keep: the mode to control storage. Defaults to 'all'. The valid choice includes
                      `all`: keep all checkpoints, `latest_#`: keep latest n checkpoints
        """
        self.root = root
        self.mode = keep
        self._check_mode()
    
    def _check_mode(self):
        if self.mode.startswith('latest'):
            n = int(self.mode.split('_')[-1])
        elif self.mode == 'all':
            return 
        else:
            raise ValueError(f'Invalid mode {self.mode}')
            
            
    def clean(self):
        file_names = os.listdir(self.root)
        file_names = filter(lambda x: x.endswith('.pth'), file_names)
        file_names = filter(lambda x: x.startswith('model-epoch'), file_names)
        file_names = [os.path.join(self.root, n) for n in file_names]
        file_names = sorted(file_names, key=lambda x: os.path.getctime(x), reverse=True)

        if self.mode == 'all':
            return
        elif self.mode.startswith('latest'):
            n = int(self.mode.split('_')[-1])
            for path in file_names[n:]:
                os.remove(path)


class Timer:
    def __init__(self):
        self._start_time = 0

    def tic(self):
        self._start_time = time.time()

    def tok(self):
        now = time.time()
        used = int(now - self._start_time)
        second = used % 60
        used = used // 60
        minutes = used % 60
        used = used // 60
        hours = used
        self._start_time = time.time()
        return "{}:{}:{}".format(hours, minutes, second)


class MetricTracker:
    def __init__(self):
        self._data = {}
        self.reset()

    def reset(self):
        self._data = {}

    def update(self, key, value, n=1):
        if key not in self._data.keys():
            self._data[key] = {'total': 0, 'count': 0}
        self._data[key]['total'] += value * n
        self._data[key]['count'] += n

    def avg(self, key):
        return self._data[key]['total'] / self._data[key]['count']

    def result(self):
        return {k: self._data[k]['total'] / self._data[k]['count'] for k in self._data.keys()}

    def summary(self):
        items = ['{}: {:.8f}'.format(k, v) for k, v in self.result().items()]
        return ' '.join(items)


class PerformanceMonitor:
    def __init__(self, mnt_mode, early_stop_threshold=0.1):
        self.mnt_mode = mnt_mode
        self.early_stop_threshold = early_stop_threshold

        assert self.early_stop_threshold > 0, 'early_stop_threshold should be greater than 0'
        assert self.mnt_mode in ['min', 'max']

        self.reset()

    def update(self, metric, info=None):
        improved = (self.mnt_mode == 'min' and self.mnt_best - metric >= self.early_stop_threshold) or \
                   (self.mnt_mode == 'max' and metric - self.mnt_best >= self.early_stop_threshold)
        self.best = False
        if improved:
            self.mnt_best = metric
            self.not_improved_count = 0
            self.best = True
            self.mnt_best_info = info
        else:
            self.not_improved_count += 1

    def is_best(self):
        return self.best == True

    def should_early_stop(self, not_improved_count=5):
        return self.not_improved_count >= not_improved_count

    def reset(self):
        self.not_improved_count = 0
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.best = False
        self.mnt_best_info = None

    def state_dict(self):
        return {'not_improved_count': self.not_improved_count, 'mnt_best': self.mnt_best, 
                'best': self.best, 'mnt_best_info': self.mnt_best_info}

    def load_state_dict(self, states):
        self.not_improved_count = states['not_improved_count']
        self.mnt_best = states['mnt_best']
        self.best = states['best']
        self.mnt_best_info = states.get('mnt_best_info')
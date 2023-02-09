from typing import NamedTuple
from pathlib import Path
import os


from tqdm import tqdm
from qqdm import qqdm
from colorama import Fore, init

from ..logging.logger import Logger
from .module import Module
from ._util import Timer, MetricTracker, PerformanceMonitor, CheckpointCleaner, text_divider, format_num, action_confirm

init(autoreset=True)

timer = Timer()


class Experiment:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.log_dir = self.save_dir / 'log'
        self.ckpt_dir = self.save_dir / 'ckpt'

    def create(self):
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        return self


class EngineConfig(NamedTuple):
    max_epochs: int = 10
    log_step: int = 20
    log_img_step: int = 20
    valid_log_img_step: int = 1

    save_per_epoch: int = 1
    valid_per_epoch: int = 1

    mnt_mode: str = 'min'
    mnt_metric: str = 'loss'
    enable_early_stop: bool = False
    early_stop_threshold: float = 0.01
    early_stop_count: int = 5

    pbar: str = 'tqdm'
    num_fmt: str = '{:8.5g}'
    ckpt_save_mode: str = 'all'
    enable_tensorboard: bool = False


class Engine:
    def __init__(self, module: Module, save_dir, **kwargs):
        self.module = module
        self.experiment = Experiment(save_dir).create()
        self.cfg = EngineConfig(**kwargs)
        self.logger = Logger(self.experiment.log_dir, self.cfg.enable_tensorboard)
        self.monitor = PerformanceMonitor(self.cfg.mnt_mode, self.cfg.early_stop_threshold)
        self.ckpt_cleaner = CheckpointCleaner(self.experiment.ckpt_dir, keep=self.cfg.ckpt_save_mode)
        self.start_epoch = 1
        self.debug_mode = False
        self.ckpt_cleaner.clean()
        self.module.engine=self

    def set_debug_mode(self):
        self.debug_mode = True

    def resume(self, model_name, base_dir=None):
        ckpt_dir = self.experiment.ckpt_dir if base_dir is None else Experiment(base_dir).ckpt_dir
        resume_path = ckpt_dir / 'model-{}.pth'.format(model_name)
        assert resume_path.exists(), '{} not found'.format(resume_path)
        self._resume_checkpoint(resume_path)
        return self

    def train(self, train_loader, valid_loader=None):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):
            self._show_divider('=', text='Epoch[{}]'.format(epoch))
            try:
                self._train(train_loader, epoch, valid_loader)
            except KeyboardInterrupt:
                msg = Fore.RED + "Oops!" + Fore.RESET + " Training was terminated by Ctrl-C. "
                "Do you want to save the latest checkpoint?"
                if action_confirm(msg):
                    save_path = self._save_checkpoint(epoch, f'interrupt_{epoch}')
                    self.logger.warning("Saving interrupted checkpoint: {} ...".format(save_path))
                return

            if self.cfg.enable_early_stop and self.monitor.should_early_stop(self.cfg.early_stop_count):
                self.logger.warning("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.monitor.not_improved_count))
                break

    # TODO: use this function to support mannuly control training dataloader outside of engine
    def _train(self, train_loader, epoch, valid_loader=None):
        timer.tic()
        result = self._train_epoch(epoch, train_loader)
        self.epoch = epoch

        # ------------------ Save logged informations into log dict ------------------ #
        log = {'epoch': epoch, 'time': timer.tok()}
        log.update(result)

        # ------------------ Print logged informations to the screen ----------------- #
        if valid_loader is not None and (epoch % self.cfg.valid_per_epoch == 0 or self.debug_mode):
            val_log = self._valid_epoch(epoch, valid_loader)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.debug_mode:
            return

        self._log_log(log)

        # --------- Evaluate model performance according to configured metric -------- #
        assert self.cfg.mnt_metric in log.keys(), '%s not in log keys' % self.cfg.mnt_metric
        self.monitor.update(log[self.cfg.mnt_metric], info={'epoch': epoch})
        self.logger.info('PeformenceMonitor: ' + str(self.monitor.state_dict()))

        # -------------------------- Save checkpoint -------------------------- #
        save_path = self._save_checkpoint(epoch, postfix='latest')
        self.logger.info("Saving latest checkpoint: {} ...".format(save_path))

        if epoch % self.cfg.save_per_epoch == 0:
            save_path = self._save_checkpoint(epoch)
            self.logger.info("Saving checkpoint: {} ...".format(save_path))

        if self.monitor.is_best():
            save_path = self._save_checkpoint(epoch, postfix='best')
            self.logger.warning("Saving current best: {} ...".format(save_path))

    def test(self, test_loader, name=None):
        old_logger = self.logger
        self.test_log_dir = self.experiment.save_dir / 'test' / 'epoch{}'.format(self.epoch-1)
        if name is not None:
            self.test_log_dir = self.test_log_dir / name
        self.test_log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.test_log_dir)

        val_log = self._valid_epoch(1, test_loader)
        log = {'test_'+k: v for k, v in val_log.items()}

        self._log_log(log)

        self.logger = old_logger

    def _log_log(self, log):
        self._show_divider('-')
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
        self._show_divider('-')

    def _train_epoch(self, epoch, train_loader):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and met
        ric in this epoch.
        """
        
        metric_tracker = MetricTracker()

        len_epoch = len(train_loader)
        pbar = self._progress_bar(total=len_epoch)
        pbar.set_description('Train [{}|{}]'.format(epoch, self.cfg.max_epochs))
        self.module.on_epoch_start(train=True)
        for batch_idx, data in enumerate(train_loader):
            gstep = (epoch - 1) * len_epoch + batch_idx + 1
            results = self.module.step(data, train=True, epoch=epoch, step=gstep)

            self.logger.tensorboard.set_step(gstep, mode='train')
            for name, value in results.metrics.items():
                metric_tracker.update(name, value)
                self.logger.tensorboard.add_scalar(name, value)

            if gstep % self.cfg.log_step == 0:
                self.logger.debug('Train Epoch: {} {} {}'.format(epoch,
                                                                 _progress(batch_idx, train_loader),
                                                                 metric_tracker.summary()))
            if gstep % self.cfg.log_img_step == 0:
                for name, img in results.imgs.items():
                    img_name = os.path.join('train', name, '{}_{}.png'.format(epoch, gstep))
                    self.logger.save_img(img_name, img)

            pbar.set_postfix(self._format_nums(metric_tracker.result()))
            pbar.update()

            if self.debug_mode:
                break

        pbar.close()
        self.module.on_epoch_end(train=True)

        return metric_tracker.result()

    def _valid_epoch(self, epoch, valid_loader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        import torch
        
        metric_tracker = MetricTracker()

        len_epoch = len(valid_loader)
        pbar = self._progress_bar(total=len_epoch)
        pbar.set_description('Valid [{}|{}]'.format(epoch, self.cfg.max_epochs))
        self.module.on_epoch_start(train=False)
        with torch.no_grad():
            len_epoch = len(valid_loader)
            for batch_idx, data in enumerate(valid_loader):
                gstep = (epoch - 1) * len_epoch + batch_idx + 1
                results = self.module.step(data, train=False, epoch=epoch, step=gstep)

                self.logger.debug(f'{batch_idx}: {results.metrics}')

                self.logger.tensorboard.set_step(gstep, mode='valid')
                for name, value in results.metrics.items():
                    metric_tracker.update(name, value)
                    self.logger.tensorboard.add_scalar(name, value)

                if gstep % self.cfg.valid_log_img_step == 0:
                    for name, img in results.imgs.items():
                        img_name = os.path.join('valid', name, '{}_{}.png'.format(epoch, gstep))
                        self.logger.save_img(img_name, img)

                pbar.set_postfix(self._format_nums(metric_tracker.result()))
                pbar.update()

                if self.debug_mode:
                    break
            pbar.close()
        self.module.on_epoch_end(train=False)
        return metric_tracker.result()

    def _save_checkpoint(self, epoch, postfix=None):
        """
        Saving checkpoints and return saved path

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model-best.pth'
        """
        import torch
        
        if postfix is None:
            postfix = f'epoch{epoch}'
        state = {
            'epoch': epoch,
            'module': self.module.state_dict(),
            'monitor': self.monitor.state_dict() if self.cfg.mnt_mode != 'off' else None,
            'config': self.cfg
        }
        filename = str(self.experiment.ckpt_dir / f'model-{postfix}.pth')
        torch.save(state, filename)
        self.ckpt_cleaner.clean()
        return filename

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        import torch
        
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.epoch = self.start_epoch

        if 'monitor' in checkpoint:
            self.monitor.load_state_dict(checkpoint['monitor'])
        else:
            self.logger.info("Monitor state dict not found, default value will be used.")

        self.module.load_state_dict(checkpoint['module'])

        self.logger.info("Checkpoint loaded. Resume from epoch {}".format(self.start_epoch-1))

    def _progress_bar(self, total):
        if self.cfg.pbar == 'qqdm':
            bar = qqdm
        elif self.cfg.pbar == 'tqdm':
            bar = tqdm
        else:
            raise ValueError('choice of progressbar [qqdm, tqdm]')
        return bar(total=total, dynamic_ncols=True)

    def _show_divider(self, sym, text='', ncols=None):
        print(text_divider(sym, text, ncols))

    def _format_nums(self, d):
        return {k: format_num(v, fmt=self.cfg.num_fmt) for k, v in d.items()}


def _progress(batch_idx, loader):
    current = batch_idx * loader.batch_size
    total = len(loader) * loader.batch_size
    return '[{}/{} ({:.0f}%)]'.format(current, total, 100.0 * current / total)

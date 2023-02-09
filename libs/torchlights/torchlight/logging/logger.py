import logging
import logging.config
import os
from pathlib import Path
import os
import yaml
import importlib 
from datetime import datetime


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_logging(save_dir, log_config=os.path.join(CURRENT_DIR, 'logger_config.yaml'), default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    if log_config.is_file():
        with log_config.open('rt') as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


log_levels = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


def get_logger(name, save_dir, verbosity=2):
    import colorlog
    setup_logging(save_dir)
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
        verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = colorlog.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger


class Logger:
    def __init__(self, log_dir, enable_tensorboard=False):
        self.log_dir = Path(log_dir)
        self.tensorboard_ = None
        self.enable_tensorboard = enable_tensorboard
        self.text = get_logger('Torchlight', self.log_dir)
        self.img_dir = self.log_dir / 'img'

    @property
    def tensorboard(self):
        if self.tensorboard_ is None:
            tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
            self.tensorboard_ = TensorboardWriter(tensorboard_dir, logger=self, 
                                                  enabled=self.enable_tensorboard)
        return self.tensorboard_

    def info(self, msg):
        self.text.info(msg)

    def debug(self, msg):
        self.text.debug(msg)

    def warning(self, msg):
        self.text.warning(msg)

    def save_img(self, name, img):
        # TODO: remove torchvision dependency
        from torchvision.utils import save_image
        save_path: Path = self.img_dir / name
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        save_image(img, save_path)


class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            # if enable is false, writer is None, then add_data would be none
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
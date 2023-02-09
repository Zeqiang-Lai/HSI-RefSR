from torchlight.trainer import config
from torchlight.trainer.entry import run_lazy, DataSource

from torchlight.utils.helper import get_obj
from torchlight.utils.reproducibility import SeedDataLoader, setup_randomness


import hsirsr.module as module
import hsirsr.data as dataset


class Source(DataSource):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_loader(self):
        train_dataset = get_obj(self.cfg.train.dataset, dataset)
        train_loader = SeedDataLoader(train_dataset, **self.cfg.train.loader)
        return train_loader

    def test_loader(self):
        test_dataset = get_obj(self.cfg.test.dataset, dataset)
        test_loader = SeedDataLoader(test_dataset, **self.cfg.test.loader)
        return test_loader


if __name__ == '__main__':
    args, cfg = config.basic_args()
    setup_randomness(2021, deterministic=False)
    module = get_obj(cfg.module, module)
    run_lazy(args, cfg, module, Source(cfg))

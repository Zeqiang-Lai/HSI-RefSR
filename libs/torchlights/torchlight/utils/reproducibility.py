from typing import Sequence, Optional
import random

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, Sampler
import numpy as np


# This module provide utils for controling reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html


def setup_randomness(seed=1234, deterministic=True):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)


class SeedDataLoader(DataLoader):
    def __init__(self,
                 dataset: Dataset,
                 seed: Optional[int] = 0,
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 sampler: Optional[Sampler[int]] = None,
                 batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0,
                 collate_fn = None,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0,
                 multiprocessing_context=None, *,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False):

        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=sampler,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=drop_last,
                         timeout=timeout,
                         worker_init_fn=seed_worker,  # use seed worker
                         multiprocessing_context=multiprocessing_context,
                         generator=seed_generator(seed),  # use seed generator
                         prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers)

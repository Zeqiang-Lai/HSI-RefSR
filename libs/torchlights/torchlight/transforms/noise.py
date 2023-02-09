import numpy as np
from skimage.util import random_noise

from ._util import LockedIterator


class GaussianNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        img_L = img + np.random.normal(0, self.sigma, img.shape)
        return img_L


class GaussianNoiseBlind(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""

    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
        self.pos = LockedIterator(self.__pos(len(sigmas)))

    def __call__(self, img):
        noise = np.random.randn(*img.shape) * self.sigmas[next(self.pos)]
        return img + noise


class GaussianNoiseBlindv2(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        noise = np.random.randn(
            *img.shape) * np.random.uniform(self.min_sigma, self.max_sigma) / 255
        return img + noise


class GaussianNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.

    def __call__(self, img):
        bwsigmas = np.reshape(self.sigmas[np.random.randint(
            0, len(self.sigmas), img.shape[0])], (-1, 1, 1))
        noise = np.random.randn(*img.shape) * bwsigmas
        return img + noise


class ImpulseNoise(object):
    """add impulse noise to the given numpy array (B,H,W)"""

    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(
            0, len(self.amounts), len(bands))]
        for i, amount in zip(bands, bwamounts):
            img[i, ...] = random_noise(
                img[i, ...], mode='s&p', amount=amount, salt_vs_pepper=self.s_vs_p)
        return img


class StripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_stripe = np.random.randint(
            np.floor(self.min_amount*W), np.floor(self.max_amount*W), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0, 1, size=(len(loc),))*0.5-0.25
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img


class DeadlineNoise(object):
    """add deadline noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(
            np.ceil(self.min_amount*W), np.ceil(self.max_amount*W), len(bands))
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[i, :, loc] = 0
        return img


class MixedNoise(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""

    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos:pos+num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img

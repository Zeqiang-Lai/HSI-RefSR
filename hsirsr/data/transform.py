from torchlight.transforms import GaussianDownsample, Upsample


class SRDegrade:
    def __init__(self, sf):
        self.sf = sf
        self.down = GaussianDownsample(sf, ksize=8, sigma=3)
        self.up = Upsample(sf, mode='cubic')

    def __call__(self, img):
        return self.up(self.down(img))

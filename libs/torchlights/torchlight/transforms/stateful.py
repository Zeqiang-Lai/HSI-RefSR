import random

class RandCrop:
    def __init__(self, img_size, crop_size):
        self.cropx = crop_size[0]
        self.cropy = crop_size[1]
        x, y = img_size[0], img_size[1]
        self.x1 = random.randint(0, x - self.cropx)
        self.y1 = random.randint(0, y - self.cropy)
        
    def __call__(self, img):
        return img[..., self.x1:self.x1+self.cropx, self.y1:self.y1+self.cropy]


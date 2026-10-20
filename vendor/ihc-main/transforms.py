import torch
import numpy as np
from torchvision import transforms


class Preprocessing(object):
    def __init__(self, mean, std, augmentor=None):
        self.augmentor = augmentor

        self.trans = transforms.Compose([transforms.ToTensor()])
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

    def __call__(self, imgs: dict):
        if self.augmentor is not None:
            imgs = self.augmentor(**imgs)
        imgs = list(imgs.values())
        labels = torch.tensor([], dtype=torch.long)
        for i in range(len(imgs)):
            if not len(imgs[i]):
                imgs[i] = torch.zeros(1, 0, 2, dtype=torch.int)
            else:
                imgs[i] = self.trans(np.asarray(imgs[i]))
            if i == 0:  # original image standardization
                for t, m, s in zip(imgs[0], self.mean, self.std):
                    t.sub_(m).div_(s)
            else:  # reshape to 2N-length vector for padding
                labels = torch.cat([labels, torch.full((imgs[i].shape[1],), i - 1)])
                imgs[i] = imgs[i].reshape(-1)
        return imgs[0], torch.cat(imgs[1:]), labels

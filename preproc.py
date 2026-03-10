import torch
import numpy as np
import torchvision.transforms as transforms

from tools.augmentations import (
    get_transforms, default_augmentations, npy_datasets,
)

# DARTS uses 'fashionmnist'; experimental_grow uses 'fashion-mnist'
_NAME_MAP = {
    "fashionmnist": "fashion-mnist",
}


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def data_transforms(dataset, cutout_length, no_augment=False):
    dataset = dataset.lower()
    ext_name = _NAME_MAP.get(dataset, dataset)

    augmentations = None if no_augment else default_augmentations.get(ext_name)
    base, aug = get_transforms(ext_name, augmentations)

    if dataset in npy_datasets:
        # npy: data is already float tensors after ToTensor; augment after
        train_transform = transforms.Compose(base + aug)
        valid_transform = transforms.Compose(base)
    else:
        # PIL datasets: augment on PIL images before ToTensor+Normalize
        train_transform = transforms.Compose(aug + base)
        valid_transform = transforms.Compose(base)

    if cutout_length > 0 and not no_augment:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform

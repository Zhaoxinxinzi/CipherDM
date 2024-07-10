# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#

import os

import jax
import jax.numpy as jnp

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from datasets.utils import get_transform


class MnistDataset:
    def __init__(self, is_train=True, target_labels=None):
        self.is_train = is_train
        self.target_labels = target_labels if target_labels is not None else list(range(10))
        self.imgs, self.labels = self._get_data()

    def _get_data(self):
        mnist_root = self._get_mnist_data_root()
        dataset = MNIST(root=mnist_root, train=self.is_train, download=True, transform=get_transform())
        imgs, labels = [], []
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            if label in self.target_labels:
                imgs.append(img)
                labels.append(label)
        return imgs, labels

    def _get_mnist_data_root(self):
        base = os.path.dirname(os.path.abspath(__file__))
        mnist_root = os.path.join(base, "mnist")
        os.makedirs(mnist_root, exist_ok=True)
        return mnist_root

    def __getitem__(self, index):
        image = self.imgs[index]
        label = self.labels[index]
        return jnp.array(image), jnp.array(label)

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataset = MnistDataset(is_train=False, target_labels=[0, 1])
    print(f"0, 1 samples: {len(dataset)}")


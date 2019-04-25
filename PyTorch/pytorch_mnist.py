from __future__ import print_function
import os
import torchvision
import torch
from torchvision import datasets, transforms


def load_mnist():
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],
             std=[0.5, 0.5, 0.5])])
    data_train = datasets.MNIST(
            root=os.getcwd() + "\\dataset\\",
            transform=transform,
            train=True,
            download=True)

    data_test = datasets.MNIST(
            root=os.getcwd() + "\\dataset\\",
            transform=transform,
            train=False)


def main():
    load_mnist()


if __name__ == '__main__':
    main()

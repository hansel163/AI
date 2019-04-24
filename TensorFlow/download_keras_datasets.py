

# Download all datasets of Keras
# Saved to C:\Users\<user>\.keras\datasets\
# That is %UserProfile%\.keras\datasets\
# files: 
#   [cifar-10-batches-py] [cifar-100-python] [fashion-mnist]
#   boston_housing.npz cifar-10-batches-py.tar.gz cifar-100-python.tar.gz      
#   imdb.npz   mnist.npz  reuters.npz
# .npz is zip of *.npy which is numpy saved array
import keras


def download_datasets():
    print("Checking boston housing ...")
    keras.datasets.boston_housing.load_data()
    print("Checking cifar10 ...")
    keras.datasets.cifar10.load_data()
    print("Checking cifar100 ...")
    keras.datasets.cifar100.load_data()
    print("Checking fashion_mnist ...")
    keras.datasets.fashion_mnist.load_data()
    print("Checking imdb ...")
    keras.datasets.imdb.load_data()
    print("Checking mnist ...")
    keras.datasets.mnist.load_data()
    print("Checking reuters ...")
    keras.datasets.reuters.load_data()


def main():
    download_datasets()


if __name__ == '__main__':
    main()
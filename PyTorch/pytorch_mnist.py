from __future__ import print_function
import os
import torchvision
import torch
from torchvision import datasets, transforms
import matplotlib as plt
from torch.autograd import Variable

MNIST_DATA_DIR = "\\dataset\\"
MODEL_FILE_NAME = "mnist_conv.pkl"


class MnistModel(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x


def load_mnist():
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],
             std=[0.5, 0.5, 0.5])])
    data_train = datasets.MNIST(
        root=os.getcwd() + MNIST_DATA_DIR,
        transform=transform,
        train=True,
        download=True)

    data_test = datasets.MNIST(
        root=os.getcwd() + MNIST_DATA_DIR, transform=transform, train=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset=data_train, batch_size=64, shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=data_test, batch_size=64, shuffle=True)

    return (data_loader_train, data_loader_test)


def show_mnist(data_loader_train):
    images, labels = next(iter(data_loader_train))
    img = torchvision.utils.make_grid(images)

    img = img.numpy().transpose(1, 2, 0)
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img*std + mean
    print([labels[i] for i in range(64)])
    plt.imshow(img)


def build_model():
    model = MnistModel()
    return model


def train_mode(model, data_loader_train, data_loader_test):
    n_epochs = 5
    model.load_state_dict(torch.load(MODEL_FILE_NAME))
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-"*10)
        for data in data_loader_train:
            X_train, y_train = data
            X_train, y_train = Variable(X_train), Variable(y_train)
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)

            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            running_correct += torch.sum(pred == y_train.data)
        testing_correct = 0
        for data in data_loader_test:
            X_test, y_test = data
            X_test, y_test = Variable(X_test), Variable(y_test)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)
        print(
            "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(
                running_loss/len(data_train),
                100*running_correct /
                len(data_train),
                100*testing_correct/len(data_test)))


def main():
    data_loader_train, data_loader_test = load_mnist()
    show_mnist(data_loader_train)
    model = build_model()
    torch.save(model.state_dict(), MODEL_FILE_NAME)

if __name__ == '__main__':
    main()

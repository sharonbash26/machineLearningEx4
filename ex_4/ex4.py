import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gcommand_loader import GCommandLoader
import soundfile
import copy
import os


# todo in the file data_loader_tesr delte the valdtion [:1000]

# newAgain the most recen file,the most new


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # we have 7 filter #size filter 3 0n 3 . !-balck white
        self.norm1 = nn.BatchNorm2d(10)
        self.pool = nn.MaxPool2d(2, 2)  # small the picture all ricu tae one value size 2  stride 2
        self.conv2 = nn.Conv2d(10, 15, 5)  #
        self.norm2 = nn.BatchNorm2d(15) #nirmul
        self.fc1 = nn.Linear(12210, 2048)  # linear fully contect lshath . input output .
        self.norm_fc1 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.1)  # histburt nuren nalam
        self.fc2 = nn.Linear(2048, 512)
        self.norm_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 100)  # 30 class
        self.norm_fc3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 30)  # 30 class

    def forward(self, x):
        x = self.norm1(F.relu(self.pool(self.conv1(x))))
        x = self.norm2(F.relu(self.pool(self.conv2(x))))
        x = x.view(x.size(0), -1)  # change to vector, minus 1 - if dont know one of mimdim  view - shithu
        x = self.dropout(F.relu(self.norm_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.norm_fc2((self.fc2(x)))))
        x = self.dropout(F.relu(self.norm_fc3((self.fc3(x)))))
        x = self.fc4(x)
        return x


model = Net()

device = torch.device("cuda" if "cuda" else "cpu")


def train(loader, criterion, optimizer):
    model.train()

    train_loss, train_acc = [0] * 2
    for input, labels in loader:
        optimizer.zero_grad() #maps gridant
        outputs = model(input) #vector 30 ,
        preds = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step() #update weight
        train_acc += torch.sum(preds == labels.data).item()

    train_loss /= len(loader)
    train_acc /= (len(loader) * 100) #100 is size of batch
    return train_loss, train_acc


def validation(loader, criterion):
    model.eval()

    val_loss, val_acc = [0] * 2
    for input, labels in loader:
        outputs = model(input)
        preds = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_acc += torch.sum(preds == labels.data).item()

    val_loss /= len(loader) #num of batch
    val_acc /= (len(loader) * 100)
    return val_loss, val_acc


def train_model(criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    for epoch in range(num_epochs):
        train_loss, train_acc = train(dataloaders['train'], criterion, optimizer)
        val_loss, val_acc = validation(dataloaders['val'], criterion)

        print('\nEpoch ' + str(epoch + 1))
        print('-' * 10)
        print('Train Loss: ' + str(train_loss) + '\tTrain Accuracy: ' + str(train_acc))
        print('Val Loss: ' + str(val_loss) + '\tVal Accuracy: ' + str(val_acc))


def test(test_loader, Net):
    model.eval()
    file = open('test_y', 'w')
    for x in test_loader:
        output = model(x)
        file.write(output)
    file.close()


def newTest(spects, all_predictions):
    # model.eval()
    with open("test_y", "w") as f:
        for spect, prediction in zip(spects, all_predictions):
            f.write("{}, {}".format(os.path.basename(spect[0]), str(prediction)))
            f.write(os.linesep)
    f.close()


def test_1():
    # model.eval()
    file = open('test_y', 'w')
    for x in range(10):
        # output=model(x)
        file.write(str(x) + '\n')
    file.close()


def main():
    torch.multiprocessing.freeze_support()
    device = torch.device("cpu")
    # model = net
    dataset = GCommandLoader('.data/train')
    train_set = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    validation = GCommandLoader('./valid')
    validation_set = torch.utils.data.DataLoader(
        validation, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    test_loader = GCommandLoader('./test')
    # testSet = torch.utils.data.DataLoader(
    #     test_loader, batch_size=100, shuffle=None,
    #     num_workers=20, pin_memory=True, sampler=None)

    # data, train_set, validation, testSet = dataset.to(device), train_set.to(device), testSet.to(device)
    dataLoader = {'train': train_set, 'val': validation_set}
    dataset_sizes = {'train': len(dataset.spects), 'val': len(validation.spects)}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_model(criterion, optimizer, dataLoader, dataset_sizes, 25)

    model.eval()  # Set model to evaluate mode
    outputs = []
    for x in test_loader:
        outputs.append(torch.argmax(model(x[0].reshape([1, x[0].shape[0], x[0].shape[1], x[0].shape[2]])), 1).item())
#x[0] photo  and size
    spects = test_loader.spects
    newTest(spects, outputs)


if __name__ == '__main__':
    main()

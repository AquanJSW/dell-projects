# -*- coding: utf-8 -*-

from gen_dataset import MyDataset
import torch
import torchvision
import torchvision.transforms as transfroms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mydataloader import mydataloader
import datetime

#   transform = transfroms.Compose([
#       #transfroms.Grayscale(),
#       transfroms.ToTensor(),
#       transfroms.Normalize((0.5,), (0.5,))
#   ])
#   trainset = MyDataset('/home/tjh/projects/py/mstar_classify/2S1.txt', transform=transform)
#   trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                             shuffle=True, num_workers=2)
#
#   classes = ('BMP2', 'BTR70', 'T72')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(512 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 50)
        self.fc5 = nn.Linear(50, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def main():
    start_time = datetime.datetime.now()
    trainloader, _ = mydataloader()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    net.to(device)

    for epoch in range(10):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (
                    epoch+1, i+1, running_loss/200))
                running_loss = 0.0


    print('Finished Training')
    stop_time = datetime.datetime.now()
    print('Running time: ', stop_time-start_time)

    PATH = '/home/tjh/projects/py/mstar_classify/mstar_trained_model.tar'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()

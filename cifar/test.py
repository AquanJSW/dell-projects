import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# A glimpse of one test batch
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # matplotlib.pyplot.imshow(img)
    # img：(H, W, C)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(testloader)
images, labels = dataiter.__next__()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[i]] for i in range(4)))

PATH = './cifar_trained_model.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net().to(device=device)
# torch.load读取torch.save的文件
net.load_state_dict(torch.load(PATH))
# 看看一个batch的预测情况
outputs = net(images.to(device=device))
# torch.max(tensor, dim) 取tensor的每一行（即遍历tensor的第一个维度：0维度）的对应dim维度的最大值
# 返回(values, indices)，values和indices都是一维tensor
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join("%5s" % classes[predicted[i].item()] for i in range(4)))

# 整个测试集的准确率
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device=device), data[1].to(device=device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum().item()

print("Accuracy of the network on the 10000 test images: %d %%" %
      (100 * correct / total))


# 看看每个类别的准确率
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device=device), data[1].to(device=device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (labels == predicted).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print("Accuracy of %5s : %2d %%" %
          (classes[i], class_correct[i] * 100 / class_total[i]))

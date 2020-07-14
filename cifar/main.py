import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([  # 将多个转换操作结合
    transforms.ToTensor(),  # 输出的数值变化范围[0, 1]
    # Normalize((R,G,B), (R,G,B)) 参数分别为 mean 和 std
    # 操作如下：channel = (channel-mean)/std
    # 就如下操作来说，[0,1] => [-1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        transform=transform)
# DataLoader返回一个iterator，例如本例，trainloader每次迭代出的元素是：
# [tensor(尺寸为4*3*32*32), tensor([8, 8, 5, 9])]
# 第一个tensor是batch; 第二个tensor是label，可见，分类型数据集可以直接用类别序号作为label
# DataLoader.__len__ == num_batch，本例中是dataset中图片总数的50000/4=12500
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
# 正好10类，下标对应label，0-9
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# show image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # matplotlib.pyplot.imshow(img)
    # img：(H, W, C)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
# classes作为一个tuple,也可以接受单元素的tensor作索引值。。。
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        # nn.MaxPool2d 本身作为一个Class，内部有forward函数：
        # def forward(self, input):
        #     return F.max_pool2d(input, self.kernel_size, self.stride,
        #                         self.padding, self.dilation, self.ceil_mode,
        #                         self.return_indices)
        # 所以，既可以先在__init__中，用torch.nn.MaxPool2d定义池化大小(此时forward函数已增加上述代码)，
        # 再在forward中传递图像；
        # 也可以直接在forward中调用完整的torch.nn.functional.max_pool2d()函数一次性传入图像和池化大小
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


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 使用GPU两步：网络导入device; 输入数据导入device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)

for epoch in range(1):

    running_loss = 0.0
    # mini-batch 注意用emumerate实现
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)  # 为什么CrossEntropyLoss这个Class也能传递参数? 可能是因为最终继承自nn.Module
        loss.backward()
        optimizer.step()

        # tensor.item()返回一般的python数，但仅适用于单元素tensor
        running_loss += loss.item()
        # 该if行体现了 mini-batch
        if i % 2000 == 1999:  # mini-batch: 2000
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))  # 每2000个batch算一次loss
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_trained_model.pth'
# net.stat_dict()返回net的所有参数，本质是OrderedDict，Dict索引如下：
# ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight',
#  'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
torch.save(net.state_dict(), PATH)


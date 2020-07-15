# -*- coding: utf-8 -*-
import torch
import numpy as np

# 生产未初始化的指定维数张量
x = torch.empty(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
z = torch.zeros(5, 3, dtype=torch.long)
print(z)
x0 = torch.tensor([4, 42], dtype=torch.float)
print(x0)
x1 = torch.tensor((43, 5), dtype=torch.long)
print(x1)
# tensor.new_ones(size)继承原Tensor的dtype和device属性
x2 = x1.new_ones((2, 2))
print(x2)
x3 = x0.new_ones((4, 3))
print(x3)
# tensor.full 填充指定数值，其他类似new_ones
x4 = x0.new_full((5, 4), fill_value=3.14)
print(x4)
# 这两个效果一样
x5 = torch.ones(2)
print(x5)
x6 = torch.ones((4,))
print(x6)

# torch.randn_like(tensor, dtype=) 继承自randn, 后者生成服从正太分布的随机数，
# 前者则是给输入张量填充正太分布随机数，即size不变
# n_like即normal_like
x7 = x0.new_zeros((3, 4))
print(x7)
x7 = torch.randn_like(x7, dtype=torch.double)
print(x7)
print(x7.size())
print(type(x7.size()))

# 下面几种操作效果相同
x8 = torch.rand(3, 4)
print(x8)
print(x7 + x8)
print(torch.add(x7, x8))
result = torch.empty(2, 4)
torch.add(x7, x8, out=result) # out后必须跟一个已知的张量，大小不必相同
print(result)
# 下面的方法x7数值已变；另外，所有改变tensor本身的操作都有下划线作为后缀，
# 可以与x7.add(x8, out=) 形成对比
print(x7.add_(x8))
print(x7)

# 兼容numpy的切片方法
print(x7[1:, :3])
print(x7[..., ::2])
print(x7[[2, 0, 1]])
print()

# tensor.view实现reshape而不改变原有的数据
#   非常类似np的reshape
x9 = torch.randn(4, 4)
y = x9.view(16)
print(y)
y0 = x9.view(-1, 2) # -1使得对应维度自动补全（16/2=8）
print(y0)

# tensor.item()适用于单元素张量，用于获取该值
y1 = torch.randn(1)
print(y1.item())

# tensor与numpy的转换，注意两个方向的转换都不会开辟新的内存空间
y2 = torch.ones(5)
y3 = y2.numpy()
print(y3)
y2.add_(1)
print(y3)
y5 = torch.from_numpy(y3)
print(y5)
y5.add_(1)
print(y3)

y4 = torch.linspace(0, 10, 5)
print(y4)

# cuda两种应用方式：
# 1 在定义张量时对device属性指定cuda
# 2 tensor.to()指定运行方式
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    y6 = torch.ones_like(x9, device=device)
    print(y6)
    x9 = x9.to(device)
    y7 = y6 + x9
    print(y7)

# 下面有关梯度和反向传播
# y8 = torch.arange(1, 5, dtype=torch.float, requires_grad=True).view(2, 2)
# requires_grad表示该节点是否需要跟踪梯度
y8 = torch.ones(2, 2, requires_grad=True, device=device)
print('y8\n', y8)
y9 = y8 + 2
print('y9\n', y9)
print('y9.grad_fn\n', y9.grad_fn)
z = y9 * y9 * 3
out = z.mean()
print('z, out\n', z, out)
# out是一个标量，backward不用参数就可反向传播，默认权重就是1
out.backward()
print('y8.grad\n', y8.grad)

z0 = torch.tensor([1, 2, 3], dtype=torch.float, requires_grad=True)
z1 = z0 * 2
# z1.data与z1相比，区别只有（前者保留了后者数据部分）吗？？？
print('z1.data\n', z1.data)
print('z1\n', z1)
while z1.data.norm() < 1000:
    z1 = z1 * 2
print('z1\n', z1)
# 对于最终的loss，若非标量，就需要对每个分量定义一个权重，相加之后变成标量loss，
# 才能求梯度，例如下面的：
# partial(z1[0]*1 + z1[1]*0.1 + z1[2]*0.001)/partial(z0_1) 就成为
# z0.grad的第一个分量
z1.backward(torch.tensor([1, 0.1, 0.001], dtype=torch.float))
print('z0.grad\n', z0.grad)
# z2 = z1.mean()
# z2.backward()
# print('z0.grad\n', z0.grad)

# with torch.no_grad()可以定义一个禁用梯度跟踪的代码块：
print('z0.requires_grad\n', z0.requires_grad)
print('(z0 ** 2).requires_grad\n', (z0 ** 2).requires_grad)
with torch.no_grad():
    print('(z0 ** 2).requires_grad\n', (z0 ** 2).requires_grad)
# tensor.detach()同上述作用：
z2 = z0.detach()
print('z2.requires_grad\n', z2.requires_grad)

# torch(tensor, other)或tensor.eq(other)返回的是boolean，other对tensor必须是可广播的，
# 只有相等才为 True
# tensor.eq_()为eq()的in-place形式
z3 = torch.tensor([1, 2, 3])
print('z3\n', z3)
z3.eq_(z3)
print('z3.eq_(z3)\n', z3)

# tensor.size()，返回一个tuple，为该张量的形状
# 同 tensor.shape
z4 = torch.ones([2, 3, 4, 5])
print('z4.size()\n', z4.size())
print('z4.shape\n', z4.shape)

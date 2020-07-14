# ~/projects/py_learn

from collections import ChainMap
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
import collections
from collections import deque
import os
import argparse

# namedtuple返回一个带属性名字的tuple类
Point = collections.namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)
# p是Point类的一个实例
print(type(p))
Point0 = collections.namedtuple('Points', ('x', 'y', 'z'))
#                                   ↑__真正的类名
p0 = Point0(1, 2, 3)
print(p0.x, p0.y, p0.z)
# p0的类名是Points，而非Point0
print(type(p0))
print(type(Point0))

# deque可接受sequence作为参数
q = deque(range(5))
print(q)
q0 = deque([1, 2, 3])
print(q0)
q1 = deque((1, 2, 'x'))
print(q1)
# 双边append, 同理pop(), popleft()
q1.append('y')
q1.appendleft('a')
print(q1)

# defaultdict在dict基础上增加了
# “引用不存在的key时不是抛出error，而是返回提前设定好的default value”
# 的特性, 另外其参数是一个无参函数，例如lambda

dd = defaultdict(lambda: 'NULL')
dd['key'] = 'value'
print(dd['key'])
print(dd['k'])

# OrderedDict
# 普通dict中没有顺序的概念，输出该dict时，顺序不一定
d = {'a': 1, 'c': 3, 'b': 2}
print(d)
d0 = dict([('a', 1), ('c', 3), ('b', 2)])
print(d0)
od = OrderedDict(d)
print(od)
try:
    iter(d)
except SyntaxError:
    print("dict is not iterable")
else:
    print("dict is iterable")
    # dict也是iterable的，可迭代keys和values，默认迭代keys
    for k in iter(d.keys()):
        print(k)
    for v in d.values():
        print(v)
# dict的pop弹出指定key，popitem随机弹出k,v对儿
d.pop('a')
print(d)
d.popitem()
print(d)
# dict.pop第二个参数类似default，可以指定默认返回值，而不是抛出错误
print(d.pop('f', "NULL"))
# OrderedDict的pop同dict.pop
print(od.pop('f', 'NULL'))
# OrderedDict的itempop方法体现了顺序字典的特性：可以实现LIFO(Last Input First Output)
# 和FIFO(First Input First Output)，而非随机弹出
od.popitem()    # LIFO
print(od)
od.popitem(last=False)  # FIFO
print(od)

# ChainMap 将多个dict按顺序连接起来，这些dict可以含有相同的keys，但在调用ChainMap时会有
# 优先级差别
# 设置默认k:v
defaults = {'color': 'red',
            'user': 'root'}
# 构造命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--color')
parser.add_argument('-u', '--user')
namespace = parser.parse_args()
cmd_line_args = {k: v for k, v in vars(namespace).items() if v}
# 三者组成ChainMap
combined = ChainMap(cmd_line_args, os.environ, defaults)    # os.environ 得到一些环境变量
# 优先级从高到低：命令行 > 环境变量 > defaults
print(combined['color'])
print(combined['user'])

# Counter是一个简单的计数器
def counter():
    c = Counter()
    # 用update方法就可以统计字符串、list等sequence
    c.update('abccdeee')
    # c继承自dict，加了一些关于计数器的方法
    print(c)
    c.update([55, 555])
    print(c)
    # most_common方法接受一个整数n，返回出现频率最高的前n个元素及其次数
    print(c.most_common(2))


if __name__ == '__main__':
    counter()

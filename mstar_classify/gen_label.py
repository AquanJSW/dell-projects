# -*- coding: utf-8 -*-

import os

# def slash_counter(str):
#     c = 0
#     for s in str:
#         if s == '/':
#             c += 1
#     return c
'''
tree:
/home/tjh/projects/py/mstar_classify/
data
├── 2S1
├── BMP2
├── BRDM2
├── BTR60
├── BTR70
├── D7
├── T62
├── T72
├── ZIL131
└── ZSU234
'''


def run_shell(cmd):
    c = os.popen(cmd)
    out = c.read()
    c.close()
    return out


PATH = '/home/tjh/projects/py/mstar_classify/'
class_dict = {'2S1': 0, 'BMP2': 1, 'BRDM2': 2, 'BTR60': 3, 'BTR70': 4,
              'D7': 5, 'T62': 6, 'T72': 7, 'ZIL131': 8, 'ZSU234': 9}

flag = '/home/tjh/projects/py/mstar_classify/data/'.__len__()
for root, dire, file in \
        os.walk('/home/tjh/projects/py/mstar_classify/data/', topdown=False):
    if not (file == []):  # 找到含图像的文件夹
        for name in file:
            dire = run_shell("echo %s | awk -F '/' '{print $8}'" % root)
            # if dire[len(dire) - 1] == '/':
            #     dire = dire[: len(dire)-1]
            dire = dire.strip()

            # read_shell('echo ".{0}/{1} {2}" >> {3}'.format(
            #     root[flag:], name, class_dict[dir], PATH))
            run_shell('echo {0}/{1} {2} >> {3}{4}.txt'.format(
                root, name, class_dict[dire], PATH, dire))

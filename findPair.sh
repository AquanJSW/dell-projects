#!/bin/bash

# 找左右导航相机中成对的图像，并分别放入左右图像文件夹中
# 思路：
# 1 以任意文件夹中的图像为基准，逐个提取图像名称name
# 2 将name的第二个字母换成不同的L或R，判断该名字是否在另一个文件夹中
# 3 如果在，则将两张图像分别复制到左右文件夹内，并分别重命名以编号1，2，3，。。。

i=1
for r_name in `ls ./right_nav`
do
    l_name_deriv=`echo $r_name | awk '{printf "%s%s", "NL", substr($0, 3)}'`
    if [ -f "./left_nav/${l_name_deriv}" ]
    then
        cp -f ./right_nav/${r_name} ./right_nav_pair/$i.jpg
        cp -f ./left_nav/${l_name_deriv} ./left_nav_pair/$i.jpg
        (( i++ ))
    fi
done

#!/bin/bash

# $ 符号的应用
# 用 $ 时最好加上双引号，或$()这样引用

# $@ 数组形式引用参数列表
# ${#var} 字符串长度
for x in "$@"
do
    echo "$x length: ${#x}"
done

# $* 引用参数整体
for x in "$*"
do
    echo "$x length: ${#x}"
done

# 参数数量
echo "$#"

# $[]用来计算表达式，可以不用在括号内边缘加空格
echo $[5 + 5]

# 显示当前bash 的选项 即 man bash 中所述
echo "$-"

# 显示最后一个后台运行进程的pid

echo "$!"

# !$ 非常有用，该变量引用上一条指令的参数，
# 使用场景，例如：
# ls /usr/share
# cd !$

# !! 重复上一条指令



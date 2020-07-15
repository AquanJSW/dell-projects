#!/bin/bash

# awk learning

# awk默认每行进行同样的操作
# 花括号内是action，默认 'pattern {action}'
awk '{print $1, $3}' log.txt

echo
# 格式化用printf, 操作同C语言
awk '{printf "%-8s %-10s\n", $1, $4}' log.txt

echo
# -F参数给定分割符，默认是任意长度的空格
# 可以用方括号连续给出分隔符
awk -F '[ ,]' '{print $1, $4}' log.txt

echo
# -v参数设定变量
# 如果变量是数，对字符串作运算时，字符串看做0
# 有几个变量就用几次 -v
awk -v a=1 '{print $1, $1+a}' log.txt
echo
awk -v a=1 -v b=2 '{print $1, $1+a, $1*b}' log.txt
echo
awk -v a='yes' '{print $1, a}' log.txt

# 可以用关系运算符过滤字符串
echo
awk '$1<3' log.txt
echo
awk '$1>2 && $2=="Are" {print $1, $2, $3}' log.txt


# 内建变量
echo 
awk '{print $0}' log.txt  # $0是完整字符串

echo
awk -F '[ ,]' 'BEGIN{printf "%9s %5s %5s %5s %5s %5s %5s %5s %5s\n", "FILENAME", "ARGC", "FNR", \
"FS", "NF", "NR", "OFS", "ORS", "RS"; printf "---------------------------------------------\n"} \
{printf "%9s %5s %5s %5s %5s %5s %5s %5s %5s\n", FILENAME, ARGC, FNR, FS, NF, NR, OFS, \
ORS, RS}' log.txt

echo
awk '$2=="are"' IGNORECASE=True log.txt   # IGNORECASE忽略大小写，也接受0，1形式的逻辑

# OFS指定行内分隔符，ORS指定行间分隔符
# 这些内建变量可以出现在多种地方
# 1 有缩略形式的可以出现在 awk 之后，例如 FS 的 -F 形式
# 2 非缩略形式既可以在 '' 之后出现，亦可以在 '{action}' 的action中出现，用分号分隔即可
echo
awk '$1>2 {OFS=" >>> "; print $4, $3, $2, $1}' FS='[ ,]' ORS=" && " log.txt


# 正则匹配
echo
echo
# 在第二列匹配 th 不区分大小写，$n ~ /pattern/ 是应用形式
# ~ 表模式开始， //中是模式
awk '$2 ~ /th/ {print $2, $4}' IGNORECASE=1 log.txt
echo

# 直接对每一行进行正则匹配不用加~
awk '/th/' IGNORECASE=1 log.txt

# 模式取反，即不匹配
echo
awk '$2 !~ /th/' IGNORECASE=1 log.txt


# AWK脚本
# 分为三部分
# 1 BEGIN{}中放的是匹配前执行的语句
# 2 {}中的是对每一行都应用的匹配语句
# 3 END{}中放的是匹配后执行的语句
echo
awk -f ./score.awk ./score.txt


# 内建函数
awk 

# python3
# encoding: utf-8
# 
# @Time    : 2022/03/15 10:29
# @Author  : enze
# @Email   : enzesu@hotmail.com
# @File    : tmp.py
# @Software: Pycharm

trail_list = [24, 8, 32, 31, 7, 15, 29, 6, 12, 26, 9, 1, 5, 28, 20, 11, 30, 21, 2, 4, 10, 13, 17, 25, 14, 22, 23, 19,
              16, 27, 18, 3]

tmp = []
for k in range(32):
    tmp.append(trail_list.index(k+1))

print(tmp)
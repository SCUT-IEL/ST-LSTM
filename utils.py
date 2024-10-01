#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/6 21:48   lintean      1.0         None
'''
import os

def makePath(path):
    if not os.path.isdir(path):  # 如果路径不存在
        os.makedirs(path)
    return path
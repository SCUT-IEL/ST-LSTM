import math

import mne
import numpy as np
from matplotlib import pyplot as plt

channel_num = 64
pos = np.load('pos.npy')
biosemi_montage = mne.channels.make_standard_montage('biosemi64')
channel_names = biosemi_montage.ch_names
biosemi_montage.plot(show_names=True)

# 绘制电极坐标
plt.figure()
plt.title('Graph(biosemi64)')
X_data, Y_data = [], []
for k_ch in range(64):
    X_data.append(pos[k_ch][0])
    Y_data.append(pos[k_ch][1])
plt.scatter(X_data, Y_data)

# 计算连线关系并绘制
edges = []
for cha1 in range(channel_num):
    for cha2 in range(channel_num):
        is_connect = False

        # 距离
        if math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(pos[cha1], pos[cha2])])) < 0.025:
            is_connect = True

        # 添加特殊连接
        if channel_names[cha1] == 'AFz' and channel_names[cha2] in ['AF3', 'AF4']:
            is_connect = True
        if channel_names[cha1] == 'POz' and channel_names[cha2] in ['PO3', 'PO4']:
            is_connect = True

        # 删除边缘连接
        if channel_names[cha1] in ['P5', 'P6'] and channel_names[cha2] in ['P9', 'P10']:
            is_connect = False
        if channel_names[cha2] in ['P5', 'P6'] and channel_names[cha1] in ['P9', 'P10']:
            is_connect = False

        # 将连接关系添加到edges中
        if is_connect:
            edges.append([cha1, cha2])
            x = [pos[cha1][0], pos[cha2][0]]
            y = [pos[cha1][1], pos[cha2][1]]
            plt.plot(x, y)

np.save('edges.npy', edges)
plt.show()

# python3
# encoding: utf-8
#
# 绘制我们自己的数据库的电极示意图。
#
# @Time    : 2022/06/13 10:24
# @Author  : enze
# @Email   : enzesu@hotmail.com
# @File    : draw_channels.py
# @Software: Pycharm


import mne
from matplotlib import pyplot as plt

biosemi_montage = mne.channels.make_standard_montage('standard_1020')
biosemi_montage.plot(show_names=True)

# 统计电极名称及其坐标
channels = {}
for k in range(len(biosemi_montage.ch_names)):
    channel_name = biosemi_montage.ch_names[k]
    channel_pos = biosemi_montage.dig[k + 3]['r'] * 10
    channels[channel_name] = channel_pos

# 绘制坐标图
my_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz',
               'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'F1',
               'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5',
               'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz',
               'FCz']

# 添加幕布
fig = plt.figure(dpi=300, figsize=(10, 10))
plt.axis('equal')
xy_scale = 2
plt.xlim(-xy_scale, xy_scale)
plt.ylim(-xy_scale, xy_scale)
plt.axis('off')

# 绘制外圆
circle = plt.Circle([0, 0], 0.95, color='black', fill=False)
plt.gcf().gca().add_artist(circle)

for cha in my_channels:
    pos = channels[cha]

    # 添加电极名称
    plt.text(pos[0], pos[1], cha, fontsize=4, family='Time New Roman', verticalalignment='center',
             horizontalalignment='center')

    # 绘制电极圆圈
    circle = plt.Circle(pos[0:2], 0.05, color='black', fill=False, linewidth=0.5)
    plt.gcf().gca().add_artist(circle)

plt.show()

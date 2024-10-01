# python3
# encoding: utf-8
# 
# @Time    : 2022/05/13 10:24
# @Author  : enze
# @Email   : enzesu@hotmail.com
# @File    : my_splite.py
# @Software: Pycharm
import numpy as np




def data_split(eeg, voice, label, target, time_len, overlap):
    """
    将序列数据转化为样本，并按照5折的方式的排列整齐
    不兼容128Hz以外的数据（含有语音和脑电信号）
    :param eeg: 脑电数据，列表形式的Trail，每个Trail的形状是Time*Channels
    :param voice: 语音数据，列表形式的Trail，每个Trail的形状是Time*2
    :param label: 标签数据，根据targe决定输出标签
    :param target: 确定左右/讲话者的分类模型
    :param time_len: 样本的长度
    :param overlap: 样本的重叠率
    :return:
    """

    sample_len = int(128 * time_len)


    all_train=[]
    all_lable=[]


    for k_tra in range(8):
        trail_eeg = eeg[k_tra]
        #trail_voice = voice[k_tra]
        trail_label = label[k_tra][target]

        #trail_voice = np.transpose(trail_voice, axes=[1,0])

        # 确定重叠率的数据
        over_samples = int(sample_len * (1 - overlap))
        over_start = list(range(0, sample_len, over_samples))
        # 根据起点划分数据
        for k_sta in over_start:
            tmp_eeg = set_samples(trail_eeg, k_sta, sample_len, overlap)
            #tmp_voice = set_samples(trail_voice, k_sta, sample_len, overlap)

            #my_trail_samples = np.concatenate((my_trail_samples, tmp_eeg), axis=0)
            all_train.append(tmp_eeg)
            #my_voice_samples = np.concatenate((my_voice_samples, tmp_voice), axis=1)
            #my_labels = np.concatenate((my_labels, trail_label * np.ones(tmp_eeg.shape[1])), axis=0)
            all_lable.append(trail_label * np.ones(tmp_eeg.shape[0]))


    return all_train,  all_lable


def set_samples(trail_data, k_sta, sample_len, overlap):
    # 切分整数长度
    data_len, channels_num = trail_data.shape[0], trail_data.shape[1]
    k_end = (data_len - k_sta) // sample_len * sample_len + k_sta
    trail_data = trail_data[k_sta:k_end, :]

    # cutoff
    trail_data = np.reshape(trail_data, [-1, sample_len, channels_num])


    trails_num = trail_data.shape[0] // 5 * 5
    trail_data = trail_data[0:trails_num, :, :]
    trail_data = np.reshape(trail_data, [-1, sample_len, channels_num])



    return trail_data



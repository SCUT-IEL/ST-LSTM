#!/usr/bin/env python
# coding: utf-8
#
# In[1]:


import sys
import torch

import numpy as np
from sklearn.model_selection import KFold

from neural_networks import sCNN


import utils as util
from importlib import reload

from preprocess_xz import preprocess
# In[2]:
from my_splite_xz import data_split

import xlwt


import torch.nn as nn

batch_size = 32
learning_rate = 1e-3

num_epoch = 100
splited_data_document = f"."
project_root_path = f"."

visualization_epoch = []
visualization_window_index = []
dataset_name = 'KUL'
l_freq, h_freq = 1, 32
time_len = 1
overlap = 0.
result = ["1"]
five_acc=[]



class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datas):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        x, y = datas[0], datas[1]
        imgs = []
        for i in range(len(x)):  # 迭代该列表#按行循环txt文本中的内
            imgs.append((x[i], y[i], i))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        # self.transform = transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        img, label, index = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        return img, label, index  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def get_logger(name, log_path):
    import logging
    reload(logging)

    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 第二步，创建一个handler，用于写入日志文件
    logfile = log_path + "/Train_" + name + ".log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def demo(name='1', visible=True, log_path=project_root_path + "/result/snn", q=None):
    global conv_weight
    accuracies = []
    device = torch.device('cuda:3')
    torch.set_num_threads(1)
    logger = get_logger(name, util.makePath(log_path))


    def test(loader, length, conv_weight):
        corrects = 0
        total_loss = 0
        net.eval()
        with torch.no_grad():

            for (inputs, labels, window_index) in loader:

                outputs = net(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs[0], labels)
                predicted = outputs[0].data.round()
                corrects += (predicted.cpu().numpy() == np.eye(2)[labels.cpu()]).sum() / 2
                total_loss += loss.item()

        acc = corrects / length
        return acc, total_loss / length


    eeg, voice, label = preprocess(dataset_name, name, l_freq=l_freq, h_freq=h_freq, is_ica=1
                                   )
    eeg, label = data_split(eeg, voice, label, 0, time_len, overlap=0.5)
    eeg = np.array(eeg)
    data = np.reshape(eeg, [-1, 128, 64])
    label = np.array(label)
    labels = np.reshape(label, [-1])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 迭代每个折叠
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold_idx + 1}")


        data_train, data_test = data[train_idx], data[test_idx]
        label_train, label_test = labels[train_idx], labels[test_idx]
        val_idx = np.random.choice(train_idx, size=len(test_idx), replace=False)
        train_idx = np.setdiff1d(train_idx, val_idx)
        data_val, label_val = data[val_idx], labels[val_idx]
        data_train, label_train = data[train_idx], labels[train_idx]

        # 数据划分
        x_tra, y_tra = torch.from_numpy(data_train).float().to(device), torch.from_numpy(
            np.array(label_train)).long().to(device)
        x_test, y_test = torch.from_numpy(data_test).float().to(device), torch.from_numpy(
            np.array(label_test)).long().to(device)
        x_val, y_val = torch.from_numpy(data_val).float().to(device), torch.from_numpy(
            np.array(label_val)).long().to(device)

        # loss 可视化



        train_loader = torch.utils.data.DataLoader(dataset=MyDataset([x_tra, y_tra]), batch_size=batch_size,
                                                   shuffle=True, num_workers=0)

        test_loader = torch.utils.data.DataLoader(dataset=MyDataset([x_test, y_test]), batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0)
        val_loader = torch.utils.data.DataLoader(dataset=MyDataset([x_val, y_val]), batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0)

        T = 10
        len_out = int(2)

        net = sCNN(len_out, T, 128).to(device)  # create model

        criterion = nn.CrossEntropyLoss().to(device)
        logger.info('Number of model parameters is {}'.format(sum(p.numel() for p in net.parameters())))

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.0007)

        best_val_acc = 0
        best_acc = 0
        best_acc_epoch = 0

        conv_weight = 0
        for epoch in range(num_epoch):
            net.train()

            for (train_fingerprints, train_ground_truth, window_index) in train_loader:

                inputs = train_fingerprints
                labels = train_ground_truth
                outputs = net(inputs)

                loss = criterion(outputs[0], labels)

                optimizer.zero_grad()
                net.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()


            train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = 0, 0, 0, 0, 0, 0

            # scheduler.step(0.1)

            # 测试
            if visible:
                train_acc, train_loss = test(train_loader, len(x_tra), conv_weight)
                test_acc, test_loss = test(test_loader, len(x_test), conv_weight)
                val_acc, val_loss = test(val_loader, len(x_val), conv_weight)

                logger.info(
                    str(epoch) + ' epoch ,loss is: ' + str(train_loss) + " " + str(val_loss) + " " + str(test_loss))
                logger.info(str(train_acc) + " " + str(val_acc) + " " + str(test_acc))
                logger.info("")


                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_acc = test_acc
                    best_acc_epoch = epoch

                if epoch == num_epoch - 1:
                    logger.info("S" + name + ": " + str(test_acc))



            logger.info("S" + name + ": " + str(best_acc) + " epoch: " + str(best_acc_epoch))

            five_acc.append(best_acc)


    global result
    #result = np.vstack((result, five_acc))

    print(accuracies)






if __name__ == "__main__":
    for num in range(1, 17):
        demo(f'{num}')

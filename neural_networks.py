##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################
from thop import profile
import torch.nn as nn
import torch
from snn import LSTM,SA
from functional import ZeroExpandInput_CNN
import numpy as np
from power.container import SPower
from power.snn_energy import energy_ann_fc, energy

class sCNN(nn.Module):
    def __init__(self, num_labels, T,simple_lenth):# T =TIME?
        super(sCNN, self).__init__()
        self.T = 10
        self.out_num = num_labels
        self.encoder1 = SA(1, 64,  pooling=2,simple_lenth=simple_lenth)
        self.encoder2 = LSTM(simple_lenth=simple_lenth,lstm_units=4)



        self.output = nn.Linear(4, num_labels)
        self.sigmoid=nn.Sigmoid()

        self.powerp = {
            "snn_process": True,
            "verbose": False
        }

    def forward(self, x):
        visualization_weights = []
        #self.energys = SPower()


        x_spike, x = ZeroExpandInput_CNN.apply(x, self.T)
        x_spike = x_spike.transpose(2, 3)
        x_spike = x_spike.transpose(1, 2)
        x_spike, x = self.encoder1(x_spike, x)
        #data_in = x.clone().cpu()
        data_spike = x_spike.clone().detach().cpu()

        ratio_lstm=(np.count_nonzero(data_spike) / np.product(data_spike.shape))
        #print("lstm",ratio_lstm)
        #macs, paras = profile(self.encoder2, inputs=(data_spike, data_in))
        x_spike, x = self.encoder2(x_spike, x) #mac of encoder2 is 4718592

        #ratio2=np.count_nonzero(data_spike)/ np.product(data_spike.shape)





        x = x.view(x.size(0), -1)

        return self.sigmoid(self.output(x)),visualization_weights






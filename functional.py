# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:49:34 2020

@author: win10
"""


import torch.nn as nn
import torch
import torch.nn.functional as F
from settings import SETTINGS
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#######################3
device = torch.device(SETTINGS.training.device)
path="/home/zhangran/zhangran/lstm_snn"
def snn_heatmap(origin_data, log_path=path, window_index=None, epoch=1):
    """
    snn的可视化 'Visual Explanations from Spiking Neural Networks using Interspike Intervals'
    @param _: Dotmap结构 其中包含: data(numpy)、figsize、title、x_label、y_label、fontsize
    @param log_path: 保存图片的路径
    @param window_index: 数据的原始index
    @param epoch: 目前模型的迭代数
    """
    spike = origin_data.data
    __title = "epoch"
    spike = torch.transpose(spike, 2, 0)
    spike = torch.transpose(spike, 1, 0)
    spike = torch.unsqueeze(spike, 0)
    if len(spike.shape) < 4:
        print(spike.shape)
        print(spike)
        print("error! check the code! ")
    else:
        # SAM实现
        gamma = 0.2
        __channel,__width,  __height, __time = spike.shape

        # for t in range(__time):
        #     for c in range(__channel):
        #         if np.sum(spike[t, c]) > 0:
        #             print("[" + str(t) + ", " + str(c) + "] have spike!")

        for time_step in range(1, __time + 1):
            sam_matrix = np.zeros(shape=[__height, __width])
            for h in range(__height):
                for w in range(__width):
                    for c in range(__channel):
                        ncs = 0
                        for t in range(time_step - 1):
                            if spike[c, w,h , t] != 0:
                                ncs += np.exp(-gamma * (time_step - 1 - t))
                        sam_matrix[h, w] += ncs * spike[c,w , h, time_step - 1]
            log_path = path + "/" + "epoch: "+str(epoch) + "_time_step: "+str(time_step)
            origin_data.data = torch.from_numpy(sam_matrix).to(device)
            data=origin_data.cpu().numpy()
            Mean = np.max(data) - np.min(data)
            if Mean!=0:

               data = (data - np.min(data)) / Mean
            origin_data.title = __title + "_timestep" + str(time_step)
            heat_img=sns.heatmap(data,vmin=0.0,vmax=1)
            fig = heat_img.get_figure()
            fig.savefig(log_path, dpi=400)
            plt.close()
#######################




class SAIF(torch.autograd.Function):

	@staticmethod
	def forward(ctx, spike_in, features_in, k_weight=None, q_weight=None,v_weight=None,device=torch.device(SETTINGS.training.device)):
		"""
		args:
			spike_in: (N, T, in_channels, iH, iW)
			features_in: (N, in_channels, iH, iW)
			weight: (out_channels, in_channels, kH, kW)
			bias: (out_channels)
		"""

		max_yuzhi=0.7
		N,T, _ ,_ = spike_in.shape

		pot_in_k=spike_in.matmul(k_weight.t())


		# init the membrane potential with the bias


		spike_out_k = torch.zeros_like( pot_in_k,device=device)
		pot_aggregate=torch.zeros([len(spike_out_k),len(spike_out_k[0,0]),len(spike_out_k[0,0,0])],device=device)
		spike_mask = torch.zeros_like(pot_aggregate,device=device).float()


		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += pot_in_k[:, t, :,:]
			bool_spike = torch.ge(pot_aggregate, max_yuzhi).float()

			bool_spike *= (1 - spike_mask)

			spike_out_k[:, t, :,:] = bool_spike
			pot_aggregate -= bool_spike*max_yuzhi

		spike_count_out_k = torch.sum(spike_out_k, dim=1)



		pot_in_q= spike_in.matmul(q_weight.t())

		# init the membrane potential with the bias

		spike_out_q = torch.zeros_like(pot_in_q, device=device)
		pot_aggregate=torch.zeros([len(spike_out_q),len(spike_out_q[0,0]),len(spike_out_q[0,0,0])],device=device)
		spike_mask = torch.zeros_like(pot_aggregate, device=device).float()

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += pot_in_q[:, t, :,:]
			bool_spike = torch.ge(pot_aggregate, max_yuzhi).float()

			bool_spike *= (1 - spike_mask)

			spike_out_q[:, t, :,:] = bool_spike
			pot_aggregate -= bool_spike*max_yuzhi

		spike_count_out_q = torch.sum(spike_out_q, dim=1)



		# init the membrane potential with the bias

		pot_in_v = spike_in.matmul(v_weight.t())



		spike_out_v = torch.zeros_like(pot_in_v, device=device)
		pot_aggregate=torch.zeros([len(spike_out_v),len(spike_out_v[0,0]),len(spike_out_v[0,0,0])],device=device)
		spike_mask = torch.zeros_like(pot_aggregate, device=device).float()

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += pot_in_v[:, t, :,:].squeeze()
			bool_spike = torch.ge(pot_aggregate, max_yuzhi).float()

			bool_spike *= (1 - spike_mask)

			spike_out_v[:, t, :,:] = bool_spike
			pot_aggregate -= bool_spike*max_yuzhi

		spike_count_out_v = torch.sum(spike_out_v, dim=1)

		spike_out_k_numpy = spike_out_k.clone().cpu()
		spike_out_q_numpy=spike_out_q.clone().cpu()
		ratio_k=(np.count_nonzero(spike_out_k_numpy) / np.product(spike_out_k_numpy.shape))
		ratio_q=(np.count_nonzero(spike_out_q_numpy) / np.product(spike_out_q_numpy.shape))

		w_spike = (torch.matmul(spike_out_k, torch.transpose(spike_out_q, 2, 3))) / 2
		w_spike_numpy = w_spike.clone().cpu()
		spike_out_v_numpy = spike_out_v.clone().cpu()
		ratio_w = (np.count_nonzero(w_spike_numpy) / np.product(w_spike_numpy.shape))
		ratio_v = (np.count_nonzero(spike_out_v_numpy) / np.product(spike_out_v_numpy.shape))



		w_spike_soft=torch.zeros_like(w_spike)
		for i in range(10):
			w_spike_soft[:,i,:,:] = torch.softmax(w_spike[:,i,:,:], 1)*1.5



		#w=(torch.bmm(spike_count_out_k, torch.transpose(spike_count_out_q, 1,2)))
		#w_spike= torch.softmax(w_spike, 1)
		#w = torch.softmax(w, 1)
		output_spike=torch.matmul(w_spike_soft,spike_out_v)


		output_spike_real=torch.zeros_like(output_spike, device=device)
		pot_aggregate=torch.zeros([len(output_spike),128,64],device=device)
		spike_mask = torch.zeros_like(pot_aggregate, device=device).float()

		for t in range(T):
			pot_aggregate += output_spike[:, t, :,:].squeeze()
			bool_spike = torch.ge(pot_aggregate, max_yuzhi).float()

			bool_spike *= (1 - spike_mask)

			output_spike_real[:, t, :,:] = bool_spike
			pot_aggregate -= bool_spike*max_yuzhi
		output=torch.sum(output_spike_real, dim=1)
		#snn_heatmap(w_spike_soft[0], epoch=1)
		#output=torch.bmm(w,spike_count_out_v)
		#print([ratio_k, ratio_q, ratio_w, ratio_v])
		return output_spike_real, output
	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		grad_spike_count_out = grad_spike_count_out.clone()

		grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding, grad_pooling = None, \
				None, None, None, None, None, None

		return grad_spike_in, grad_spike_count_out, grad_weight, grad_device, grad_bias, \
				grad_stride, grad_padding, grad_pooling

class LSTMIF_AvgPool(torch.autograd.Function):

	@staticmethod
	def forward(ctx, spike_in, features_in, weight, device=torch.device(SETTINGS.training.device), lstm_units=4,
				simple_lenth=128, bn1d=None):
		"""
        args:
            spike_in: (N, T, in_channels, iH, iW)
            features_in: (N, in_channels, iH, iW)
            weight: (out_channels, in_channels, kH, kW)
            bias: (out_channels)
        """

		snn_lstm = nn.LSTM(64, lstm_units, 1, batch_first=True)
		for i in range(len(snn_lstm.all_weights[0])):
			snn_lstm.all_weights[0][i].data = weight[0][i].data
		# tanh_snn=nn.Tanh()

		N, T, iH, iW = spike_in.shape

		pot_aggregate = torch.zeros_like(features_in)
		N, outW = pot_aggregate.shape
		spike_out = torch.zeros(N, T, outW, device=device)
		spike_mask = torch.zeros_like(pot_aggregate, device=device).float()

		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			# spike_in[:, t, :, :]=tanh_snn(spike_in[:, t, :, :])

			# spike_in[:, t, :, :]=dropout(spike_in[:,t,:,:])

			output, _ = (snn_lstm(spike_in[:, t, :, :]))
			output = (output.permute(0, 2, 1)).permute(0, 2, 1)
			# output = tanh_snn(output)
			output = torch.reshape(output, (-1, 1, int(simple_lenth), lstm_units))
			# output = torch.transpose(output, 2, 3)
			output = F.avg_pool2d(output, (int(simple_lenth), 1))
			output = torch.reshape(output, (-1, lstm_units))

			pot_aggregate += output
			bool_spike = torch.ge(pot_aggregate, 0.7).float()

			bool_spike *= (1 - spike_mask)

			spike_out[:, t, :] = bool_spike
			pot_aggregate -= bool_spike * 0.7

		# spike_mask += bool_spike
		# spike_mask[spike_mask > 0] = 1

		spike_count_out = torch.sum(spike_out, dim=1)

		return spike_out, spike_count_out

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		grad_spike_count_out = grad_spike_count_out.clone()

		grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding, grad_pooling = None, \
																									  None, None, None, None, None, None

		return grad_spike_in, grad_spike_count_out, grad_weight, grad_device, grad_bias, \
			   grad_stride, grad_padding, grad_pooling

######################

























########################
class ZeroExpandInput_CNN(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input_image, T, device=torch.device(SETTINGS.training.device)):
		"""
		Args:
			input_image: normalized within (0,1)
		"""
		#N, dim = input_image.shape
		#input_image_sc = input_image
		#zero_inputs = torch.zeros(N, T-1, dim).to(device)
		#input_image = input_image.unsqueeze(dim=1)
		#input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

		#return input_image_spike, input_image_sc
		# if len(input_image.shape)==4:
		# 	input_image = input_image.squeeze(1)
		# print(input_image.shape)
		# input_image = input_image.sum(1)

		##################################
		# input_image_tmp = (input_image-input_image.min())/(input_image.max()-input_image.min()+1e-10) # normalized to [0-1]
		# encode_window = int(T/3)-1
		# input_image_spike = torch.zeros(batch_size,T,channel,spec_length,fing_width).to(device)
		# input_sc_index =((1-input_image_tmp)*encode_window).ceil().unsqueeze(1).long()
		# input_image_spike=input_image_spike.scatter(1,input_sc_index,1).float()
		####################################

		batch_size,  spec_length, fing_width = input_image.shape
		input_image_sc = input_image
		zero_inputs = torch.zeros(batch_size, spec_length, fing_width, T-1).to(device)
		input_image = input_image.unsqueeze(dim=-1)
		input_image_spike = torch.cat((input_image, zero_inputs), dim=-1)
		return input_image_spike, input_image_sc

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		return None, None, None

class ZeroExpandInput_MLP(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input_image, T, device=torch.device(SETTINGS.training.device)):
		"""
		Args:
			input_image: normalized within (0,1)
		"""
		N, dim = input_image.shape
		input_image_sc = input_image
		zero_inputs = torch.zeros(N, T-1, dim).to(device)
		input_image = input_image.unsqueeze(dim=1)
		input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

		return input_image_spike, input_image_sc

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		return None, None, None




import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functional import   LSTMIF_AvgPool,SAIF
from settings import SETTINGS


channels_num=64

###############################################


class SA(nn.Module):
	"""
	W
	"""
	def __init__(self, Cin, Cout, device=torch.device(SETTINGS.training.device), stride=1, padding=0, bias=True, weight_init=2.0, pooling=1,lstm_units=4,simple_lenth=128):

		super(SA, self).__init__()
		self.SAIF = SAIF.apply

		self.bn1d_k = torch.nn.BatchNorm1d(lstm_units,affine=True)
		self.bn1d_q = torch.nn.BatchNorm1d(lstm_units, affine=True)
		self.bn1d_v = torch.nn.BatchNorm1d(64, affine=True)
		self.dense_k=torch.nn.Linear(64, lstm_units)
		self.dense_q = torch.nn.Linear(64, lstm_units)
		self.dense_v = torch.nn.Linear(64, 64)
		self.device = device
		self.dropout=nn.Dropout(p=0)
		self.stride = stride
		self.padding = padding
		self.pooling = pooling


	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate conv2d layer
		k = (self.dense_k(input_features_sc).permute(0,2,1))
		linearif_weight_k = self.dense_k.weight
		bnGamma = self.bn1d_k.weight
		bnVar = self.bn1d_k.running_var
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		k_weightNorm = torch.mul(linearif_weight_k.permute(1, 0), ratio).permute(1, 0)



		q = self.bn1d_q(self.dense_q(input_features_sc).permute(0,2,1))
		linearif_weight_q = self.dense_q.weight
		bnGamma = self.bn1d_q.weight
		bnVar = self.bn1d_q.running_var
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		q_weightNorm = torch.mul(linearif_weight_q.permute(1, 0), ratio).permute(1, 0)






		w = (torch.bmm(torch.transpose(k, 1,2), q))/2

		w=torch.softmax(w,1)




		v=self.bn1d_v(self.dense_v(input_features_sc).permute(0,2,1))#zai zhe li jia do
		v=F.relu(v)
		linearif_weight_v = self.dense_v.weight
		bnGamma = self.bn1d_v.weight
		bnVar = self.bn1d_v.running_var
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		v_weightNorm = torch.mul(linearif_weight_v.permute(1, 0), ratio).permute(1, 0)

		output = torch.bmm(w, v.permute(0,2,1))





		output_features_st, output_features_sc = self.SAIF(input_feature_st, output,\
														k_weightNorm,q_weightNorm,v_weightNorm, self.device
														)

		return output_features_st, output_features_sc
####################


##########################






class LSTM(nn.Module):
	"""
	W
	"""
	def __init__(self,  device=torch.device(SETTINGS.training.device), stride=1, padding=0,  pooling=1,lstm_units=4,simple_lenth=128):

		super(LSTM, self).__init__()
		self.LSTMIF = LSTMIF_AvgPool.apply
		self.lstm = torch.nn.LSTM(64,lstm_units,1,batch_first=True,dropout=0.3)
		self.bn1d=nn.BatchNorm1d(lstm_units,affine=True)
		self.bn1d_snn = nn.BatchNorm1d(lstm_units, affine=False)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.pooling = pooling
		self.lstm_units=lstm_units
		self.sample_lenth=simple_lenth
		self.tanh=nn.Tanh()
		self.dropout=nn.Dropout(p=0.3)

	def forward(self, input_feature_st, input_features_sc):

		# weight update based on the surrogate conv2d layer
		#input_features_sc=self.tanh(input_features_sc)

		#input_features_sc=self.dropout(input_features_sc)
		output,_=(self.lstm(input_features_sc))


		output=(output.permute(0,2,1)).permute(0,2,1)

		bnGamma = self.bn1d.weight
		bnBeta = self.bn1d.bias
		bnMean = self.bn1d.running_mean
		bnVar = self.bn1d.running_var
		self.bn1d_snn.weight=bnGamma
		self.bn1d_snn.bias=bnBeta
		self.bn1d_snn.running_mean=bnMean
		self.bn1d_snn.running_var=bnVar
		lstm_weight = self.lstm.all_weights






		output = torch.reshape(output, (-1, 1,int(self.sample_lenth), self.lstm_units ))

		output=F.avg_pool2d(output,(int(self.sample_lenth),1))
		output=torch.reshape(output,(-1,self.lstm_units))




		output_features_st, output_features_sc = self.LSTMIF(input_feature_st, output,\
														lstm_weight, self.device,self.lstm_units
														,self.sample_lenth,self.bn1d_snn)

		return output_features_st, output_features_sc
##################################################


class sDropout(nn.Module):
	def __init__(self, layerType, pDrop):
		super(sDropout, self).__init__()

		self.pKeep = 1 - pDrop
		self.type = layerType # 1: Linear 2: Conv

	def forward(self, x_st, x_sc):
		if self.training:
			T = x_st.shape[-1]
			mask = torch.bernoulli(x_sc.data.new(x_sc.data.size()).fill_(self.pKeep))/self.pKeep
			x_sc_out = x_sc * mask
			x_st_out = torch.zeros_like(x_st)
			
			for t in range(T):
				# Linear Layer
				if self.type == 1:
					x_st_out[:,t,:] = x_st[:,t,:] * mask
				# Conv1D Layer
				elif self.type == 2:
					x_st_out[:,t,:,:] = x_st[:,t,:,:] * mask
				# Conv2D Layer					
				elif self.type == 3:
					x_st_out[..., t] = x_st[..., t] * mask
		else:					
			x_sc_out = x_sc
			x_st_out = x_st
			
		return x_st_out, x_sc_out
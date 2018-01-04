import torch
from torch.autograd import Variable
import torch.nn as nn

class SpM(nn.Module):
	def __init__(self,input_dim):
		super(SpM,self).__init__()
		self.dnn = nn.Sequential(
						nn.Linear(input_dim,256),
						nn.BatchNorm1d(256),
						nn.ReLU(),
						nn.Linear(256,256),
						nn.BatchNorm1d(256),
						nn.ReLU(),
						nn.Linear(256,input_dim)
				   )
		
	def forward(self,x):
		return torch.sum(self.dnn(x),dim=1)
	
	def get_Aw(self,x):
		return self.dnn(x)


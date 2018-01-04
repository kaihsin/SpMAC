import os,sys
import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import SpM
import matplotlib.pyplot as plt

#Gt_PATH = 'G_exact_b20c15.npy'
#Kdw_PATH = 'Kdw_exact_b20c15.npy'
Aw_PATH  = 'Aw_exact_b20c15.npy'
w_PATH   = 'w_exact_b20c15.npy'
dw = 0.0039062500
input_dim = 1024

Valid_Aw = np.load(Aw_PATH)
ws = np.load(w_PATH)

x = np.ones(input_dim)*dw
x = torch.from_numpy(x).float()

model = SpM(input_dim)
model.load_state_dict(torch.load('SpM.model'))

model.eval()
pred_Aw = model.get_Aw(Variable(x.unsqueeze(0),requires_grad=False))
pred_Aw = pred_Aw.squeeze(0).data.numpy()


plt.plot(ws,Valid_Aw,'-',label='exact Aw')
plt.plot(ws,pred_Aw,'x',label='predict Aw')
plt.show()



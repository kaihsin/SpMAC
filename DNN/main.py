import os,sys
import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import SpM

Gt_PATH = 'G_exact_b20c15.npy'
Kdw_PATH = 'Kdw_exact_b20c15.npy'
Aw_PATH  = 'Aw_exact_b20c15.npy'
dw = 0.0039062500

y = np.load(Gt_PATH)
x = np.load(Kdw_PATH)
input_dim = np.shape(x)[1]

learn_rate = 0.00001
regu = 1e-9
valid_rate = 0.9
Batch_sz = 128
epoch = 40




N_Train = int(len(y)*0.9)

#print (np.shape(y))
#print (np.shape(x))

y = torch.from_numpy(y).float()
x = torch.from_numpy(x).float()

train_loader = torch.utils.data.DataLoader(\
                torch.utils.data.TensorDataset(x[:N_Train],y[:N_Train]),\
                batch_size=Batch_sz,\
                shuffle=True)

valid_loader = torch.utils.data.DataLoader(\
                torch.utils.data.TensorDataset(x[N_Train:],y[N_Train:]),\
                batch_size=Batch_sz,\
                shuffle=False)

del y 
del x


model = SpM(input_dim)
optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate,weight_decay=regu)
loss_fx   = nn.MSELoss()
if sys.argv[1] == 1:
	model.load_state_dict(torch.load('SpM.model'))



model.cuda()
model.train()
print ("[Start]")
for e in range(epoch):
	for i, (x,y) in enumerate(train_loader):
		pred = model(Variable(x.cuda(),requires_grad=False))
		loss = loss_fx(pred,Variable(y.cuda(),requires_grad=False).float())
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i%100 == 99:
			print ("  [Batch %d] loss :%f"%(i,loss.cpu().data.numpy()[0]))



	print ("e %d] loss : %f"%(e,loss.cpu().data.numpy()[0]))

model.cpu()
torch.save(model.state_dict(),'SpM.model')


## test:
model.cuda()
model.eval()
Chi2 = 0
tot = 0
for e in range(epoch):
	for i, (x,y) in enumerate(valid_loader):
		pred = model(Variable(x.cuda(),requires_grad=False))
		Chi2 += np.sum((y.numpy() - pred.cpu().data.numpy())**2)
		tot += len(pred)	

print (Chi2/tot)




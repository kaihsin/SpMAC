from testfunc import *
import numpy as np 
import matplotlib.pyplot as plt

# must be symetric in this case
wi = -2
wf = 2
Nw = 1024

Ntaus = 100000
tau_i = 0
tau_f = 10

b = 2   # cos distortion
c = 1.5 # gauss distortion


## b control the cosine oscillation 
## c control the gaussian 
## handy : 
##         if b > c , cosine dominant 
## 		   if c > b , gaussian dominant with cosine ripple 
gen = Cosine_Gaussian(2,1.5) 

## get Gt 
dw = (wf-wi)/Nw
dtau = (tau_f - tau_i)/Ntaus

ws = np.arange(wi,wf,dw)
taus = np.arange(tau_i,tau_f,dtau)

## training set.
G_t = gen.Gt(taus)
Kdw_tj = np.array([gen.kernel(ws,t)*dw for t in taus])

## validation set.
Aw = gen.target(ws)

np.save('G_exact_b20c15',G_t)
np.save('Kdw_exact_b20c15',Kdw_tj)
np.save('Aw_exact_b20c15',Aw)
np.save('w_exact_b20c15',ws)
f = open('details_exact_b20c15','w')
f.write('dw = %10.10f'%(dw))
f.close()




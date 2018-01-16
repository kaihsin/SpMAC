from testfunc import *
import numpy as np 
import matplotlib.pyplot as plt



## b control the cosine oscillation 
## c control the gaussian 
## handy : 
##         if b > c , cosine dominant 
## 		   if c > b , gaussian dominant with cosine ripple 
gen = Cosine_Gaussian(2,1.5) 
dw = 0.0001
wi = 10
taus = np.arange(0,10,0.01)
w = np.arange(-wi,wi,dw)

## get Gt 
y = gen.Gt(taus)

## mimic error bar (std):
y_std = np.random.uniform(0.01,0.02,len(y))

plt.close()

plt.figure(1)
plt.title("G(t)")
plt.errorbar(taus,y,yerr=y_std,label='Gt with random noise',fmt='x')
plt.plot(taus,[ np.sum(gen.kernel(w,taus[i])*gen.target(w))*dw for i in range(len(taus))],'-',label='num_int')
plt.xlabel("t")

plt.figure(2)
plt.plot(w,gen.kernel(w,taus[2]),'.',label='kernal')
plt.plot(w,gen.target(w),'.',label='target')
plt.legend()
plt.xlabel("w")

plt.show()






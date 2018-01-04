import numpy as np
import os,sys
from math import *

class Cosine_Gaussian:
	"""
		G(t) = \int target(w,b) * kernel(w,tau) dw

		@target function :
			y = cos(b*x) * e^{-(c*x)^2}
			
		@kernel function :
			y = e^{-(t*x)^2}

		@Gt exact:
			G(t) = \sqrt(\pi)/(c+t) * exp( -[b/(2*(c+t))]^2 )

	"""
	def __init__(self,b,c):
		self.b = b
		self.c = c

	def kernel(self,x,tau):
		return np.exp(-tau**2 * x**2)

	def target(self,x):
		return np.cos(self.b*x) * self.c / np.sqrt(np.pi) * np.exp(0.25*(self.b/self.c)**2-(self.c*x)**2) 

	def Gt(self,tau):
		return self.c/np.sqrt(tau**2+self.c**2) * np.exp(0.25*self.b**2*(self.c**-2 - (self.c**2+tau**2)**-1))





def ReSample(x,std,Nsample):
	return np.random.normal(x,std,(len(x),Nsample))

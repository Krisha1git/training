import numpy as np
#import pandas as pd
from numpy import genfromtxt

array = genfromtxt("insurance.csv", delimiter=',')
X = array[ :, 0:5]
y = array[ :, 5]
y = np.reshape(y,(1338,1))
m = len(y)
#m= number of samples
n = 5 #number of features
#feature normalise
mu = np.zeros((1,n))
sigma = np.zeros((1,n))
mu = np.mean(X,0)
sigma = np.std(X,0)
X = (X - mu)/sigma

#we treat the constant term theta0 in linear regression as theta0*X0 where X0=1
X = np.c_[  np.ones(m),X ]

alpha = 0.1
num_iters = 500

theta = np.random.randn(6,1)
#print(theta)
#print(X.shape)
#gradient descent
for i in range(1,400,1):
    A = (np.dot(X,theta)) - y
    B = (np.dot(np.transpose(X),A))/m
    theta = theta - (alpha*B)

#print (A.shape)    
#print (theta.shape)
#print(B.shape)
print(theta)

import numpy as np
#import pandas as pd
from numpy import genfromtxt

array = genfromtxt("insurance.csv", delimiter=',')

X = array[ :, 0:5]
y = array[ :, 5]
m = len(y)
#we treat the constant term theta0 in linear regression as theta0*X0 where X0=1
X=np.c_[  np.ones(m),X ]

# % Calculate the parameters from the normal equation
a = np.linalg.inv((np.dot((np.transpose(X)),X))),
b=np.dot( (np.transpose(X)) , y)
theta=np.dot(a,b)
print (theta)
#age=int(input('age'))
#sex=int(input('sex'))
#bmi=float(input('bmi'))
#children=int(input('children'))
#smoker=int(input('smoker'))
#region=int(input('region'))
#values=np.array([[1,age,sex,bmi,children,smoker]])
#charge=np.dot(values,np.transpose(theta))
#print (charge)


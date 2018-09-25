# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:53:48 2018

@author: Yi Tai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#隨機建值，樣本數100
trainData = np.linspace(0,10,100)+np.random.rand(100)
trainData = trainData.reshape(50,2)
trainData = pd.DataFrame(trainData)
print(trainData)
trainData_x = trainData[0]
trainData_y = trainData[1]

#----------------------

mu = trainData_x.mean()
sigma = trainData_x.std()
def s(x):
    return (x - mu) / sigma

trainData_z = s(trainData_x)
print(trainData_z)

#----------------------

theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
    return theta0 + theta1 * x
def cost(x,y):
    return 0.5*np.sum((y-f(x) )**2)


ETA = 1e-3

diff=1

count=0

error = cost(trainData_z,trainData_y)

#計算最佳解
while diff > 1e-2:
    tmp_theta0= theta0 - ETA*np.sum((f(trainData_z)-trainData_y))
    tmp_theta1= theta1 - ETA*np.sum((f(trainData_z)-trainData_y)*trainData_z)
    theta0 = tmp_theta0
    theta1 = tmp_theta1

    current_error = cost(trainData_z,trainData_y)
    diff=error - current_error
    error=current_error
    count +=1
    log = '{}次:theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count,theta0,theta1,diff))
    x=np.linspace(-2,2,100)
    plt.scatter(trainData_z,trainData_y,label='Training Data')
    plt.plot(x,f(x),'r-',label='GD')
    plt.legend(loc='upper left')
    plt.show()






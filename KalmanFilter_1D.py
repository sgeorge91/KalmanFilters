# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:37:26 2023

@author: https://teyvonia.com/kalman-filter-1d-localization-python/ ; SVG
"""
import random
import matplotlib.pyplot as plt
import numpy as np
Z=np.zeros(100)
for i in range(1,100):
    Z[i]=(1*i*.2)+np.random.normal(0,.5)
truth=np.zeros(100)
for i in range(1,100):
    truth[i]=(1*i*.2)
def initialize():
    x = 0 
    p = 0.5
    return x, p

def predict(x, p):
    # Prediction 
    dt=.2
    v=1.#+np.random.normal(0,.01)
    
    x = x + dt*v+np.random.normal(0,.1) # State Transition Equation (Dynamic Model or Prediction Model) 
    p_v=0
    q=.1                 
    p = p + (dt**2 * p_v) + q       # Predicted Covariance equation
    return x, p

def measure(i):
    z = Z[i]
    return z

def update(x, p, z):
    r=.5
    k = p / ( p + r)                # Kalman Gain
    x = x + k * (z - x)             # State Update
    p = (1 - k) * p                 # Covariance Update
    return x, p

def runKalmanFilter():
    x, p = initialize()
    n=100
    ps=[]
    xs=[]
    xs.append(x)
    ps.append(p)
    xapr=[]
    xapr.append(x)
    papr=[]
    papr.append(p)
    for j in range(1, n):
        x, p = predict(x, p)
        xapr.append(x)
        papr.append(p)
        z = measure(j)
        x, p = update(x, p, z)
        xs.append(x)
        ps.append(p)
    return(xs,ps,xapr,papr)

xs,ps,xapr,papr=runKalmanFilter()
times=np.arange(100)
times=times*.2
plt.xlim([0,5])
plt.ylim([0,5])
plt.plot(times[0:25],xapr[0:25], label='Apr estimate')
plt.plot(times[0:25],xs[0:25], label='Apos estimate')
plt.plot(times[0:25],Z[0:25], label= 'Measurement')
plt.scatter(times[0:25],truth[0:25], label='Ground truth')
plt.xlabel('Time (s)')
plt.ylabel('x(t)')
plt.legend()
plt.show()
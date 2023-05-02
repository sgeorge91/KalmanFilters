# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:25:35 2023

@author: SVG
"""

#The Ensemble Kalman Filter For the Lorenz System
import numpy as np
import random
import matplotlib.pyplot as plt
def funct1(x,y,z):
    sig=10.
    xd=sig*(y-x)
    return xd
def funct2(x,y,z):
    rho=28.
    yd= x*(rho-z)-y
    return yd
def funct3(x,y,z):
    beta=8./3.
    zd=x*y-(beta*z)
    return zd

def OneStep(ar, n):
    x0=ar[0]
    y0=ar[1]
    z0=ar[2]
    h=0.05
    k11=funct1(x0,y0,z0)
    k12=funct2(x0,y0,z0)
    k13=funct3(x0,y0,z0)
   
    k21=funct1(x0+(.5*h*k11),y0+(k12*.5*h),z0+(k13*.5*h))
    k22=funct2(x0+(.5*h*k11),y0+(k12*.5*h),z0+(k13*.5*h))
    k23=funct3(x0+(.5*h*k11),y0+(k12*.5*h),z0+(k13*.5*h))
    
    k31=funct1(x0+(.5*h*k21),y0+(k22*.5*h),z0+(k23*.5*h))
    k32=funct2(x0+(.5*h*k21),y0+(k22*.5*h),z0+(k23*.5*h))
    k33=funct3(x0+(.5*h*k21),y0+(k22*.5*h),z0+(k23*.5*h))
    
    k41=funct1(x0+(h*k31),y0+(k32*h),z0+(k33*h))
    k42=funct2(x0+(h*k31),y0+(k32*h),z0+(k33*h))
    k43=funct3(x0+(h*k31),y0+(k32*h),z0+(k33*h))
    
    x=x0+((k11+(2*(k21+k31))+k41)*h/6.0)
    y=y0+((k12+(2*(k22+k32))+k42)*h/6.0)
    z=z0+((k13+(2*(k23+k33))+k43)*h/6.0)
    return(x,y,z)
N=500
tl=150
ini=np.array([1,1,10])
obs=np.zeros((tl,3))
obs[0][0],obs[0][1],obs[0][2]=OneStep(ini,1)
H=np.array([[1,0,0],[0,1,0],[0,0,1]])
#Observations are generated
for i in range (1,tl):
    obs[i][0],obs[i][1],obs[i][2]=OneStep([obs[i-1][0],obs[i-1][1],obs[i-1][2]],.5)
for i in range(0,tl):
    obs[i][0]=obs[i][0]+random.uniform(-1*5,5)
    obs[i][1]=obs[i][1]+random.uniform(-1*5,5)
    obs[i][2]=obs[i][2]+random.uniform(-1*5,5)
###True state
truth=np.zeros((tl,3))
truth[0][0],truth[0][1],truth[0][2]=OneStep(ini,0)
for i in range (1,tl):
    truth[i][0],truth[i][1],truth[i][2]=OneStep([truth[i-1][0],truth[i-1][1],truth[i-1][2]],0)
devs=np.zeros((tl,3))
for i in range(0,tl):
    devs[i]=(obs[i]-truth[i])#np.outer(obs[i],truth[i]))#(
R=np.matmul(devs.T,devs)/(tl-1)#np.mean(devs, axis=0)
#temp_0= np.transpose(obs)-np.matmul(H,np.transpose(truth))
#R=np.matmul(np.transpose(temp_0),temp_0)/(N-1)
###Ensemble is initialized
en=np.zeros((N,3))
#for j in range(0,tl):
stup=[]
for i in range (0,N):
    ini=np.array([1,1,10])
    stup.append([random.uniform(-.1,.1),random.uniform(-.1,.1),random.uniform(-.1,.1)])
    ini=ini+stup[i]
    en[i][0],en[i][1],en[i][2]=np.array(OneStep(ini,.2))
m=np.zeros(3)
v=np.zeros(3)
en_d=np.zeros((N,3))
inn=np.zeros((N,3))
x=[]
temp=0
temp2=[]
fs=np.zeros((tl,3))
for t in range(0,tl):    
    for j in range (0,3):
        m[j]=np.mean(en[j])
        v[j]=np.var(en[j])

    for j in range (0,N):
        en_d[j]=((en[j]-m))#**2)**.5
    #Innovation vectors
    for i in range(0,N):            
        inn[i]=obs[t]-np.matmul(H,en[i])
    
    inn_cov=[]#np.zeros((N,3))
    #Innovation covariance
    for j in range (0,N):
        inn_cov.append(np.outer(inn[i],inn[i].T))
    
    P=np.matmul(en_d.T,en_d)/(N-1)
    
    #Calculating the Kalman gain
    inv=np.matmul(P,H.T)
    inv=np.matmul(H,inv)
    inv=inv+R
    inv=np.linalg.inv(inv)
    kp=np.matmul(H.T,inv)
    K=np.matmul(P,kp)
    
    for i in range (0,N):
        #temp=obs[t]-np.matmul(H,en[i])
        temp=en[i]+np.matmul(K,inn[i])
        temp2.append(temp)
        ini=np.array([temp[0],temp[1],temp[2]])
        ini=ini+np.array([random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)])
        en[i][0],en[i][1],en[i][2]=np.array(OneStep(ini,.1))
        en[i][0]=en[i][0]+random.uniform(-.1,.1)
        en[i][1]=en[i][1]+random.uniform(-.1,.1)
        en[i][2]=en[i][2]+random.uniform(-.1,.1)
    
   
    x.append(np.mean(temp2,axis=0))
    temp2=[]
    fs[t]=np.mean(en, axis=0)
times=np.arange(tl)
times=times*.01
os=[i[0] for i in obs]
xs=[i[0] for i in x]
fss=[i[0] for i in fs]
txs=[i[0] for i in truth]
plt.plot(times, os , label ='Measurement')
plt.plot(times,xs, label= 'Analysis estimate')
plt.plot(times, fss, label ='Forecast estimate')
plt.scatter(times, txs, label='Ground truth')
plt.xlabel('Time (s)')
plt.ylabel('x(t)')
plt.legend()
plt.show()

"""$K_n = P_n^-H^T(H_nP_n^-H_n^T + R_n)^{-1}$

where:

    $K_n$ is the updated Kalman gain matrix at time step $n$
    $P_n^-$ is the forecast error covariance matrix at time step $n$
    $H_n$ is the observational operator matrix at time step $n$
    $R_n$ is the observational error covariance matrix at time step $n$"""

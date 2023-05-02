# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:25:35 2023

@author: SVG
"""

#The Ensemble Kalman Filter For the Lorenz System
import numpy as np
import random
import matplotlib.pyplot as plt
def embedding(x,d=2,tau=0):
    """
    Returns embedded vectors in dimension d from a time series.
    Parameter
    ---------
    x : 1-d numpy array
        Input time series for which embedding is done
    d : int
        Dimension of the embedding space. Default, 2
    tau : int
	Time delay for the embedding.
    Returns
    -------
    dvec : n-d numpy array
	Array of embedded vectors
    """     
    x=np.array(x)
    if(tau==0):
        mu=np.mean(x)
        sig2=np.var(x)
        xn=x-mu
        acf=np.correlate(xn,xn,'full')[len(xn)-1:]
        acf=acf/sig2/len(xn)
        tau=np.where(acf<(1./np.exp(1)))[0][0]
    n=int(len(x)-d*tau)
    dvec=np.zeros((n,d))
    for i in range(0,len(dvec)):
        for j in range(0,d):
            dvec[i][j]=x[i+j*tau]
    
    return(dvec)
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
def TakenUpdate(ini,t0):
    skip=100
    ind=list(np.arange(0,max(t0-skip,0)))
    ind.extend(np.arange(min(t0+skip,tl),tl))
    dis=[]
    dis=[np.mean(abs(obs[i]-ini)) for i in ind]
    disi=np.argsort(dis)
    mx=np.mean(obs[disi[0:20]],axis=0)
    return(mx[0],ini[0],ini[1])
    
    
    
    
N=50
tl=1000
ini=np.array([1,1,10])
obs1=np.zeros((tl+21,3))
obs1[0][0],obs1[0][1],obs1[0][2]=OneStep(ini,1)
H=np.array([[1,0,0],[0,1,0],[0,0,1]])
#Observations are generated
for i in range (1,tl+21):
    obs1[i][0],obs1[i][1],obs1[i][2]=OneStep([obs1[i-1][0],obs1[i-1][1],obs1[i-1][2]],.5)
for i in range(0,tl+21):
    obs1[i][0]=obs1[i][0]+random.uniform(-1*7,7)
    obs1[i][1]=obs1[i][1]+random.uniform(-1*7,7)
    obs1[i][2]=obs1[i][2]+random.uniform(-1*7,7)
tobs=[i[0] for i in obs1]
obs=embedding(tobs,3,1)

###True state
truth=np.zeros((tl+27,3))
truth[0][0],truth[0][1],truth[0][2]=OneStep(ini,0)
for i in range (1,tl+27):
    truth[i][0],truth[i][1],truth[i][2]=OneStep([truth[i-1][0],truth[i-1][1],truth[i-1][2]],0)
tobs=[i[0] for i in truth]
truth=embedding(tobs,3,1)
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
    en[i][0],en[i][1],en[i][2]=np.array(TakenUpdate(ini,0))
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
        en[i][0],en[i][1],en[i][2]=np.array(TakenUpdate(ini,t))
        en[i][0]=en[i][0]+random.uniform(-1,1)
        en[i][1]=en[i][1]+random.uniform(-1,1)
        en[i][2]=en[i][2]+random.uniform(-1,1)
    
   
    x.append(np.mean(temp2,axis=0))
    temp2=[]
    fs[t]=np.mean(en, axis=0)
    print(t)    
times=np.arange(tl)
times=times*.01
os=[i[0] for i in obs]
xs=[i[0] for i in x]
fss=[i[0] for i in fs]
txs=[i[0] for i in truth]
plt.plot(times[800:999], os[800:999] , label ='Measurement')
plt.plot(times[800:999],xs[801:1000], label= 'Analysis estimate')
plt.plot(times[800:999], fss[800:999], label ='Forecast estimate')
plt.scatter(times[800:999], txs[800:999], label='Ground truth')
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
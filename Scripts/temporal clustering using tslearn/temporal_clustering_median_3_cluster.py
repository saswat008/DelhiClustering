#!/usr/bin/env python
# coding: utf-8

# In[38]:


import tslearn
import numpy as np
import pandas as pd


# In[39]:


df_input = pd.read_csv('dataset_median.csv',header=None)
np_input = df_input.values
print(np_input.shape)


# In[40]:


X_train = np_input[:,1:]
print(X_train.shape)


# In[41]:


from scipy.spatial.distance import cdist 
from tslearn.clustering import TimeSeriesKMeans
distortions = [] 
inertias = [] 
mapping1 = {}   # Used for distortion calculation, didnt use
mapping2 = {} 
K = range(1,10) 
  
for k in K: 
    #Building and fitting the model 
#     kmeanModel = KMeans(n_clusters=k).fit(X) 
    km = TimeSeriesKMeans(n_clusters=k, metric="dtw",max_iter = 900,tol = 1e-08)
    km.fit(X_train)     
      
    inertias.append(km.inertia_) 
    predictions = km.predict(X_train)
  
    print(km.inertia_)
    mapping2[k] = km.inertia_ 
    for c in range(k):
        c_0 = np.argwhere(predictions==c)
        print(c_0.shape[0],end=' ')
    print('----------------------------------------')


# In[42]:


import matplotlib.pyplot as plt 
plt.plot(K, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show() 


# In[47]:


from tslearn.clustering import TimeSeriesKMeans
km = TimeSeriesKMeans(n_clusters=3, metric="dtw",max_iter = 900,tol = 1e-08,random_state=3)
km.fit(X_train)


# In[48]:


predictions = km.predict(X_train)
for c in range(3):
    c_0 = np.argwhere(predictions==c)
    print(c_0.shape[0],end=' ')
c_assign = np.zeros(32)
for k in range(3):
    c_0 = np.argwhere(predictions==k)
    c_assign[c_0] = k
#     print(k,c_0)
print(c_assign)


# In[49]:


x = [ i for i in range(204)]

colors = ['green','red','blue','yellow','purple','cyan']
K = 3
for k in range(K):
    cluster_avg = 0
    size =0
    c_k_stations = np.argwhere(c_assign == k).flatten()
    cluster_trend = np.zeros((204,))
#     print(cluster_trend.shape)
    for station in c_k_stations:
        
        temporal_trend_max  = X_train[station]
        y = temporal_trend_max
#         print(y.shape)
        cluster_trend  = cluster_trend + y
        cluster_avg += np.sum(y)
        print(c_k_stations,np.sum(y)/204)
        size +=1
#         if(k in [0,1]):
#         plt.plot(x,y,color = colors[k])
    # Single trend per cluster
    cluster_trend = cluster_trend / size
#     if(k==0):
    plt.plot(x,cluster_trend,color = colors[k],label = "Cluster "+str(k+1))
    print("cluster ",k,"-->",cluster_avg/(size*204))
    
plt.legend(loc="upper left")
plt.show()

    


# In[ ]:





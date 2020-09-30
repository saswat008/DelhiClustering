#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tslearn
import numpy as np
import pandas as pd


# In[2]:


df_input = pd.read_csv('dataset_median_without_outliers.csv',header=None)
np_input = df_input.values
print(np_input.shape)


# In[3]:


X_train = np_input[:,1:]
print(X_train.shape)


# In[4]:


# from scipy.spatial.distance import cdist 
# from tslearn.clustering import TimeSeriesKMeans
# distortions = [] 
# inertias = [] 
# mapping1 = {}   # Used for distortion calculation, didnt use
# mapping2 = {} 
# K = range(1,10) 
  
# for k in K: 
#     #Building and fitting the model 
# #     kmeanModel = KMeans(n_clusters=k).fit(X) 
#     km = TimeSeriesKMeans(n_clusters=k, metric="dtw",max_iter = 900,tol = 1e-08)
#     km.fit(X_train)     
      
#     inertias.append(km.inertia_) 
#     predictions = km.predict(X_train)
  
#     print(km.inertia_)
#     mapping2[k] = km.inertia_ 
#     for c in range(k):
#         c_0 = np.argwhere(predictions==c)
#         print(c_0.shape[0],end=' ')
#     print('----------------------------------------')


# In[5]:


# import matplotlib.pyplot as plt 
# plt.plot(K, inertias, 'bx-') 
# plt.xlabel('Values of K') 
# plt.ylabel('Inertia') 
# plt.title('The Elbow Method using Inertia') 
# plt.show() 


# In[6]:


# from scipy.spatial.distance import cdist 
from tslearn.clustering import TimeSeriesKMeans
km = TimeSeriesKMeans(n_clusters=3, metric="dtw",max_iter = 900,tol = 1e-08,random_state=3)
km.fit(X_train)


# In[7]:


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


# In[8]:


import matplotlib.pyplot as plt 
x = [ i for i in range(204)]

colors = ['green','blue','red','yellow','purple','cyan']
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
y_label = " PM values :(unit - Âµg/mÂ³)"
x_label =  " Sequence of days from Aug 2019 to Aug 2020 "
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.show()

    


# In[9]:


#----------------------- Cluster Assignment file ------------------------------
data_dir = '../../Data/'
import pandas as pd
df_lat_lon=pd.read_csv(data_dir+'location_name_lat_lon.csv', sep=',')
df_lat_lon


# In[10]:


df_input = pd.read_csv('dataset_median_without_outliers.csv',header=None)
np_input = df_input.values
print(np_input.shape)
print(np_input[:,0])


# In[11]:


clusters = []
lat = []
long = []
location = []
for i in range(np_input.shape[0]):
    loc = np_input[i,0]
    for j in range(df_lat_lon.shape[0]):
        if(df_lat_lon['location'][j]==loc):
            location.append(loc)
            lat.append( df_lat_lon['latitude'][j])
            long.append(df_lat_lon['longitude'][j])
            print(loc,df_lat_lon['latitude'][j])
    clusters.append(c_assign[i])
    
#     df['Cluster'][loc] = c_assign[i]
print(len(clusters))
# print(long)
    


# In[11]:


new_df = pd.DataFrame() 
new_df["location"] = location
new_df["latitude"] = lat
new_df["longitude"] = long
new_df["cluster"] = clusters
print(new_df)


# In[12]:


new_df.to_csv("locations_with_cluster_assignments_3Clusters.csv")


# In[ ]:





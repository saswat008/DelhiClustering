#!/usr/bin/env python
# coding: utf-8

# In[149]:


import tslearn
import numpy as np
import pandas as pd


# In[150]:


df_input = pd.read_csv('dataset.csv',header=None)
np_input = df_input.values
print(np_input[0][:].shape)


# In[151]:


from tslearn.utils import to_time_series_dataset
number_of_stations= np_input.shape[0]
number_of_timesteps =  (int)((np_input.shape[1]-1 )/2)
number_of_features = 2 # min max
X_train = np.zeros(( number_of_stations, number_of_timesteps,  number_of_features  ))
for i in range(number_of_stations):     #np_input.shape[0]):
    
    time_steps = (int)((np_input.shape[1]-1 )/2)
#     print(time_steps)
    ts = np.zeros(( time_steps , 2 ))
    for j in range(1,np_input.shape[1],2):
        [mi,ma] = [ np_input[i][j],np_input[i][j+1] ]
        ts[(int)((j-1)/2)] = [mi,ma]
#     print(ts.shape)
    X_train[i] = ts
    
print(X_train.shape)


# In[152]:


from tslearn.clustering import TimeSeriesKMeans
km = TimeSeriesKMeans(n_clusters=6, metric="dtw",max_iter = 900,tol = 1e-08,random_state= 12)
km.fit(X_train)


# In[153]:


predictions = km.predict(X_train)
c_assign = np.zeros(32)
for k in range(6):
    c_0 = np.argwhere(predictions==k)
    c_assign[c_0] = k
#     print(k,c_0)
print(c_assign)
    


# In[154]:


# Assign cluster assignments
print(X_train.shape)


# In[155]:


five_station_max =  X_train[0].T[0]
# two_station_min_max = X_train[18,29]
print(five_station_max.shape)
print(five_station_max[:5])


# In[156]:



# one_station_min_max.reshape((1,204,2))

# np.reshape(one_station_min_max,(1,204,2) )

#one_station_max = one_station_min_max.T[0]
#x = [ i for i in range(204)]
## print(x.s)
#y = five_station_max
#from matplotlib import pyplot as plt
#plt.plot(x,y)


# In[148]:
from matplotlib import pyplot as plt

x = [ i for i in range(204)]
# for station in range(32):
    # ALl stations
#     temporal_trend_max  = X_train[station].T[0]
#     x = [ i for i in range(204)]
# # print(x.s)
#     y = temporal_trend_max
#     plt.plot(x,y)
    # CLuster Zero stations
c_0_stations = np.argwhere(c_assign == 0).flatten()
for station in c_0_stations:
    print(c_0_stations)
    temporal_trend_max  = X_train[station].T[0]
    y = temporal_trend_max
    #plt.plot(x,y,color = 'green')
    
c_1_stations = np.argwhere(c_assign == 1).flatten()
for station in c_1_stations:
    print(c_1_stations)
    temporal_trend_max  = X_train[station].T[0]
    y = temporal_trend_max
    #plt.plot(x,y,color = 'red')
c_2_stations = np.argwhere(c_assign == 2).flatten()
for station in c_2_stations:
    print(c_2_stations)
    temporal_trend_max  = X_train[station].T[0]
    y = temporal_trend_max
    #plt.plot(x,y,color = 'blue')
    
c_3_stations = np.argwhere(c_assign == 3).flatten()
for station in c_3_stations:
    print(c_3_stations)
    temporal_trend_max  = X_train[station].T[0]
    y = temporal_trend_max
	
    plt.plot(x,y,label = station)
    
c_4_stations = np.argwhere(c_assign == 4).flatten()
for station in c_4_stations:
    print(c_4_stations)
    temporal_trend_max  = X_train[station].T[0]
    y = temporal_trend_max
    #plt.plot(x,y,color = 'purple')
    
c_5_stations = np.argwhere(c_assign == 5).flatten()
for station in c_5_stations:
    print(c_5_stations)
    temporal_trend_max  = X_train[station].T[0]
    y = temporal_trend_max
    #plt.plot(x,y,color = 'cyan')
    
plt.legend(loc="upper left")
plt.show()


# In[ ]:





# In[ ]:





# In[101]:


# #-------------------------FOR CARTO PLOT--------------------------------------------------------
# data_dir = '../../Data/'
# import pandas as pd
# df=pd.read_csv(data_dir+'location_name_lat_lon.csv', sep=',')
# df


# In[82]:


# df_input = pd.read_csv('dataset.csv',header=None)
# np_input = df_input.values
# print(np_input.shape)
# print(np_input[:,0])


# In[104]:


# clusters = []
# lat = []
# long = []
# location = []
# for i in range(np_input.shape[0]):
#     loc = np_input[i,0]
#     for j in range(df.shape[0]):
#         if(df['location'][j]==loc):
#             location.append(loc)
#             lat.append( df['latitude'][j])
#             long.append(df['longitude'][j])
#             print(loc,df['latitude'][j])
#     clusters.append(c_assign[i])
    
# #     df['Cluster'][loc] = c_assign[i]
# print(len(clusters))
# # print(long)
    


# In[91]:


# df_input['Cluster'] = clusters


# In[93]:


# df_input.shape


# In[107]:


# new_df = pd.DataFrame() 


# In[110]:


# new_df["location"] = location
# new_df["latitude"] = lat
# new_df["longitude"] = long
# new_df["cluster"] = clusters


# In[111]:


# print(new_df)


# In[112]:


# new_df.to_csv("locations_with_cluster_assignments_6Ckusters.csv")


# In[ ]:





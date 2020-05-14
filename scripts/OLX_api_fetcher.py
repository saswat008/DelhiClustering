#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import numpy as np

columns = 11
rows = 11
slongitude = 76.846585
slatitude = 28.866605
height = 0.009043717*5
lat = slatitude
lon = slongitude
search_keyword = "houses and aprtments"

Title=[]
Latitude=[]
Longitude=[]
Id=[]

for i in range(rows):
    for j in range(columns):
        
        for page_no in range(50):
            try:
                
                payload = {'facet_limit': 100,'lang':'en', 'latitude': lat,'location_facet_limit':20,                           'longitude':lon,'page':page_no,'query':search_keyword}
                response = requests.get('https://www.olx.in/api/relevance/search',params=payload,headers={'Cache-Control': 'no-cache'})
                print(response.url)
                jsonResponse = response.json()
                if(jsonResponse['empty']==False):
                    for ad in jsonResponse['data']:
                        location = ad['locations']
                        ids=ad['id']
                        if ids not in Id:
                            Id.append(ad['id'])
                            Latitude.append(location[0]['lat'])
                            Longitude.append(location[0]['lon'])
                            Title.append(ad['title'])
                else:
                    break
            except:
                pass
        
        print(i,j,"completed",lat,lon,len(Id))
        lon += (5/(111.32*np.cos(lat*(np.pi/180))))
    
    lat -= height
    lon = slongitude


import pandas as pd
df = pd.DataFrame({'Id':Id,'Title':Title,'Latitude':Latitude,'Longitude':Longitude}) 
df.to_csv('olx_ads.csv', index=False, encoding='utf-8')
print("File stored successfully.")


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 23:36:08 2023

@author: talam
"""

import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


for dataset in [ '2018-19','2019-20','2020-21']:
    
    
    if dataset=='2018-19':
        title_1='2018-19'
        
        
    if dataset=='2019-20':
         
         title_1='2019-20'    
        
    if dataset=='2020-21':
         
                
          title_1='2020-21'
          
          
            

    #map_dict=dict(df2)
    
    df = pd.read_csv('./csvs/Master {} with Labels FIPS.csv'.format(title_1))
     
    
    
    fig = go.Figure()
    
    fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', 
                        color="Labels",
                        size="STUDENT_COUNT_ALL_Attendance", color_continuous_scale=px.colors.sequential.Viridis, size_max=40,
                        zoom=5, height=1000, mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    
  
    fig.update_geos(fitbounds="locations")
    
    fig.write_image('./images/Cities_Labels_{}.png'.format(title_1))
    
    fig = go.Figure()
    
    fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', 
                        color="SAT Combined Score",
                        size="STUDENT_COUNT_ALL_Attendance", color_continuous_scale=px.colors.sequential.Viridis, size_max=40,
                        zoom=5, height=1000, range_color=[700,1600],mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    
  
    fig.update_geos(fitbounds="locations")
    
    fig.write_image('./images/Cities_SAT_{}.png'.format(title_1))
    
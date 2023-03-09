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
    
    
    
    
    fig.add_trace(go.Scattergeo(
        #locationmode = df['SCHOOL_DSTRCT_NM'] ,
        lon = df['Longitude'],
        lat = df['Latitude'],
        text = df['SCHOOL_DSTRCT_NM'],
        marker = dict(
            size = df['STUDENT_COUNT_ALL_Attendance']/100,
            color = 'paleturquoise',
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area',
            ),
        name = 'test'))

    fig.update_layout(
            title_text = '2014 US city populations<br>(Click legend to toggle traces)',
            showlegend = True,
            geo = dict(
                scope = 'usa',
                landcolor = 'rgb(217, 217, 217)',
            )
        )
    fig.update_geos(fitbounds="locations")
    
    fig.write_image('./images/Cities_{}.png'.format(title_1))
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:31:37 2023

@author: talam
"""
import json
import pandas as pd
import numpy as np
import descartes
import matplotlib as plt
import geopandas as gpd
import plotly_express as px
import requests
import seaborn as sns


# ga_json='./data/5m-US-counties.json'
# with open(ga_json,'r') as f:
#     data=json.load(f.text)
    
r = requests.get(
    'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
    )
counties = json.loads(r.text)
target_states = ["13"]
al = [i for i in counties["features"] if i["properties"]["STATE"] in target_states]
df2=pd.json_normalize(al, max_level=1)
df2=df2.set_index('properties.NAME')
df2=df2['id']


for dataset in [ '2018-19','2019-20','2020-21']:
    
    
    if dataset=='2018-19':
        title_1='2018-19'
        
        
    if dataset=='2019-20':
         
         title_1='2019-20'    
        
    if dataset=='2020-21':
         
                
          title_1='2020-21'
          
          
            

    map_dict=dict(df2)
    
    df = pd.read_csv('./csvs/{}_with labels.csv'.format(title_1))
    df['SCHOOL_DSTRCT_NM']=df['SCHOOL_DSTRCT_NM'].replace({' County':''},regex=True)
    df['SCHOOL_DSTRCT_NM']=df['SCHOOL_DSTRCT_NM'].replace({' Public Schools':''},regex=True)
    df['FIPS']=df.SCHOOL_DSTRCT_NM.map(map_dict)





    fig = px.choropleth(df, geojson=counties, locations='FIPS', color='Labels',
                        color_continuous_scale='Viridis',
                        range_color=(0, 10),
                        scope='usa',
                        labels={'Labels': 'Overall'}
                        )
    fig.update_geos(fitbounds='locations', visible=False)
    fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      title={'text': '{} School Year'.format(title_1),  'y':0.9, 'x':0.5,'xanchor': 'center','yanchor': 'top' })
    fig.write_image('./images/Chloropeth_{}_Overall.png'.format(title_1))
    
    fig = px.choropleth(df, geojson=counties, locations='FIPS', color='{} SAT DSTRCT_AVG_SCORE_VAL'.format(title_1),
                        color_continuous_scale='Viridis',
                        range_color=(0, 30),
                        scope='usa',
                        labels={'Labels': 'SAT Combined Score'}
                        )
    fig.update_geos(fitbounds='locations', visible=False)
    fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      title={'text': '{} School Year'.format(title_1),  'y':0.9, 'x':0.5,'xanchor': 'center','yanchor': 'top' })
    fig.write_image('./images/Chloropeth_{}_SAT.png'.format(title_1))
    
    fig = px.choropleth(df, geojson=counties, locations='FIPS', color='{}_Instruction_y'.format(title_1),
                        color_continuous_scale='Viridis',
                        range_color=(0, 30),
                        scope='usa',
                        labels={'Labels': 'Instruction Expenditures'}
                        )
    fig.update_geos(fitbounds='locations', visible=False)
    fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      title={'text': '{} School Year'.format(title_1),  'y':0.9, 'x':0.5,'xanchor': 'center','yanchor': 'top' })
    fig.write_image('./images/Chloropeth_{}_InstructionExp.png'.format(title_1))
    
    
    
    heatmap = sns.heatmap(df.iloc[:,2:-2].corr(), vmin=-1, vmax=1, annot=True,annot_kws={'size':8}, cmap='viridis')
    
    plt.rcParams['figure.figsize']=(20,20)
    heatmap.set_title('Correlation {}'.format(title_1), fontdict={'fontsize':18}, pad=12)
    hm=heatmap.get_figure()
    hm.savefig('./images/heatmap_{}.png'.format(title_1), dpi=300, bbox_inches='tight')
    hm.clear()
    
    
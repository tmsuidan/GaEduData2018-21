#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:13:30 2019

@author: tmsuidan
"""


import numpy as np
import pandas as pd




#from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans




import os
directory="./images/km"
if not os.path.exists(directory):
    os.makedirs(directory)
# directory="./images/em"
# if not os.path.exists(directory):
#     os.makedirs(directory)

# directory="./csvs/em"
# if not os.path.exists(directory):
#     os.makedirs(directory)
directory="./csvs/kmeans"
if not os.path.exists(directory):
    os.makedirs(directory)
for dataset in [ '2018-19','2019-20','2020-21']:
    
    np.random.seed(123)
    if dataset=='2018-19':
        
        df = pd.read_csv('./data/Master_18-19.csv')
        df.replace('TFS',0,inplace=True)
        df.replace(np.nan,0,inplace=True)
        df=df.iloc[:,2:]
       
        features=df.values
        title_1='2018-19'
        n_clust=10
        
        
        
        
        
        
        
        
        
        #https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
        
        # Normalize feature
       
        
    if dataset=='2019-20':
        
        df = pd.read_csv('./data/Master_19-20.csv')
        df.replace('TFS',0,inplace=True)
        df.replace(np.nan,0,inplace=True)
        df=df.iloc[:,2:]
       
        features=df.values
        title_1='2019-20' 
        n_clust=10
        
        
    if dataset=='2020-21':
         
         df = pd.read_csv('./data/Master_20-21.csv')
         df.replace('TFS',0,inplace=True)
         df.replace(np.nan,0,inplace=True)
         df=df.iloc[:,2:]
        
         features=df.values
         title_1='2020-21'
         n_clust=10
         
    
         
    
    
    
        
        
    km = KMeans(
        n_clusters=n_clust, init='k-means++',
        n_init=50, max_iter=300,
        tol=1e-04, random_state=123
    )
    
    km.fit_predict(features,y=None,sample_weight=None)
    labels_df=pd.DataFrame(km.labels_)
    labels_df.to_csv('./csvs/Labels_{}.csv'.format(title_1),index=False)
    
for dataset in [ '2018-19','2019-20','2020-21']:
    
    
    if dataset=='2018-19':
        
        df = pd.read_csv('./data/Master_18-19.csv')
        
        df.replace('TFS',0,inplace=True)
        df.replace(np.nan,0,inplace=True)
        title_1='2018-19'
        df2=pd.read_csv('./csvs/Labels_{}.csv'.format(title_1))
        df['Labels']=df2['0']
        df.to_csv('./csvs/{}_with labels.csv'.format(title_1),index=False)
        
        
        
        
        
        
        
        
        
        
        #https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
        
        # Normalize feature
       
        
    if dataset=='2019-20':
        
        df = pd.read_csv('./data/Master_19-20.csv')
        df.replace('TFS',0,inplace=True)
        df.replace(np.nan,0,inplace=True)
        
        title_1='2019-20' 
        df2=pd.read_csv('./csvs/Labels_{}.csv'.format(title_1))
        df['Labels']=df2['0']
        df.to_csv('./csvs/{}_with labels.csv'.format(title_1),index=False)
        
        
        
    if dataset=='2020-21':
         
          df = pd.read_csv('./data/Master_20-21.csv')
          df.replace('TFS',0,inplace=True)
          df.replace(np.nan,0,inplace=True)
          
          title_1='2020-21'
          df2=pd.read_csv('./csvs/Labels_{}.csv'.format(title_1))
          df['Labels']=df2['0']
          df.to_csv('./csvs/{}_with labels.csv'.format(title_1),index=False)
          
            
    
    
    
    
    
    
   
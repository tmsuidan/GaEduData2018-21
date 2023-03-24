#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:13:30 2019

@author: tmsuidan
"""


import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score



#from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance



import matplotlib.pyplot as plt



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
        
        df = pd.read_csv('./data/Master_2018-19 after _select.csv')
        df.replace('TFS',0,inplace=True)
        df.replace(np.nan,0,inplace=True)
        df=df.iloc[:,2:]
       
        features=df.values
        title_1='2018-19'
        
        
        
        
        
        
        
        
        
        #https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
        
        # Normalize feature
       
        
    if dataset=='2019-20':
        
        df = pd.read_csv('./data/Master_2019-20 after _select.csv')
        df.replace('TFS',0,inplace=True)
        df.replace(np.nan,0,inplace=True)
        df=df.iloc[:,2:]
       
        features=df.values
        title_1='2019-20' 
        
        
    if dataset=='2020-21':
         
         df = pd.read_csv('./data/Master_2020-21 after _select.csv')
         df.replace('TFS',0,inplace=True)
         df.replace(np.nan,0,inplace=True)
         df=df.iloc[:,2:]
        
         features=df.values
         title_1='2020-21'
         
    
         
    distortions = []
    silhouette_scores=[]
    
    centers=[]
    #mi_list=[]
    
    n_comp=[]
    labels=[]
    iter_list=[]
    jsl=[]
    
    clusters=[]
    

    for i in range(2, 30):
        
        
        km = KMeans(
            n_clusters=i, init='k-means++',
            n_init=50, max_iter=300,
            tol=1e-04, random_state=123, algorithm='auto'
        )
        
        km.fit_predict(features,y=None,sample_weight=None)
        labels.append(km.labels_)
        
        distortions.append(km.inertia_)
        n_comp.append(i)
        
        centers.append(km.cluster_centers_)
        iter_list.append(km.n_iter_)
        
        #pd.DataFrame(clusters).to_csv('./csvs/kmeans/{}_clusters_{}.csv'.format(title_1,  i ))
        
        
        
        
        silhouette_scores.append(silhouette_score(features, km.labels_))
        
        visualizer = InterclusterDistance(km,  min_size=1, \
                                  max_size=features.shape[0], \
                                  embedding='mds', \
                                  scoring='membership',\
                                  legend=False, \
                                  random_state=123,\
                                  is_fitted=False)

        visualizer.fit(features)        # Fit the data to the visualizer
        visualizer.poof(outpath='./images/km/{}_km_yb_intclustd_{}.png'.format(title_1, i)) 
        plt.close()
    kmdf=pd.DataFrame({
                        'Number of Iterations':iter_list, \
                        'Number of Components':n_comp,'Distortion':distortions,\
                        'Silhouette Score': silhouette_scores, \
                        
                        })   
    labelsdf=pd.DataFrame(np.NAN, index=range(features.shape[0]),columns=range(2, features.shape[1]+1))
    for j in range(len(clusters)):
        labelsdf.iloc[:,j]=clusters[j]
    
    labelsdf.to_csv('./csvs/{}_kmeansclusterslabels.csv'.format(title_1))
    
    kmdf.to_csv('./csvs/{}_kmeans_{}.csv'.format(title_1,i))
    pd.DataFrame(centers).to_csv('./csvs/{}_kmcenters.csv'.format(title_1))
   
    
    model=KMeans(init='k-means++',
                    n_init=50, max_iter=300,
                    tol=1e-04, random_state=123, algorithm='auto')
    visualizer = KElbowVisualizer(model, metric='silhouette', k=np.arange(2,20), timings=False,locate_elbow=False)
    visualizer.fit(features)    
    visualizer.poof(outpath='./images/{}_km_yb_sil.png'.format(title_1)) 
    
    
   
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:13:30 2019

@author: tmsuidan
"""


import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score



from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance



import matplotlib.pyplot as plt
from time import perf_counter


import os
directory="./images/km"
if not os.path.exists(directory):
    os.makedirs(directory)
    directory="./images/em"
if not os.path.exists(directory):
    os.makedirs(directory)

directory="./csvs/em"
if not os.path.exists(directory):
    os.makedirs(directory)
directory="./csvs/kmeans"
if not os.path.exists(directory):
    os.makedirs(directory)
for dataset in [ '2018-19','2019-20','2020-21']:
    for fr_algo in ['nonscaled','scale']:
        #,'ica','rpg','rps','lda']:
        np.random.seed(123)
        if dataset=='2018-19':
            
            df = pd.read_csv('./data/Master_18-19.csv')
            df.replace('TFS',0,inplace=True)
            df.replace(np.nan,0,inplace=True)
            df=df.iloc[:,2:]
           
            features=df.values
            title_1='2018-19'
            
            
            
            
            
            
            
            
            
            #https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
            
            # Normalize feature
           
            
        if dataset=='2019-20':
            
            df = pd.read_csv('./data/Master_19-20.csv')
            df.replace('TFS',0,inplace=True)
            df.replace(np.nan,0,inplace=True)
            df=df.iloc[:,2:]
           
            features=df.values
            title_1='2019-20' 
            
            
        if dataset=='2020-21':
             
             df = pd.read_csv('./data/Master_20-21.csv')
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
        

        for i in range(2, features.shape[1]/2):
            
            
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
            
            pd.DataFrame(clusters).to_csv('./csvs/kmeans/{}_clusters_{}.csv'.format(title_1,  i ))
            
            
            
            
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
        visualizer = KElbowVisualizer(model, metric='silhouette', k=np.arange(2,features.shape[1]/2), locate_elbow=False)
        visualizer.fit(features)    
        visualizer.poof(outpath='./images/{}_km_yb_sil.png'.format(title_1)) 
        
        plt.close()
        model=KMeans(random_state=123)
        visualizer = KElbowVisualizer(model,  k=np.arange(2,features.shape[1]), locate_elbow=False)
        visualizer.fit(features)    
        visualizer.poof(outpath='./images/{}_km_yb_dist.png'.format(title_1)) 
        
        plt.close()
        model=KMeans(random_state=123)
        visualizer = KElbowVisualizer(model, metric='calinski_harabasz', k=np.arange(2,features.shape[1]), locate_elbow=False)
        visualizer.fit(features)    
        visualizer.poof(outpath='./images/{}_km_yb_ch.png'.format(title_1)) 
        plt.close()
        
            
        
        #adapted from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
        lowest_bic = np.infty
        bic_list= []
        
        comp_list=[]
        cv_list=[]
        
        
        weights=[]
        means=[] 
        aic_list=[]
        
        init_paramsl=[]
        iter_list=[]
        loglower=[]
        labels_list=[]
        
        n_components_range = range(2,features.shape[1]+1)
        cv_types = [ 'diag', 'full']
        for cv_type in cv_types:
            for ip in ['kmeans','random']:
                labels_list=[]
                
                for n_components in n_components_range:
                    
                
                # Fit a Gaussian mixture with EM
                    
                    em = GaussianMixture(n_components=n_components, max_iter=300, \
                                          n_init=30, covariance_type=cv_type, \
                                          random_state=123, warm_start=True, init_params=ip)
                    em.fit(features)
                    
                    bic_list.append(em.bic(features))
                    
                    aic_list.append(em.aic(features))
                   
                    comp_list.append(n_components)
                    cv_list.append(cv_type)
                    init_paramsl.append(ip)
                   
                    weights.append(em.weights_)
                    means.append(em.means_)
                    iter_list.append(em.n_iter_)
                    loglower.append(em.lower_bound_)
                    train_labels=em.fit_predict(features)
                    labels_list.append(train_labels)
                    pd.DataFrame(train_labels).to_csv('./csvs/em/{}_train_labels_{}_{}_{}.csv'.format(title_1, cv_type, ip, n_components ))
                    
                    
                    
                    train_probs=em.score_samples(features)
                    pd.DataFrame(train_probs).to_csv('./csvs/em/{}_train_probs_{}_{}_{}.csv'.format(title_1,  cv_type, ip, n_components ))
                    
                    train_prior=em.predict_proba(features)
                    pd.DataFrame(train_prior).to_csv('./csvs/em/{}_train_prior_{}_{}_{}.csv'.format(title_1, cv_type, ip, n_components ))
                   
                labelsdf=pd.DataFrame(np.nan, index=range(len(train_labels)),columns=range(2, features.shape[1]+1))
                for i in range(len(labels_list)):
                    labelsdf.iloc[:,i]=labels_list[i]
                
                labelsdf.to_csv('./csvs/{}_em_clusterslabels_train_{}_{}.csv'.format(title_1, cv_type, ip))
               
        df=pd.DataFrame({'CV Type':cv_list,'init param':init_paramsl,\
                          'Number of Components':comp_list,\
                          'Train BIC':bic_list, \
                          'AIC Train':aic_list, \
                           \
                          'Number of Iterations':iter_list, \
                          'Log Likelihood': loglower})
        df.to_csv('./csvs/{}_em_bic.csv'.format(title_1))
       
        
        wts=pd.DataFrame(weights).to_csv('./csvs/{}_em_weights.csv'.format(title_1))
        meansdf=pd.DataFrame(means).to_csv('./csvs/{}_em_means.csv'.format(title_1))
        
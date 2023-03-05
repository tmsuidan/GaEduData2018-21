#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:13:30 2019

@author: tmsuidan
"""


import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
from sklearn.metrics import adjusted_mutual_info_score as adjmi
from scipy.spatial.distance import jensenshannon as js
from sklearn.decomposition import PCA

from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection as SRP
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.decomposition import FactorAnalysis
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
for dataset in [ 'qsar','dr']:
    for fr_algo in ['nonscaled','scale', 'fa','pca',  'ica','srp','grp']:
        #,'ica','rpg','rps','lda']:
        np.random.seed(123)
        if dataset=='qsar':
            
            data = (pd.read_csv('./data/biodeg.csv', header=None)).values
            title_1='qsar'
            features = np.array(data[:, 0:-1])
            classes = np.array(data[:, -1])
            print(np.unique(classes))
            d=features.shape[1]
            
            
            X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size = 0.3, random_state = 123, stratify=classes)
            
            #https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
            
            # Normalize feature
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            n_comp_pca=14
            n_comp_ica=41
            n_comp_lda=41
            n_comp_rp=41
            n_comp_fa=40
            
            
        if dataset=='dr':
            
            data = (pd.read_csv('./data/messidor_features.arff', header=None)).values
            title_1='dr' 
            features = np.array(data[:, 0:-1])
            classes = np.array(data[:, -1])
            print(np.unique(classes))
            d=features.shape[1]
            
            X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size = 0.3, random_state = 123, stratify=classes)
            
            #adapted from https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
            
            # Normalize feature
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            n_comp_pca=7
            n_comp_ica=16
            n_comp_lda=19
            n_comp_rp=19
            n_comp_fa=17
            
        if fr_algo=='scale':
            
            train=X_train_scaled
            test=X_test_scaled
            #d=train.shape[1]
            
        if fr_algo=='nonscaled': 
            
            train=X_train
            test=X_test
            #d=train.shape[1]
       
        if fr_algo=='pca':
            pca=PCA(n_components=n_comp_pca, whiten=True, random_state=123)
            pca.fit(X_train_scaled, y_train)
            train=pca.fit_transform(X_train_scaled,y_train)
            test=pca.transform(X_test_scaled)
            #d=train.shape[1]
       
        if fr_algo=='ica':
            ica=FastICA(n_components=n_comp_ica,random_state=123)
            ica.fit(X_train_scaled, y_train)
            train=ica.transform(X_train_scaled)
            test=ica.transform(X_test_scaled)
            #d=train.shape[1]
        if fr_algo=='srp':
            srp=SRP(n_components=n_comp_rp,random_state=123)
            srp.fit(X_train_scaled, y_train)
            train=srp.transform(X_train_scaled)
            test=srp.transform(X_test_scaled)
            #d=train.shape[1]
        if fr_algo=='grp':
            grp=GRP(n_components=n_comp_rp,random_state=123)
            grp.fit(X_train_scaled, y_train)
            train=grp.transform(X_train_scaled)
            test=grp.transform(X_test_scaled)
            #d=train.shape[1]
        if fr_algo=='fa':
            FA=FactorAnalysis(n_components=n_comp_fa, svd_method='lapack', random_state=123)
            FA.fit(X_train_scaled, y_train)
            train=FA.transform(X_train_scaled)
            test=FA.transform(X_test_scaled)
            #d=train.shape[1]
        
        distortions = []
        train_silhouette_scores=[]
        test_silhouette_scores=[]
        centers=[]
        train_mi=[]
        test_mi=[]
        n_comp=[]
        labels=[]
        iter_list=[]
        jsl=[]
        train_time=[]
        test_time=[]
        train_clusters=[]
        test_clusters=[]

        for i in range(2, d+1):
            start=perf_counter()
            
            km = KMeans(
                n_clusters=i, init='k-means++',
                n_init=50, max_iter=300,
                tol=1e-04, random_state=123, algorithm='auto'
            )
            km.fit(train)
            end=perf_counter()
            train_time.append(end-start)
            distortions.append(km.inertia_)
            n_comp.append(i)
            train_mi.append(adjmi(y_train,km.predict(train),average_method='arithmetic'))
            test_mi.append(adjmi(y_test,km.predict(test),average_method='arithmetic'))
            centers.append(km.cluster_centers_)
            iter_list.append(km.n_iter_)
            train_cluster_labels=km.fit_predict(train, y_train )
            train_clusters.append(train_cluster_labels)
            pd.DataFrame(train_cluster_labels).to_csv('./csvs/kmeans/{}_train_labels_{}_{}.csv'.format(title_1, fr_algo, i ))
            labels.append(train_cluster_labels)
            start=perf_counter()
            test_cluster_labels=km.predict(test)
            end=perf_counter()
            test_time.append(end-start)
            test_clusters.append(test_cluster_labels)
            pd.DataFrame(test_cluster_labels).to_csv('./csvs/kmeans/{}_test_labels_{}_{}.csv'.format(title_1, fr_algo, i ))
            train_cluster_trans=km.fit_transform(train, y_train)
            test_cluster_trans=km.transform(test)
            pd.DataFrame(train_cluster_trans).to_csv('./csvs/kmeans/{}_train_trans_{}_{}.csv'.format(title_1, fr_algo, i ))
            pd.DataFrame(test_cluster_trans).to_csv('./csvs/kmeans/{}_test_trans_{}_{}.csv'.format(title_1, fr_algo, i ))
            train_silhouette_scores.append(silhouette_score(train, train_cluster_labels))
            test_silhouette_scores.append(silhouette_score(test, test_cluster_labels))
            visualizer = InterclusterDistance(km,  min_size=1, \
                                      max_size=train.shape[0], \
                                      embedding='mds', \
                                      scoring='membership',\
                                      legend=False, \
                                      random_state=123,\
                                      is_fitted=False)

            visualizer.fit(train)        # Fit the data to the visualizer
            visualizer.poof(outpath='./images/km/{}_km_yb_intclustd_{}_{}.png'.format(title_1,fr_algo, i)) 
            plt.close()
        kmdf=pd.DataFrame({
                           'Number of Iterations':iter_list, \
                           'Number of Components':n_comp,'Distortion':distortions,\
                           'Train Silhouette Score': train_silhouette_scores, \
                           'Test Silhouette Score': test_silhouette_scores, \
                           'Train MI':train_mi,'Test MI':test_mi,\
                           'Train time': train_time, 'Test time': test_time})   
        labelsdf=pd.DataFrame(data=np.NAN, index=range(len(y_train)),columns=range(2, d+1))
        for i in range(len(train_clusters)):
            labelsdf.iloc[:,i]=train_clusters[i]
        labelsdf['y_train']=y_train
        labelsdf.to_csv('./csvs/{}_kmeansclusterslabels_train__{}.csv'.format(title_1,fr_algo))
        labelsdf=pd.DataFrame(data=np.NAN, index=range(len(y_test)),columns=range(2, d+1))
        for i in range(len(test_clusters)):
            labelsdf.iloc[:,i]=test_clusters[i]
        labelsdf['y_test']=y_test
        labelsdf.to_csv('./csvs/{}_kmeansclusterslabels_test__{}.csv'.format(title_1,fr_algo))
        kmdf.to_csv('./csvs/{}_kmeans_{}.csv'.format(title_1,fr_algo))
        pd.DataFrame(centers).to_csv('./csvs/{}_kmcenters_{}.csv'.format(title_1,fr_algo))
       
        
        model=KMeans(init='k-means++',
                        n_init=50, max_iter=300,
                        tol=1e-04, random_state=123, algorithm='auto')
        visualizer = KElbowVisualizer(model, metric='silhouette', k=np.arange(2,41), locate_elbow=False)
        visualizer.fit(train)    
        visualizer.poof(outpath='./images/{}_km_yb_sil_{}.png'.format(title_1,fr_algo)) 
        
        plt.close()
        model=KMeans(random_state=123)
        visualizer = KElbowVisualizer(model,  k=np.arange(2,41), locate_elbow=False)
        visualizer.fit(train)    
        visualizer.poof(outpath='./images/{}_km_yb_dist_{}.png'.format(title_1,fr_algo)) 
        
        plt.close()
        model=KMeans(random_state=123)
        visualizer = KElbowVisualizer(model, metric='calinski_harabasz', k=np.arange(2,41), locate_elbow=False)
        visualizer.fit(train)    
        visualizer.poof(outpath='./images/{}_km_yb_ch_{}.png'.format(title_1,fr_algo)) 
        plt.close()
        
            
        
        #adapted from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
        lowest_bic = np.infty
        bic_train = []
        bic_test=[]
        comp_list=[]
        cv_list=[]
        train_mi=[]
        test_mi=[]
        weights=[]
        means=[] 
        aic_train=[]
        aic_test=[]
        init_paramsl=[]
        iter_list=[]
        loglower=[]
        train_time=[]
        test_time=[]
        train_labels_list=[]
        test_labels_list=[]
        n_components_range = range(2,d+1)
        cv_types = [ 'diag', 'full']
        for cv_type in cv_types:
            for ip in ['kmeans','random']:
                train_labels_list=[]
                test_labels_list=[]
                for n_components in n_components_range:
                    
                
                # Fit a Gaussian mixture with EM
                    start=perf_counter()
                    em = GaussianMixture(n_components=n_components, max_iter=300, \
                                         n_init=30, covariance_type=cv_type, \
                                         random_state=123, warm_start=True, init_params=ip)
                    em.fit(train, y_train)
                    end=perf_counter()
                    train_time.append(end-start)
                    bic_train.append(em.bic(train))
                    bic_test.append(em.bic(test))
                    aic_train.append(em.aic(train))
                    aic_test.append(em.aic(test))
                    comp_list.append(n_components)
                    cv_list.append(cv_type)
                    init_paramsl.append(ip)
                    train_mi.append(adjmi(y_train,em.predict(train),average_method='arithmetic'))
                    test_mi.append(adjmi(y_test,em.predict(test),average_method='arithmetic'))
                    weights.append(em.weights_)
                    means.append(em.means_)
                    iter_list.append(em.n_iter_)
                    loglower.append(em.lower_bound_)
                    train_labels=em.fit_predict(train, y_train)
                    train_labels_list.append(train_labels)
                    pd.DataFrame(train_labels).to_csv('./csvs/em/{}_train_labels_{}_{}_{}_{}.csv'.format(title_1, fr_algo, cv_type, ip, n_components ))
                    start=perf_counter()
                    test_labels=em.predict(test)
                    end=perf_counter()
                    test_time.append(end-start)
                    test_labels_list.append(test_labels)
                    pd.DataFrame(test_labels).to_csv('./csvs/em/{}_test_labels_{}_{}_{}_{}.csv'.format(title_1, fr_algo, cv_type, ip, n_components ))
                    train_probs=em.score_samples(train)
                    pd.DataFrame(train_probs).to_csv('./csvs/em/{}_train_probs_{}_{}_{}_{}.csv'.format(title_1, fr_algo, cv_type, ip, n_components ))
                    test_probs=em.score_samples(test)
                    pd.DataFrame(test_probs).to_csv('./csvs/em/{}_test_probs_{}_{}_{}_{}.csv'.format(title_1, fr_algo, cv_type, ip, n_components ))
                    train_prior=em.predict_proba(train)
                    pd.DataFrame(train_prior).to_csv('./csvs/em/{}_train_prior_{}_{}_{}_{}.csv'.format(title_1, fr_algo, cv_type, ip, n_components ))
                    test_prior=em.predict_proba(test)
                    pd.DataFrame(test_prior).to_csv('./csvs/em/{}_test_prior_{}_{}_{}_{}.csv'.format(title_1, fr_algo, cv_type, ip, n_components ))
                    
                    if bic_test[-1] < lowest_bic:
                        lowest_bic = bic_test[-1]
                        best_em = em
                labelsdf=pd.DataFrame(data=np.NAN, index=range(len(y_train)),columns=range(2, d+1))
                for i in range(len(train_labels_list)):
                    labelsdf.iloc[:,i]=train_labels_list[i]
                labelsdf['y_train']=y_train
                labelsdf.to_csv('./csvs/{}_em_clusterslabels_train_{}_{}_{}.csv'.format(title_1,fr_algo, cv_type, ip))
                labelsdf=pd.DataFrame(data=np.NAN, index=range(len(y_test)),columns=range(2, d+1))
                for i in range(len(test_labels_list)):
                    labelsdf.iloc[:,i]=test_labels_list[i]
                labelsdf['y_test']=y_test
                labelsdf.to_csv('./csvs/{}_em_clusterslabels_test_{}_{}_{}.csv'.format(title_1,fr_algo, cv_type, ip))
        df=pd.DataFrame({'CV Type':cv_list,'init param':init_paramsl,\
                         'Number of Components':comp_list,\
                         'Train BIC':bic_train,'Test BIC':bic_test, \
                         'AIC Train':aic_train, 'AIC Test': aic_test,\
                         'Train MI':train_mi, 'Test MI': test_mi, \
                         'Number of Iterations':iter_list, \
                         'Log Likelihood': loglower, 'Train time': train_time,\
                         'Test time': test_time})
        df.to_csv('./csvs/{}_em_bic_{}.csv'.format(title_1,fr_algo))
        print(best_em.get_params)
        bestparams=open('./csvs/bestparams_{}_em_{}.txt'.format(title_1,fr_algo), 'w')
        bestparams.write(str(best_em.get_params))
        bestparams.write('\n')
        bestparams.write('best BIC: ' )
        bestparams.write('{}'.format(best_em.bic(train)))
        bestparams.close()
        
        wts=pd.DataFrame(weights).to_csv('./csvs/{}_em_weights_{}.csv'.format(title_1,fr_algo))
        meansdf=pd.DataFrame(means).to_csv('./csvs/{}_em_means_{}.csv'.format(title_1,fr_algo))
        
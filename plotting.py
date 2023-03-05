#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:20:29 2019

@author: tmsuidan https://www.datacamp.com/community/tutorials/introduction-t-sne
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer,balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_mutual_info_score as adjmi
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection as SRP
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.decomposition import FactorAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from matplotlib import cm
plt.rcParams["legend.scatterpoints"] = 1




import os
directory="./images/fr"
if not os.path.exists(directory):
    os.makedirs(directory)
directory="./images/nn"
if not os.path.exists(directory):
    os.makedirs(directory)
path1='./params'
directory="./csvs"
if not os.path.exists(directory):
    os.makedirs(directory)

np.random.seed(123)
def fashion_scatter(x, colors, dataset, n='noclust', transform='original', clust='kmeans'):

    colors=np.array(colors)
    cmap=cm.viridis
    # create a scatter plot.
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
    sc = ax1.scatter(x[:,0], x[:,1], lw=0, s=40, alpha=0.4,c=colors.ravel(), cmap=cm.get_cmap(cmap, len(np.unique(colors))))
    cbar = fig.colorbar(sc, extend='both', shrink=0.9, ax=ax1)
    lns3, lns4 = ax1.get_legend_handles_labels()
    
    #plt.legend( loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)

    legend1 = ax1.legend(*sc.legend_elements(),
                      scatterpoints=len(np.unique(colors)), loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=5)
    ax1.add_artist(legend1)
    fig.tight_layout()

    
    plt.savefig('./images/fr/{}_{}_{}_{}.png'.format(dataset,transform, clust, n))
    plt.close()
    plt.clf()

    return fig, ax1, sc




data = (pd.read_csv('./data/biodeg.csv', header=None)).values

features = np.array(data[:, 0:-1])
classes = np.array(data[:, -1])
print(np.unique(classes))
d=features.shape[1]

X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size = 0.3, random_state = 123, stratify=classes)



# Normalize feature
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clust_labels=pd.read_csv('./csvs/kmeans/qsar_train_labels_scale_2.csv', index_col=0)
fashion_scatter(X_train_scaled, y_train, 'qsar',n='noclust', transform='original', clust='none')
fashion_scatter(X_train_scaled, clust_labels, 'qsar',  n=2, transform='original', clust='kmeans')
clust_labels=pd.read_csv('./csvs/kmeans/qsar_train_labels_scale_3.csv', index_col=0)
fashion_scatter(X_train_scaled, clust_labels, 'qsar', n=3, transform='original', clust='kmeans')
clust_labels=pd.read_csv('./csvs/em/qsar_train_labels_scale_diag_kmeans_33.csv', index_col=0)
fashion_scatter(X_train_scaled, clust_labels, 'qsar', n=33, transform='original', clust='em')

pca=PCA(n_components=14, whiten=True, random_state=123)
pca.fit(X_train_scaled, y_train)
train=pca.fit_transform(X_train_scaled,y_train)
test=pca.transform(X_test_scaled)
pca_clust_labels=pd.read_csv('./csvs/kmeans/qsar_train_labels_pca_4.csv', index_col=0)

fashion_scatter(train, y_train, 'qsar',n='noclust', transform='pca', clust='none')
fashion_scatter(train, pca_clust_labels, 'qsar',n=4, clust='kmeans', transform='pca')
pca_clust_labels=pd.read_csv('./csvs/em/qsar_train_labels_pca_full_kmeans_14.csv', index_col=0)
fashion_scatter(train, pca_clust_labels, 'qsar',n=14, clust='em', transform='pca')

FA=FA=FactorAnalysis(n_components=40, svd_method='lapack', random_state=123)
FA.fit(X_train_scaled, y_train)
train=FA.fit_transform(X_train_scaled, y_train)
test=FA.transform(X_test_scaled)
fa_clust_labels=pd.read_csv('./csvs/kmeans/qsar_train_labels_fa_2.csv', index_col=0)
fashion_scatter(train, y_train, 'qsar', n='noclust', transform='fa', clust='none')


fashion_scatter(train, fa_clust_labels, 'qsar',  n=2, transform='fa', clust='kmeans')
fa_clust_labels=pd.read_csv('./csvs/em/qsar_train_labels_fa_full_random_12.csv', index_col=0)
fashion_scatter(train, fa_clust_labels, 'qsar',  n=12, transform='fa', clust='em')

ica=FastICA(n_components=41,random_state=123)
ica.fit(X_train_scaled, y_train)
train=ica.transform(X_train_scaled)
test=ica.transform(X_test_scaled)
ica_clust_labels=pd.read_csv('./csvs/kmeans/qsar_train_labels_ica_24.csv', index_col=0)

fashion_scatter(train, ica_clust_labels, 'qsar', n=24, transform='ica', clust='kmeans')
ica_clust_labels=pd.read_csv('./csvs/kmeans/qsar_train_labels_ica_2.csv', index_col=0)
fashion_scatter(train, ica_clust_labels, 'qsar', n=2, transform='ica', clust='kmeans')
fashion_scatter(train, y_train, 'qsar',n='noclust', transform='ica', clust='none')
ica_clust_labels=pd.read_csv('./csvs/em/qsar_train_labels_ica_diag_random_21.csv', index_col=0)
fashion_scatter(train, ica_clust_labels, 'qsar',  n=21, transform='ica', clust='em')

srp=SRP(n_components=41,random_state=123)
srp.fit(X_train_scaled, y_train)
train=srp.transform(X_train_scaled)
test=srp.transform(X_test_scaled)
srp_clust_labels=pd.read_csv('./csvs/kmeans/qsar_train_labels_srp_3.csv', index_col=0)

fashion_scatter(train, y_train, 'qsar',transform='srp',clust='none')
fashion_scatter(train, srp_clust_labels, 'qsar', transform='srp', n=3,clust='kmeans')
srp_clust_labels=pd.read_csv('./csvs/em/qsar_train_labels_srp_full_random_3.csv', index_col=0)
fashion_scatter(train, srp_clust_labels, 'qsar',  n=3, transform='srp', clust='em')

grp=GRP(n_components=41,random_state=123)
grp.fit(X_train_scaled, y_train)
train=grp.transform(X_train_scaled)
test=grp.transform(X_test_scaled)
grp_clust_labels=pd.read_csv('./csvs/kmeans/qsar_train_labels_grp_2.csv', index_col=0)

fashion_scatter(train, y_train, 'qsar',transform='grp',clust='none')
fashion_scatter(train, grp_clust_labels, 'qsar', transform='grp', n=2,clust='kmeans')
grp_clust_labels=pd.read_csv('./csvs/em/qsar_train_labels_grp_full_kmeans_4.csv', index_col=0)
fashion_scatter(train, grp_clust_labels, 'qsar',  n=4, transform='grp', clust='em')



data = (pd.read_csv('./data/messidor_features.arff', header=None)).values

features = np.array(data[:, 0:-1])
classes = np.array(data[:, -1])
print(np.unique(classes))
d=features.shape[1]

X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size = 0.3, random_state = 123, stratify=classes)



# Normalize feature
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clust_labels=pd.read_csv('./csvs/kmeans/dr_train_labels_scale_2.csv', index_col=0)
fashion_scatter(X_train_scaled, y_train, 'dr',n='noclust', transform='original', clust='none')
fashion_scatter(X_train_scaled, clust_labels, 'dr',  n=2, transform='original', clust='kmeans')

clust_labels=pd.read_csv('./csvs/em/dr_train_labels_scale_full_kmeans_7.csv', index_col=0)
fashion_scatter(X_train_scaled, clust_labels, 'dr', n=7, transform='original', clust='em')

pca=PCA(n_components=7, whiten=True, random_state=123)
pca.fit(X_train_scaled, y_train)
train=pca.fit_transform(X_train_scaled,y_train)
test=pca.transform(X_test_scaled)

fashion_scatter(train, y_train, 'dr',n='noclust', transform='pca', clust='none')
pca_clust_labels=pd.read_csv('./csvs/kmeans/dr_train_labels_pca_2.csv', index_col=0)
fashion_scatter(train, pca_clust_labels, 'dr',n=2, clust='kmeans', transform='pca')
pca_clust_labels=pd.read_csv('./csvs/kmeans/dr_train_labels_pca_5.csv', index_col=0)
fashion_scatter(train, pca_clust_labels, 'dr',n=5, clust='kmeans', transform='pca')
pca_clust_labels=pd.read_csv('./csvs/em/dr_train_labels_pca_full_kmeans_11.csv', index_col=0)
fashion_scatter(train, pca_clust_labels, 'dr',n=11, clust='em', transform='pca')

FA=FA=FactorAnalysis(n_components=17, svd_method='lapack', random_state=123)
FA.fit(X_train_scaled, y_train)
train=FA.fit_transform(X_train_scaled, y_train)
test=FA.transform(X_test_scaled)
fashion_scatter(train, y_train, 'dr', n='noclust', transform='fa', clust='none')
fa_clust_labels=pd.read_csv('./csvs/kmeans/dr_train_labels_fa_8.csv', index_col=0)

fashion_scatter(train, fa_clust_labels, 'dr',  n=8, transform='fa', clust='kmeans')
fa_clust_labels=pd.read_csv('./csvs/em/dr_train_labels_fa_full_kmeans_9.csv', index_col=0)
fashion_scatter(train, fa_clust_labels, 'dr',  n=9, transform='fa', clust='em')

ica=FastICA(n_components=16,random_state=123)
ica.fit(X_train_scaled, y_train)
train=ica.transform(X_train_scaled)
test=ica.transform(X_test_scaled)
fashion_scatter(train, y_train, 'dr',n='noclust', transform='ica', clust='none')
ica_clust_labels=pd.read_csv('./csvs/kmeans/dr_train_labels_ica_16.csv', index_col=0)

fashion_scatter(train, ica_clust_labels, 'dr', n=16, transform='ica', clust='kmeans')
ica_clust_labels=pd.read_csv('./csvs/kmeans/dr_train_labels_ica_3.csv', index_col=0)
fashion_scatter(train, ica_clust_labels, 'dr', n=3, transform='ica', clust='kmeans')

ica_clust_labels=pd.read_csv('./csvs/em/dr_train_labels_ica_full_kmeans_6.csv', index_col=0)
fashion_scatter(train, ica_clust_labels, 'dr',  n=6, transform='ica', clust='em')

srp=SRP(n_components=19,random_state=123)
srp.fit(X_train_scaled, y_train)
train=srp.transform(X_train_scaled)
test=srp.transform(X_test_scaled)
fashion_scatter(train, y_train, 'dr',transform='srp',clust='none')
srp_clust_labels=pd.read_csv('./csvs/kmeans/dr_train_labels_srp_3.csv', index_col=0)


fashion_scatter(train, srp_clust_labels, 'dr', transform='srp', n=3,clust='kmeans')
srp_clust_labels=pd.read_csv('./csvs/em/dr_train_labels_srp_full_kmeans_7.csv', index_col=0)
fashion_scatter(train, srp_clust_labels, 'dr',  n=7, transform='srp', clust='em')

grp=GRP(n_components=19,random_state=123)
grp.fit(X_train_scaled, y_train)
train=grp.transform(X_train_scaled)
test=grp.transform(X_test_scaled)
fashion_scatter(train, y_train, 'dr',transform='grp',clust='none')
grp_clust_labels=pd.read_csv('./csvs/kmeans/dr_train_labels_grp_2.csv', index_col=0)
fashion_scatter(train, grp_clust_labels, 'dr', transform='grp', n=2,clust='kmeans')
grp_clust_labels=pd.read_csv('./csvs/em/dr_train_labels_grp_full_kmeans_7.csv', index_col=0)
fashion_scatter(train, grp_clust_labels, 'dr',  n=7, transform='grp', clust='em')

def plot_iters_curve(filename,  imagename, x_scale='log', vertical=False, index_column='param_max_iter',plot_x_axis='Number of Iterations'):
    df2=pd.read_csv(filename, index_col=index_column)
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
   
    ax1.set_xlabel(plot_x_axis)
    ax1.set_ylabel("Balanced Accuracy Score")
    plt.xscale(x_scale)
    
    plt.grid()
    plt.fill_between(df2.index, df2['mean_train_score'] - df2['std_train_score'],
                     df2['mean_train_score'] + df2['std_train_score'], alpha=0.1,
                     color="g")
    plt.fill_between(df2.index,  df2['mean_test_score']-df2['std_test_score'],
                     df2['mean_test_score']+df2['std_test_score'], alpha=0.1, color="tab:purple")
    lns3=plt.plot(df2.index,df2['mean_train_score'], 'o-', color="g",  marker=',',
             linewidth=3, label="Training Score")
    lns4=plt.plot(df2.index, df2['mean_test_score'], 'o-', color="tab:purple", marker=',',
             linewidth=3, label="Cross-validation Score")

    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.set_ylim(0,2.5*df2[['mean_fit_time','mean_score_time']].values.max())
    plt.fill_between(df2.index, df2['mean_fit_time'] - df2['std_fit_time'],
                     df2['mean_fit_time'] + df2['std_fit_time'], alpha=0.1,
                     color="b")
    plt.fill_between(df2.index,  df2['mean_score_time']-df2['std_score_time'],
                     df2['mean_score_time']+df2['std_score_time'], alpha=0.1, color="r")
    lns7=plt.plot(df2.index,df2['mean_fit_time'], 'o-', color="b",linestyle='--',marker=',',
             linewidth=2, label="Training Time")
    lns8=plt.plot(df2.index, df2['mean_score_time'], 'o-', color="r",linestyle='--',marker=',',
             linewidth=2, label="Cross-validation Time")
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)

    #plt.legend(loc="best")
    fig.tight_layout()
   
    plt.savefig('./images/nn/{}.png'.format(imagename))
    plt.close()
    plt.clf()
    
plot_iters_curve('./csvs/nn/qsarnncv_fa.csv','qsarnncv_fa')
plot_iters_curve('./csvs/nn/qsarnncv_fa_em_labels.csv','qsarnncv_fa_em_labels')
plot_iters_curve('./csvs/nn/qsarnncv_fa_em_values.csv','qsarnncv_fa_em_values')
plot_iters_curve('./csvs/nn/qsarnncv_fa_em_values_score.csv','qsarnncv_fa_em_values_score')
plot_iters_curve('./csvs/nn/qsarnncv_fa_kmeans_labels.csv','qsarnncv_fa_kmeans_labels')
plot_iters_curve('./csvs/nn/qsarnncv_fa_kmeans_values.csv','qsarnncv_fa_kmeans_values')
plot_iters_curve('./csvs/nn/qsarnncv_grp.csv','qsarnncv_grp')
plot_iters_curve('./csvs/nn/qsarnncv_grp_em_labels.csv','qsarnncv_grp_em_labels')
plot_iters_curve('./csvs/nn/qsarnncv_grp_em_values.csv','qsarnncv_grp_em_values')
plot_iters_curve('./csvs/nn/qsarnncv_grp_em_values_score.csv','qsarnncv_grp_em_values_score')
plot_iters_curve('./csvs/nn/qsarnncv_grp_kmeans_labels.csv','qsarnncv_grp_kmeans_labels')
plot_iters_curve('./csvs/nn/qsarnncv_grp_kmeans_values.csv','qsarnncv_grp_kmeans_values')
plot_iters_curve('./csvs/nn/qsarnncv_ica.csv','qsarnncv_ica')
plot_iters_curve('./csvs/nn/qsarnncv_ica_em_labels.csv','qsarnncv_ica_em_labels')
plot_iters_curve('./csvs/nn/qsarnncv_ica_em_values.csv','qsarnncv_ica_em_values')
plot_iters_curve('./csvs/nn/qsarnncv_ica_em_values_score.csv','qsarnncv_ica_em_values_score')
plot_iters_curve('./csvs/nn/qsarnncv_ica_kmeans_labels.csv','qsarnncv_ica_kmeans_labels')
plot_iters_curve('./csvs/nn/qsarnncv_ica_kmeans_values.csv','qsarnncv_ica_kmeans_values')
plot_iters_curve('./csvs/nn/qsarnncv_orig.csv','qsarnncv_scale')
plot_iters_curve('./csvs/nn/qsarnncv_pca.csv','qsarnncv_pca')
plot_iters_curve('./csvs/nn/qsarnncv_pca_em_labels.csv','qsarnncv_pca_em_labels')
plot_iters_curve('./csvs/nn/qsarnncv_pca_em_values.csv','qsarnncv_pca_em_values')
plot_iters_curve('./csvs/nn/qsarnncv_pca_em_values_score.csv','qsarnncv_pca_em_values_score')
plot_iters_curve('./csvs/nn/qsarnncv_pca_kmeans_labels.csv','qsarnncv_pca_kmeans_labels')
plot_iters_curve('./csvs/nn/qsarnncv_pca_kmeans_values.csv','qsarnncv_pca_kmeans_values')
plot_iters_curve('./csvs/nn/qsarnncv_scale_em_labels.csv','qsarnncv_scale_em_labels')
plot_iters_curve('./csvs/nn/qsarnncv_scale_em_values.csv','qsarnncv_scale_em_values')
plot_iters_curve('./csvs/nn/qsarnncv_scale_em_values_score.csv','qsarnncv_scale_em_values_score')
plot_iters_curve('./csvs/nn/qsarnncv_scale_kmeans_labels.csv','qsarnncv_scale_kmeans_labels')
plot_iters_curve('./csvs/nn/qsarnncv_scale_kmeans_values.csv','qsarnncv_scale_kmeans_values')
plot_iters_curve('./csvs/nn/qsarnncv_scale_em_values_both.csv','qsarnncv_scale_both')
plot_iters_curve('./csvs/nn/qsarnncv_srp.csv','qsarnncv_srp')
plot_iters_curve('./csvs/nn/qsarnncv_srp_em_labels.csv','qsarnncv_srp_em_labels')
plot_iters_curve('./csvs/nn/qsarnncv_srp_em_values.csv','qsarnncv_srp_em_values')
plot_iters_curve('./csvs/nn/qsarnncv_srp_kmeans_labels.csv','qsarnncv_srp_kmeans_labels')
plot_iters_curve('./csvs/nn/qsarnncv_srp_kmeans_values.csv','qsarnncv_srp_kmeans_values')




def em_plot(dataset, fr_algo):
    df=pd.read_csv('./csvs/{}_em_bic_{}.csv'.format(dataset,fr_algo), index_col=0)
    dfdiag=df[df['CV Type']=='diag' ]
    dfdiagkmeans= dfdiag[dfdiag['init param']=='kmeans']
    dfdiagrandom= dfdiag[dfdiag['init param']=='random']
    dffull=df[df['CV Type']=='full' ]
    dffullkmeans= dffull[dffull['init param']=='kmeans']
    dffullrandom= dffull[dffull['init param']=='random']
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("BIC")
    plt.plot(dfdiagkmeans['Number of Components'], dfdiagkmeans['Train BIC'], label='Diag kmeans')
    plt.plot(dfdiagrandom['Number of Components'], dfdiagrandom['Train BIC'], label='Diag random')
    plt.plot(dffullrandom['Number of Components'], dffullrandom['Train BIC'], label='Full random')
    plt.plot(dffullkmeans['Number of Components'], dffullkmeans['Train BIC'], label='Full kmeans')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    plt.tight_layout()
    ax1.set_facecolor('white')
    
    plt.grid()
    plt.savefig('./images/em/{}_BICncomp_{}.png'.format(dataset,fr_algo))
    plt.close()
    plt.clf()
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("Time(s)")
    plt.plot(dfdiagkmeans['Number of Components'], dfdiagkmeans['Train time'], label='Diag kmeans')
    plt.plot(dfdiagrandom['Number of Components'], dfdiagrandom['Train time'], label='Diag random')
    plt.plot(dffullrandom['Number of Components'], dffullrandom['Train time'], label='Full random')
    plt.plot(dffullkmeans['Number of Components'], dffullkmeans['Train time'], label='Full kmeans')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    plt.tight_layout()
    
    ax1.set_facecolor('white')
    
    plt.grid()
    plt.savefig('./images/em/{}_timencomp_{}.png'.format(dataset,fr_algo))
    plt.close()
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("Time(s)")
    plt.plot(dfdiagkmeans['Number of Components'], dfdiagkmeans['Number of Iterations'], label='Diag kmeans')
    plt.plot(dfdiagrandom['Number of Components'], dfdiagrandom['Number of Iterations'], label='Diag random')
    plt.plot(dffullrandom['Number of Components'], dffullrandom['Number of Iterations'], label='Full random')
    plt.plot(dffullkmeans['Number of Components'], dffullkmeans['Number of Iterations'], label='Full kmeans')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    plt.tight_layout()
    
    ax1.set_facecolor('white')
    
    plt.grid()
    plt.savefig('./images/em/{}_iterncomp_{}.png'.format(dataset,fr_algo))
    plt.close()
    plt.clf()

em_plot('qsar', 'fa')
em_plot('qsar', 'ica')
em_plot('qsar', 'pca')
em_plot('qsar', 'scale')
em_plot('qsar', 'nonscaled')
em_plot('qsar', 'srp')
em_plot('qsar', 'grp')
em_plot('dr', 'fa')
em_plot('dr', 'ica')
em_plot('dr', 'pca')
em_plot('dr', 'scale')
em_plot('dr', 'nonscaled')
em_plot('dr', 'srp')
em_plot('dr', 'grp')

def em_plot2(dataset, fr_algo, cv_type, init_type, y_scale='log'):
    df=pd.read_csv('./csvs/{}_em_bic_{}.csv'.format(dataset,fr_algo), index_col=0)
    df2=df[df['CV Type']==cv_type ]
    df3= df2[df2['init param']==init_type]
    #df3=df3.set_index('Number of Components')
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
    #ax1.set_ylim(df3['Train BIC'].values.min(), df3['Train BIC'].values.max())
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("BIC")
    ax1.set_yscale(y_scale)
    lns3=plt.plot(df3['Number of Components'], df3['Train BIC'], label='Train BIC',\
                 color="g",marker=',', linewidth=3)
    lns4=plt.plot(df3['Number of Components'], df3['Test BIC'], label='Test BIC',\
                  color="tab:purple", marker=',', linewidth=3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.set_ylim(0,3*df3[['Train time', 'Test time']].values.max())
    lns7=plt.plot(df3['Number of Components'],df3['Train time'], label='Train time(s)', linestyle='--',\
                  color="b")
    lns8=plt.plot(df3['Number of Components'],df3['Test time'], label='Test time(s)', color="r", \
                  linestyle='--',)
    #plt.legend()
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    fig.tight_layout()
    
    plt.savefig('./images/em/{}_bestBICncomp_{}_{}_{}.png'.format(dataset,fr_algo, cv_type, init_type))
    plt.close()
    plt.clf()
    
    

em_plot2('qsar', 'fa', 'full', 'random',  y_scale='linear')
em_plot2('qsar', 'ica', 'diag', 'random', y_scale='linear')
em_plot2('qsar', 'pca', 'full', 'kmeans')
em_plot2('qsar', 'scale', 'diag', 'kmeans', y_scale='linear')
em_plot2('qsar', 'nonscaled', 'diag', 'random', y_scale='linear')
em_plot2('qsar', 'srp', 'full', 'random', y_scale='linear')
em_plot2('qsar', 'grp', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'fa', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'ica', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'pca', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'scale', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'nonscaled', 'full', 'kmeans')
em_plot2('dr', 'srp', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'grp', 'full', 'kmeans', y_scale='linear')

def em_plot3(dataset, fr_algo, cv_type, init_type, y_scale='linear'):
    df=pd.read_csv('./csvs/{}_em_bic_{}.csv'.format(dataset,fr_algo), index_col=0)
    df2=df[df['CV Type']==cv_type ]
    df3= df2[df2['init param']==init_type]
    #df3=df3.set_index('Number of Components')
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
    #ax1.set_ylim(df3['Train BIC'].values.min(), df3['Train BIC'].values.max())
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("BIC")
    ax1.set_yscale(y_scale)
    lns3=plt.plot(df3['Number of Components'], df3['Train MI'], label='Train MI',\
                 color="g",marker=',', linewidth=3)
    lns4=plt.plot(df3['Number of Components'], df3['Test MI'], label='Test MI',\
                  color="tab:purple", marker=',', linewidth=3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.set_ylim(0,3*df3[['Train time', 'Test time']].values.max())
    lns7=plt.plot(df3['Number of Components'],df3['Train time'], label='Train time(s)', linestyle='--',\
                  color="b")
    lns8=plt.plot(df3['Number of Components'],df3['Test time'], label='Test time(s)', color="r", \
                  linestyle='--',)
    #plt.legend()
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    fig.tight_layout()
    
    plt.savefig('./images/em/{}_bestMIncomp_{}_{}_{}.png'.format(dataset,fr_algo, cv_type, init_type))
    plt.close()
    plt.clf()
    
    

em_plot3('qsar', 'fa', 'full', 'random')
em_plot3('qsar', 'ica', 'diag', 'random', y_scale='linear')
em_plot3('qsar', 'pca', 'full', 'kmeans')
em_plot3('qsar', 'scale', 'diag', 'kmeans', y_scale='linear')
em_plot3('qsar', 'nonscaled', 'diag', 'random', y_scale='linear')
em_plot3('qsar', 'srp', 'full', 'random', y_scale='linear')
em_plot3('qsar', 'grp', 'full', 'kmeans', y_scale='linear')
em_plot3('dr', 'fa', 'full', 'kmeans')
em_plot3('dr', 'ica', 'full', 'kmeans', y_scale='linear')
em_plot3('dr', 'pca', 'full', 'kmeans')
em_plot3('dr', 'scale', 'full', 'kmeans', y_scale='linear')
em_plot3('dr', 'nonscaled', 'full', 'kmeans')
em_plot3('dr', 'srp', 'full', 'kmeans', y_scale='linear')
em_plot3('dr', 'grp', 'full', 'kmeans', y_scale='linear')

def em_plot4(dataset, fr_algo, cv_type, init_type, y_scale='linear'):
    df=pd.read_csv('./csvs/{}_em_bic_{}.csv'.format(dataset,fr_algo), index_col=0)
    df2=df[df['CV Type']==cv_type ]
    df3= df2[df2['init param']==init_type]
    #df3=df3.set_index('Number of Components')
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
    #ax1.set_ylim(df3['Train BIC'].values.min(), df3['Train BIC'].values.max())
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("BIC")
    ax1.set_yscale(y_scale)
    lns3=plt.plot(df3['Number of Components'], df3['Train BIC'], label='Train BIC',\
                 color="g",marker=',', linewidth=3)
#    lns4=plt.plot(df3['Number of Components'], df3['Test BIC'], label='Test BIC',\
#                  color="tab:purple", marker=',', linewidth=3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.set_ylim(0,3*df3[['Train time', 'Test time']].values.max())
    lns7=plt.plot(df3['Number of Components'],df3['Train time'], label='Train time(s)', linestyle='--',\
                  color="b")
    lns8=plt.plot(df3['Number of Components'],df3['Test time'], label='Test time(s)', color="r", \
                  linestyle='--',)
    #plt.legend()
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    fig.tight_layout()
    
    plt.savefig('./images/em/{}_bestBICtrainncomp_{}_{}_{}.png'.format(dataset,fr_algo, cv_type, init_type))
    plt.close()
    plt.clf()
    
    

em_plot4('qsar', 'fa', 'full', 'random')
em_plot4('qsar', 'ica', 'diag', 'random', y_scale='linear')
em_plot4('qsar', 'pca', 'full', 'kmeans')
em_plot4('qsar', 'scale', 'diag', 'kmeans', y_scale='linear')
em_plot4('qsar', 'nonscaled', 'diag', 'random', y_scale='linear')
em_plot4('qsar', 'srp', 'full', 'random', y_scale='linear')
em_plot4('qsar', 'grp', 'full', 'kmeans', y_scale='linear')
em_plot4('dr', 'fa', 'full', 'kmeans')
em_plot4('dr', 'ica', 'full', 'kmeans', y_scale='linear')
em_plot4('dr', 'pca', 'full', 'kmeans')
em_plot4('dr', 'scale', 'full', 'kmeans', y_scale='linear')
em_plot4('dr', 'nonscaled', 'full', 'kmeans')
em_plot4('dr', 'srp', 'full', 'kmeans', y_scale='linear')
em_plot4('dr', 'grp', 'full', 'kmeans', y_scale='linear')

def fr_plot(dataset, fr_algo, filename):
    df=pd.read_csv(filename, index_col=0)
    df=df.set_index('Number of Components')
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("Number of Iterations")
    lns3=plt.plot(df.index, df['Number of Iterations'], label='Iterations', color="g")
    #plt.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.set_ylim(0,2.5*df[['Time(s)']].values.max())
    lns7=plt.plot(df.index,df['Time(s)'], label='Time(s)', color="tab:purple")
    #plt.legend()
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    fig.tight_layout()
#    #plt.legend(loc="best")
    plt.savefig('./images/fr/{}_itertime_{}.png'.format(dataset,fr_algo))
    plt.close()
    plt.clf()
    plt.plot(df.index, df['Mean Train Log Likelihood'], label='Mean Train Log Likelihood')
    plt.plot(df.index, df['Mean Test Log Likelihood'], label='Mean Test Log Likelihood')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Log likelihood')
    #plt.xticks(df.index)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    ax1.set_facecolor('white')
    
    plt.grid()
    plt.tight_layout()
    plt.savefig('./images/fr/{}_choose_n_{}.png'.format(dataset,fr_algo))
    plt.close()
    plt.clf()
    
    
fr_plot('dr','fa','./csvs/fa/dr_fa.csv')    
fr_plot('qsar','fa','./csvs/fa/qsar_fa.csv')   

def ica_plot(dataset, fr_algo):
    df=pd.read_csv('./csvs/ica/{}_{}.csv'.format(dataset,fr_algo), index_col=0)
    df=df.set_index('Number of Components')
    #print(df.head())
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("Mean Kurtosis of Components")
    plt.plot(df.index, df['Mean Train Kurtosis'], label='Train')
    plt.plot(df.index, df['Mean test Kurtosis'], label='Test')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    plt.tight_layout()
    
    ax1.set_facecolor('white')
    
    plt.grid()
    plt.savefig('./images/fr/{}_kurtosis_{}.png'.format(dataset,fr_algo))
    plt.close()
    plt.clf()
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("Number of Iterations")
    lns3=plt.plot(df.index, df['Iterations'], label='Iterations', color="g")
    #plt.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.set_ylim(0,2.5*df[['Time(s)']].values.max())
    lns7=plt.plot(df.index,df['Time(s)'], label='Time(s)', color="tab:purple")
    #plt.legend()
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    fig.tight_layout()
#    #plt.legend(loc="best")
    plt.savefig('./images/fr/{}_itertime_{}.png'.format(dataset,fr_algo))
    plt.close()
    plt.clf()
    
ica_plot('dr','ica')
ica_plot('qsar','ica')

def rp_plot(dataset, fr_algo):
    df=pd.read_csv('./csvs/rp/{}_{}.csv'.format(dataset, fr_algo), index_col=0)
    df=df.set_index('Number of entered components')
    #print(df.head())
    
    
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("Reconstruction Error")
    lns3=plt.plot(df.index, df['Reconstruction Error'], label='Reconstruction Error', color="g")
    plt.fill_between(df.index,  df['Reconstruction Error']-df['RE STD'],
                     df['Reconstruction Error']+df['RE STD'], alpha=0.1, color="g")
    #plt.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.set_ylim(0,2.5*df[['Time(s)']].values.max())
    lns7=plt.plot(df.index,df['Time(s)'], label='Time(s)', color="tab:purple")
    #plt.legend()
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    fig.tight_layout()
#    #plt.legend(loc="best")
    plt.savefig('./images/fr/{}_re_time_{}.png'.format(dataset, fr_algo))
    plt.close()
    plt.clf()
    
rp_plot('dr', 'grp')
rp_plot('qsar', 'grp')
rp_plot('dr', 'srp')
rp_plot('qsar', 'srp')

def pca_plot(dataset):
    df=pd.read_csv('./csvs/pca/{}_pca.csv'.format(dataset), index_col=0)
    df=df.set_index('Number of Components')
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
       
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel("Time(s)")
    lns3=plt.plot(df.index, df['Time(s)'], label='Time', color="g")
    
    
    lns3, lns4 = ax1.get_legend_handles_labels()
    
    
    plt.legend(lns3, lns4, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    fig.tight_layout()

    plt.savefig('./images/fr/{}_pca_time.png'.format(dataset))
    plt.close()
    plt.clf()
    
pca_plot('dr')
pca_plot('qsar')

def km_plot(dataset, fr_algo):
    df=pd.read_csv('./csvs/{}_kmeans_{}.csv'.format(dataset,fr_algo), index_col=0)
    df.set_index('Number of Components')
    #print(df.index)
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
       
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel("Adjusted Mutual Information")
    lns3=plt.plot(df['Number of Components'],df['Train MI'], 'o-', color="g",marker=',',
             linewidth=1, label="Train MI")
    lns4=plt.plot(df['Number of Components'], df['Test MI'], 'o-', color="tab:purple",marker=',',
             linewidth=1, label="Test MI")
    
    #plt.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Iterations")
    #ax2.set_ylim(0,3*df[['Train time', 'Test time']].values.max())
    lns7=plt.plot(df['Number of Components'], df['Number of Iterations'], label='Iterations', \
                  linestyle='--', color="b")
#    lns7=plt.plot(df['Number of Components'],df['Train time'], label='Train time(s)', linestyle='--',\
#                  color="b")
#    lns8=plt.plot(df['Number of Components'],df['Test time'], label='Test time(s)', color="r", \
#                  linestyle='--',)
    #plt.legend()
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
    fig.tight_layout()
#    #plt.legend(loc="best")
    plt.savefig('./images/km/{}_kmeans_ami_iter_{}.png'.format(dataset,fr_algo))
    plt.close()
    plt.clf()
    
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('white')
    plt.grid()
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel("Silhouette Score")
    lns3=plt.plot(df['Number of Components'],df['Train Silhouette Score'], 'o-', color="g",  marker=',',
             linewidth=3, label="Training Silhouette")
    lns4=plt.plot(df['Number of Components'], df['Test Silhouette Score'], 'o-', color="tab:purple", marker=',',
             linewidth=3, label="Test Silhouette")

    
    ax2 = ax1.twinx()
    
    ax2.set_ylabel("Time(s)")
    ax2.set_ylim(0,3*df[['Train time', 'Test time']].values.max())
    #ax2.set_ylim(0,2.5*df[['Train MI','Test MI']].values.max())
    lns7=plt.plot(df['Number of Components'],df['Train time'], label='Train time(s)', linestyle='--',\
                  color="b")
    lns8=plt.plot(df['Number of Components'],df['Test time'], label='Test time(s)', color="r", \
                  linestyle='--',)
    
    
    
    lns3, lns4 = ax1.get_legend_handles_labels()
    lns7, lns8 = ax2.get_legend_handles_labels()
    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)

    #plt.legend(loc="best")
    fig.tight_layout()
   
    plt.savefig('./images/km/{}_kmeans_sil_time_{}.png'.format(dataset,fr_algo))
    plt.close()
    plt.clf()
    
km_plot('qsar', 'nonscaled')
km_plot('qsar', 'scale')
km_plot('qsar', 'fa')
km_plot('qsar', 'pca')
km_plot('qsar', 'ica')
km_plot('qsar', 'srp')
km_plot('qsar', 'grp')
km_plot('dr', 'nonscaled')
km_plot('dr', 'scale')
km_plot('dr', 'fa')
km_plot('dr', 'pca')
km_plot('dr', 'ica')
km_plot('dr', 'srp')
km_plot('dr', 'grp')

'''https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html'''
def plot_confusion_matrix(filename, figname,classes,normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    

    cm=pd.read_csv(filename)
    cm=(cm.iloc[:,1:]).values
    print(cm)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
#           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt =  'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('./images/nn/{}_nonorm.png'.format(figname))
    plt.close()
    plt.clf()
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' 
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('./images/nn/{}_norm.png'.format(figname))
    plt.close()
    plt.clf()
    return ax



plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_fa.csv',figname='qsar_nncf_fa', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_fa_em_labels.csv',figname='qsar_nncf_fa_em_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_fa_em_values.csv',figname='qsar_nncf_fa_em_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_fa_em_values_score.csv',figname='qsar_nncf_fa_em_values_score', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_fa_kmeans_labels.csv',figname='qsar_nncf_fa_kmeans_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_fa_kmeans_values.csv',figname='qsar_nncf_fa_kmeans_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_grp.csv',figname='qsar_nncf_grp', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_grp_em_labels.csv',figname='qsar_nncf_grp_em_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_grp_em_values.csv',figname='qsar_nncf_grp_em_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_grp_em_values_score.csv',figname='qsar_nncf_grp_em_values_score', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_grp_kmeans_labels.csv',figname='qsar_nncf_grp_kmeans_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_grp_kmeans_values.csv',figname='qsar_nncf_grp_kmeans_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_ica.csv',figname='qsar_nncf_ica', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_ica_em_labels.csv',figname='qsar_nncf_ica_em_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_ica_em_values.csv',figname='qsar_nncf_ica_em_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_ica_em_values_score.csv',figname='qsar_nncf_ica_em_values_score', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_ica_kmeans_labels.csv',figname='qsar_nncf_ica_kmeans_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_ica_kmeans_values.csv',figname='qsar_nncf_ica_kmeans_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_orig.csv',figname='qsar_nncf_scale', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_pca.csv',figname='qsar_nncf_pca', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_pca_em_labels.csv',figname='qsar_nncf_pca_em_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_pca_em_values.csv',figname='qsar_nncf_pca_em_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_pca_em_values_score.csv',figname='qsar_nncf_pca_em_values_score', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_pca_kmeans_labels.csv',figname='qsar_nncf_pca_kmeans_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_pca_kmeans_values.csv',figname='qsar_nncf_pca_kmeans_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)

plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_scale_em_labels.csv',figname='qsar_nncf_scale_em_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_scale_em_values_both.csv',figname='qsar_nncf_scale_both', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_scale_em_values.csv',figname='qsar_nncf_scale_em_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_scale_em_values_score.csv',figname='qsar_nncf_scale_em_values_score', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_scale_kmeans_labels.csv',figname='qsar_nncf_scale_kmeans_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_scale_kmeans_values.csv',figname='qsar_nncf_scale_kmeans_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_srp.csv',figname='qsar_nncf_srp', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_srp_em_labels.csv',figname='qsar_nncf_srp_em_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_srp_em_values.csv',figname='qsar_nncf_srp_em_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_srp_em_values_score.csv',figname='qsar_nncf_srp_em_values_score', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_srp_kmeans_labels.csv',figname='qsar_nncf_srp_kmeans_labels', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)
plot_confusion_matrix(filename='./csvs/nn/qsar_nncf_srp_kmeans_values.csv',figname='qsar_nncf_srp_kmeans_values', classes=['non biodegradable','biodegradable'],normalize=False, 
                          title=None,
                          cmap=plt.cm.Blues)

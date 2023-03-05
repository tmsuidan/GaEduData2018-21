# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:01:49 2019

@author: talam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from matplotlib import cm
plt.rcParams["legend.scatterpoints"] = 1
import os
directory="./images/em"
if not os.path.exists(directory):
    os.makedirs(directory)
directory="./images/fr"
if not os.path.exists(directory):
    os.makedirs(directory)
    
#dataset='qsar'
#fr_algo='fa'
#
#def em_plot(dataset, fr_algo):
#    df=pd.read_csv('./csvs/{}_em_bic_{}.csv'.format(dataset,fr_algo), index_col=0)
#    dfdiag=df[df['CV Type']=='diag' ]
#    dfdiagkmeans= dfdiag[dfdiag['init param']=='kmeans']
#    dfdiagrandom= dfdiag[dfdiag['init param']=='random']
#    dffull=df[df['CV Type']=='full' ]
#    dffullkmeans= dffull[dffull['init param']=='kmeans']
#    dffullrandom= dffull[dffull['init param']=='random']
#    fig=plt.figure()
#    ax1 = fig.add_subplot(111)
#       
#    ax1.set_xlabel('Number of Components')
#    ax1.set_ylabel("BIC")
#    plt.plot(dfdiagkmeans['Number of Components'], dfdiagkmeans['Train BIC'], label='Diag kmeans')
#    plt.plot(dfdiagrandom['Number of Components'], dfdiagrandom['Train BIC'], label='Diag random')
#    plt.plot(dffullrandom['Number of Components'], dffullrandom['Train BIC'], label='Full random')
#    plt.plot(dffullkmeans['Number of Components'], dffullkmeans['Train BIC'], label='Full kmeans')
#    plt.legend()
#    ax1.set_facecolor('white')
#    
#    plt.grid()
#    plt.savefig('./images/em/{}_BICncomp_{}.png'.format(dataset,fr_algo))
#    plt.close()
#    
#    fig=plt.figure()
#    ax1 = fig.add_subplot(111)
#       
#    ax1.set_xlabel('Number of Components')
#    ax1.set_ylabel("Time(s)")
#    plt.plot(dfdiagkmeans['Number of Components'], dfdiagkmeans['Time(s)'], label='Diag kmeans')
#    plt.plot(dfdiagrandom['Number of Components'], dfdiagrandom['Time(s)'], label='Diag random')
#    plt.plot(dffullrandom['Number of Components'], dffullrandom['Time(s)'], label='Full random')
#    plt.plot(dffullkmeans['Number of Components'], dffullkmeans['Time(s)'], label='Full kmeans')
#    plt.legend()
#    ax1.set_facecolor('white')
#    
#    plt.grid()
#    plt.savefig('./images/em/{}_timencomp_{}.png'.format(dataset,fr_algo))
#    plt.close()
#    fig=plt.figure()
#    ax1 = fig.add_subplot(111)
#       
#    ax1.set_xlabel('Number of Components')
#    ax1.set_ylabel("Time(s)")
#    plt.plot(dfdiagkmeans['Number of Components'], dfdiagkmeans['Number of Iterations'], label='Diag kmeans')
#    plt.plot(dfdiagrandom['Number of Components'], dfdiagrandom['Number of Iterations'], label='Diag random')
#    plt.plot(dffullrandom['Number of Components'], dffullrandom['Number of Iterations'], label='Full random')
#    plt.plot(dffullkmeans['Number of Components'], dffullkmeans['Number of Iterations'], label='Full kmeans')
#    plt.legend()
#    ax1.set_facecolor('white')
#    
#    plt.grid()
#    plt.savefig('./images/em/{}_iterncomp_{}.png'.format(dataset,fr_algo))
#    plt.close()
#
#em_plot('qsar', 'fa')
#em_plot('qsar', 'ica')
#em_plot('qsar', 'pca')
#em_plot('qsar', 'scale')
#em_plot('qsar', 'nonscaled')
#em_plot('qsar', 'srp')
##
#def km_plot(dataset, fr_algo):
#    df=pd.read_csv('./csvs/{}_kmeans_{}.csv'.format(dataset,fr_algo), index_col=0)
#    df.set_index('Number of Components')
#    fig=plt.figure()
#    ax1 = fig.add_subplot(111)
#    ax1.set_facecolor('white')
#    plt.grid()
#       
#    ax1.set_xlabel('Number of Components')
#    ax1.set_ylabel("Adjusted Mutual Information")
#    lns3=plt.plot(df.index,df['Train MI'], 'o-', color="g",marker=',',
#             linewidth=1, label="Train MI")
#    lns4=plt.plot(df.index, df['Test MI'], 'o-', color="tab:purple",marker=',',
#             linewidth=1, label="Test MI")
#    
#    #plt.legend()
#    ax2 = ax1.twinx()
#    ax2.set_ylabel("Time(s)")
#    ax2.set_ylim(0,3*df[['Train time', 'Test time']].values.max())
#    lns7=plt.plot(df.index,df['Train time'], label='Train time(s)', linestyle='--',\
#                  color="b")
#    lns8=plt.plot(df.index,df['Test time'], label='Test time(s)', color="r", \
#                  linestyle='--',)
#    #plt.legend()
#    lns3, lns4 = ax1.get_legend_handles_labels()
#    lns7, lns8 = ax2.get_legend_handles_labels()
#    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
#    fig.tight_layout()
##    #plt.legend(loc="best")
#    plt.savefig('./images/km/{}_kmeans_ami_time_{}.png'.format(dataset,fr_algo))
#    plt.close()
#    
#    fig=plt.figure()
#    ax1 = fig.add_subplot(111)
#    ax1.set_facecolor('white')
#    plt.grid()
#    ax1.set_xlabel('Number of Components')
#    ax1.set_ylabel("Silhouette Score")
#    lns3=plt.plot(df.index,df['Train Silhouette Score'], 'o-', color="g",  marker=',',
#             linewidth=3, label="Training Silhouette")
#    lns4=plt.plot(df.index, df['Test Silhouette Score'], 'o-', color="tab:purple", marker=',',
#             linewidth=3, label="Test Silhouette")
#
#    
#    ax2 = ax1.twinx()
#    ax2.set_ylabel("Iterations")
#    #ax2.set_ylim(0,2.5*df[['Train MI','Test MI']].values.max())
#    lns7=plt.plot(df.index, df['Number of Iterations'], label='Iterations', \
#                  linestyle='--', color="b")
#    
#    lns3, lns4 = ax1.get_legend_handles_labels()
#    lns7, lns8 = ax2.get_legend_handles_labels()
#    ax2.legend(lns3+lns7, lns4+lns8, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
#
#    #plt.legend(loc="best")
#    fig.tight_layout()
#   
#    plt.savefig('./images/km/{}_kmeans_sil_iter_{}.png'.format(dataset,fr_algo))
#    plt.close()
#    
#km_plot('qsar', 'nonscaled')
#km_plot('qsar', 'scale')
#km_plot('qsar', 'fa')
#km_plot('qsar', 'pca')
#km_plot('qsar', 'ica')
#km_plot('qsar', 'srp')
#km_plot('qsar', 'grp')
#km_plot('dr', 'nonscaled')
#km_plot('dr', 'scale')
#km_plot('dr', 'fa')
#km_plot('dr', 'pca')
#km_plot('dr', 'ica')
#km_plot('dr', 'srp')
#km_plot('dr', 'grp')


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
    
    

em_plot2('qsar', 'fa', 'full', 'random')
em_plot2('qsar', 'ica', 'diag', 'random', y_scale='linear')
em_plot2('qsar', 'pca', 'full', 'kmeans')
em_plot2('qsar', 'scale', 'diag', 'kmeans', y_scale='linear')
em_plot2('qsar', 'nonscaled', 'diag', 'random', y_scale='linear')
em_plot2('qsar', 'srp', 'full', 'random', y_scale='linear')
em_plot2('qsar', 'grp', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'fa', 'full', 'random')
em_plot2('dr', 'ica', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'pca', 'full', 'kmeans')
em_plot2('dr', 'scale', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'nonscaled', 'full', 'kmeans')
em_plot2('dr', 'srp', 'full', 'kmeans', y_scale='linear')
em_plot2('dr', 'grp', 'full', 'kmeans', y_scale='linear')
#    
#fr_plot('dr','fa','./csvs/fa/dr_fa.csv')    
#fr_plot('qsar','fa','./csvs/fa/qsar_fa.csv')   
#    

#    
#    fig=plt.figure()
#    ax1 = fig.add_subplot(111)
#       
#    ax1.set_xlabel('Number of Components')
#    ax1.set_ylabel("Time(s)")
#    plt.plot(dfdiagkmeans['Number of Components'], dfdiagkmeans['Time(s)'], label='Diag kmeans')
#    plt.plot(dfdiagrandom['Number of Components'], dfdiagrandom['Time(s)'], label='Diag random')
#    plt.plot(dffullrandom['Number of Components'], dffullrandom['Time(s)'], label='Full random')
#    plt.plot(dffullkmeans['Number of Components'], dffullkmeans['Time(s)'], label='Full kmeans')
#    plt.legend()
#    ax1.set_facecolor('white')
#    
#    plt.grid()
#    plt.savefig('./images/em/{}_timencomp_{}.png'.format(dataset,fr_algo))
#    plt.close()
#    fig=plt.figure()
#    ax1 = fig.add_subplot(111)
#       
#    ax1.set_xlabel('Number of Components')
#    ax1.set_ylabel("Time(s)")
#    plt.plot(dfdiagkmeans['Number of Components'], dfdiagkmeans['Number of Iterations'], label='Diag kmeans')
#    plt.plot(dfdiagrandom['Number of Components'], dfdiagrandom['Number of Iterations'], label='Diag random')
#    plt.plot(dffullrandom['Number of Components'], dffullrandom['Number of Iterations'], label='Full random')
#    plt.plot(dffullkmeans['Number of Components'], dffullkmeans['Number of Iterations'], label='Full kmeans')
#    plt.legend()
#    ax1.set_facecolor('white')
#    
#    plt.grid()
#    plt.savefig('./images/em/{}_iterncomp_{}.png'.format(dataset,fr_algo))
#    plt.close()

#df=pd.read_csv('./csvs/pca/{}_pca.csv'.format(dataset, index_col=2)
#fig=plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.set_facecolor('white')
#plt.grid()
#   
#ax1.set_xlabel('Number of Components')
#ax1.set_ylabel("Time(s)")
#lns3=plt.plot(df.index, df['Time(s)'], label='Time', color="g")
##plt.legend()
#
#lns3, lns4 = ax1.get_legend_handles_labels()
#
#plt.legend(lns3, lns4, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
#fig.tight_layout()
##    #plt.legend(loc="best")
#plt.savefig('./images/fr/{}_pca_time_{}.png'.format(dataset))
#plt.close()
#def pca_plot(dataset):
#    df=pd.read_csv('./csvs/pca/{}_pca.csv'.format(dataset), index_col=0)
#    df=df.set_index('Number of Components')
#    fig=plt.figure()
#    ax1 = fig.add_subplot(111)
#    ax1.set_facecolor('white')
#    plt.grid()
#       
#    ax1.set_xlabel('Number of Components')
#    ax1.set_ylabel("Time(s)")
#    lns3=plt.plot(df.index, df['Time(s)'], label='Time', color="g")
#    
#    
#    lns3, lns4 = ax1.get_legend_handles_labels()
#    
#    
#    plt.legend(lns3, lns4, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
#    fig.tight_layout()
#
#    plt.savefig('./images/fr/{}_pca_time.png'.format(dataset))
#    plt.close()
#    
#pca_plot('dr')
#pca_plot('qsar')

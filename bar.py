# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:11:52 2023

@author: talam
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import numpy as np

import os
directory="./images/bar"
if not os.path.exists(directory):
    os.makedirs(directory)

df2018=pd.read_csv('./data/Master_2018-19.csv',index_col='SCHOOL_DSTRCT_NM')
df2018=df2018.iloc[0:180,:]
plot_types=list(df2018.columns)
plot_types=plot_types[1:]

df2018=df2018.add_suffix('_2018-19')

df2019=pd.read_csv('./data/Master_2019-20.csv',index_col='SCHOOL_DSTRCT_NM').add_suffix('_2019-20')
df2019=df2019.iloc[0:180,:]
df2020=pd.read_csv('./data/Master_2020-21.csv',index_col='SCHOOL_DSTRCT_NM').add_suffix('_2020-21')
df2020=df2020.iloc[0:180,:]



for i in plot_types:
    df=pd.DataFrame({'District':df2018.index,\
                     '2018-19':df2018[i+'_2018-19'],\
                     '2019-20':df2019[i+'_2019-20'],\
                     '2020-21':df2020[i+'_2020-21']},\
                    )
        
    test_df=df.melt(id_vars='District')
    
    p1=sns.FacetGrid(data=test_df, col='District', col_wrap=15,  hue='variable').set_titles( fontsize=18)
    p1.map(sns.barplot, 'variable', 'value', order=test_df['variable'].unique()).add_legend()#, hue='variable')
    for ax in p1.axes:
        for p in ax.patches:
                 ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 3),
                     textcoords='offset points', rotation=90)
    plt.rcParams['figure.figsize']=(30,24)
    p1.fig.subplots_adjust(top=0.9)
    p1.fig.suptitle('{}'.format(i), fontsize=20)
    
    p1.savefig('./images/bar/bar_{}.png'.format(i), dpi=300, bbox_inches='tight')
    p1.fig.clf()
    # y_min=math.floor(np.min(df.iloc[:,1:].min())*.1)
    
    # y_max=math.ceil(np.max(df.iloc[:,1:].max())*1.1)
    # steps=math.floor((y_max-y_min)/10)
    # if steps==0: steps=1
    # y_ticks_l=list(range(y_min,y_max,steps))
    
    # width=0.15    
    # df1=df.iloc[0:30,:]
    # df1.name='1st30'
    # df2=df.iloc[30:61,:]
    # df2.name='2nd30'
    # df3=df.iloc[61:91,:]
    # df3.name='3rd30'
    # df4=df.iloc[91:121,:]
    # df4.name='4th30'
    # df5=df.iloc[121:151,:]
    # df5.name='5th30'
    # df6=df.iloc[151:,:]
    # df6.name='6th30'
   
    
    
    # for k in [df1,df2,df3,df4,df5,df6]:
    #     k_str=k.name
    #     labels=list(k.index)
    #     x=np.arange(len(labels))
    #     fig,ax=plt.subplots(figsize=(25,15))
        
        
    #     rects1=ax.bar(x-0.2,k.iloc[:,0],width,label='2018-19', color='blue')
    #     for p in rects1:
    #        height = p.get_height()
    #        ax.annotate('{}'.format(height),
    #           xy=(p.get_x() + p.get_width() / 2, height),
    #           xytext=(0, 3), # 3 points vertical offset
    #           textcoords="offset points",
    #           ha='center', va='bottom', rotation=90)
    #        #ax.set_yticklabels(np.arange(y_min,y_max,(y_max-y_min)/10 ))
    #     rects2=ax.bar(x,k.iloc[:,1],width,label='2019-20', color='turquoise')
    #     for p in rects2:
    #        height = p.get_height()
    #        ax.annotate('{}'.format(height),
    #           xy=(p.get_x() + p.get_width() / 2, height),
    #           xytext=(0, 3), # 3 points vertical offset
    #           textcoords="offset points",
    #           ha='center', va='bottom', rotation=90)
    #        #ax.set_yticklabels(np.arange(y_min,y_max,(y_max-y_min)/10 ))
    #     rects3=ax.bar(x+0.2,k.iloc[:,2],width,label='2020-21', color='red')
    #     for p in rects3:
    #        height = p.get_height()
    #        ax.annotate('{}'.format(height),
    #           xy=(p.get_x() + p.get_width() / 2, height),
    #           xytext=(0, 3), # 3 points vertical offset
    #           textcoords="offset points",
    #           ha='center', va='bottom', rotation=90)
    #        #ax.set_yticklabels(np.arange(y_min,y_max,(y_max-y_min)/10 ))
    #     ax.legend(fontsize=16)
    #     ax.set_title(i, fontsize=20)
    #     ax.tick_params(labelsize=20)
        
    #     ax.set_xticks(x)
        
    #     ax.set_xticklabels(k.index, rotation=90)
    #     ax.set_yticklabels(y_ticks_l)
    #     plt.yticks(y_ticks_l)
    #     plt.ylim(y_min,y_max)
    #     plt.ylabel('{}'.format(i))
    #     plt.tight_layout()
    #     plt.savefig('./images/bar/bar_{}_{}.jpg'.format(i,k_str))
    #     plt.close()
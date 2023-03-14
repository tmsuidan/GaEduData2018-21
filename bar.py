# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:11:52 2023

@author: talam
"""
import pandas as pd
import matplotlib.pyplot as plt

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
    df=pd.DataFrame({'2018-19':df2018[i+'_2018-19'],\
                     '2019-20':df2019[i+'_2019-20'],\
                     '2020-21':df2020[i+'_2020-21']},\
                    index=df2018.index)
    width=0.15
    
    
        
    df1=df.iloc[0:30,:]
    df1.name='1st30'
    df2=df.iloc[30:61,:]
    df2.name='2nd30'
    df3=df.iloc[61:91,:]
    df3.name='3rd30'
    df4=df.iloc[91:121,:]
    df4.name='4th30'
    df5=df.iloc[121:151,:]
    df5.name='5th30'
    df6=df.iloc[151:,:]
    df6.name='6th30'
   
    
    
    for k in [df1,df2,df3,df4,df5,df6]:
        k_str=k.name
        labels=list(k.index)
        x=np.arange(len(labels))
        fig,ax=plt.subplots(figsize=(25,15))
        
        
        rects1=ax.bar(x-0.2,k.iloc[:,0],width,label='2018-19', color='blue')
        rects2=ax.bar(x,k.iloc[:,1],width,label='2019-20', color='turquoise')
        rects3=ax.bar(x+0.2,k.iloc[:,2],width,label='2020-21', color='red')
        ax.legend(fontsize=16)
        ax.set_title(i, fontsize=20)
        ax.tick_params(labelsize=20)
        
        ax.set_xticks(x)
        
        ax.set_xticklabels(k.index, rotation=90)
        plt.tight_layout()
        plt.savefig('./images/bar/bar_{}_{}.jpg'.format(i,k_str))
        plt.close()
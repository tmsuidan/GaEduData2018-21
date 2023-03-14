# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:11:52 2023

@author: talam
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    width=0.3
    
    
    df1=df.iloc[0:60,:]
    df1.name='1st60'
    df2=df.iloc[61:121,:]
    df2.name='2nd60'
    df3=df.iloc[122:,:]
    df3.name='3rd60'
    
    for k in [df1,df2,df3]:
        k_str=k.name
        labels=list(k.index)
        x=np.arange(len(labels))
        fig,ax=plt.subplots(figsize=(40,15))
        
        
        rects1=ax.bar(x-width/3,k.iloc[:,0],width,label='2018-19', color='green')
        rects2=ax.bar(x,k.iloc[:,1],width,label='2019-20', color='blue')
        rects3=ax.bar(x+width/3,k.iloc[:,2],width,label='2020-21', color='red')
        ax.legend()
        ax.set_title(i)
        ax.tick_params(labelsize=20)
        
        ax.set_xticks(x)
        
        ax.set_xticklabels(k.index, rotation=90)
        plt.tight_layout()
        plt.savefig('./images/bar_{}_{}.jpg'.format(i,k_str))
        plt.close()
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:18:52 2020

@author: User
"""
#%%

def get_unique_num(X_array):

    unique_array = np.unique(X_array)
    
    return len(unique_array)
    
    
#%%

import pandas as pd
import numpy as np

data = pd.read_csv(r'./segmentation.data',sep=',', engine = 'python', names = np.arange(20))

data_one =  pd.read_csv(r'./segment_one.data', sep=' ',engine = 'python', names = np.arange(20))


data_one.drop(columns = 2)

data_one['19'] = (data_one[data_one.columns[-1]] == 1) *1



data_one.to_csv(r'D:\Research_for_imbalanced_data\data_handling\sampling\dataset\raw/segment.csv', index= False)
#%%

data1 = pd.read_csv(r'./Data/cm.csv')


data1[data1.columns[np.setdiff1d(np.arange(len(data1.columns)), np.array([15]))]]

data2 = pd.read_csv(r'./Data/kc.csv')
data3 = pd.read_csv(r'./Data/oil.csv')
data3 = data3[data3.columns[::-1]]
data3 = data3[data3.columns[np.setdiff1d(np.arange(len(data3.columns)), np.array([ 3, 26]))]]
data4 = pd.read_csv(r'./Data/pc.csv')

data1.to_csv(r'./sampling/dataset/raw/cm.csv',index=False)
data2.to_csv(r'./sampling/dataset/raw/kc.csv',index=False)
data3.to_csv(r'./sampling/dataset/raw/oil.csv',index=False)
data4.to_csv(r'./sampling/dataset/raw/pc.csv',index=False)

#%%

cols = ['age',
'workclass',
'fnlwgt',
'education',
'educationNum',
'maritalStatus',
'occupation',
'relationship',
'race',
'sex',
'capitalGain',
'capitalLoss',
'hoursPerWeek',
'nativeCountry',
]

del_col = ['age',
           'fnlwgt',
           'educationNum',
           'capitalGain',
           'capitalLoss',
           'hoursPerWeek',
           'nativeCountry'
           ]

data5_train = pd.read_csv(r'./Data/ADA/ada_train.csv', names = cols)
data5_test = pd.read_csv(r'./Data/ADA/ada_test.csv', names = cols)
data5_vali = pd.read_csv(r'./Data/ADA/ada_valid.csv', names = cols)

#data5_pre = pd.read_csv(r'./Data/ADA_/ada_train.data',names = cols,sep=' ')
#%%
check = pd.concat([data5_train[del_col], data5_test[del_col], data5_vali[del_col]])

no_uni = check[(check.nativeCountry != '?') & (check.nativeCountry != 'United-States')]
uni = check[check.nativeCountry =='United-States']
final = pd.concat([no_uni,uni.sample(1000)])
final.nativeCountry = (final.nativeCountry == 'China') * 1
final.to_csv(r'./sampling/dataset/raw/pc.csv',index = False)
#%%



data6_train = pd.read_csv(r'./Data/HIVA/hiva_train.data')
data6_test = pd.read_csv(r'./Data/ADA/ada_test.csv', names = cols)
data6_vali = pd.read_csv(r'./Data/ADA/ada_valid.csv', names = cols)

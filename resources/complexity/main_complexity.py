# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:15:21 2024

@author: dohyeon
"""


import os
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from cal_complexity import complexity_neighbors as com_neigh, complexity_feature_based as com_feat


random_states = np.sort([0, 4, 18, 19, 32, 38, 44, 49, 54, 58, 61, 65, 69, 70, 71, 74, 76, 83, 88, 97, 20, 31, 59, 39, 23, 82, 1, 30, 26, 86])
dataset_fnames = os.listdir(r'../dataset/prep')
minority_label = 1
majority_label = 0
k_value = 5



for fname in dataset_fnames:
    start_time = time.time()
    temp_df = pd.read_csv(r'../dataset/prep/%s'%(fname))

    X = temp_df.drop(columns=temp_df.columns[-1]).values
    y = temp_df[temp_df.columns[-1]].values

    result_df = pd.DataFrame(columns=['dataset','rel_rs','rs','fold','N1','N2','F1','F2'])
    i=0
    for rs_idx, rs in enumerate(random_states):
        rs_time = time.time()
        # 5-fold 교차 검증 설정
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    
        # 교차 검증 시작
        for fidx, (train_index, _) in enumerate(kf.split(X, y)):
            X_train = X[train_index]
            y_train = y[train_index]
            
            # 각 fold에서 정규화 (StandardScaler 사용)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
    
            X_min = X_train[y_train == minority_label, :]
            X_maj = X_train[y_train == majority_label, :]
    
            neighbors = NearestNeighbors(n_neighbors=k_value + 1, algorithm='auto').fit(X_train)
            neighbors1 = NearestNeighbors(n_neighbors=k_value + 1, algorithm='auto').fit(X_min)
            neighbors2 = NearestNeighbors(n_neighbors=k_value + 1, algorithm='auto').fit(X_maj)
    
    
            _, min_all_ind = neighbors.kneighbors(X_min, return_distance=True)
    
            min_min_dist, _ = neighbors1.kneighbors(X_min, return_distance=True)
    
            min_maj_dist, _ = neighbors2.kneighbors(X_min, return_distance=True)
    
    
            neigh_class = com_neigh(min_all_ind = min_all_ind[:,1:],
                                    min_min_dist = min_min_dist[:,1:],
                                    min_maj_dist = min_maj_dist[:,1:],
                                    y_array = y_train,
                                    k_value = k_value,
                                    minority_label = minority_label)
    
            feat_class = com_feat(X_array = X_train,
                                  y_array = y_train,
                                  minority_label = minority_label)
    
            neigh_factor = neigh_class.main()
            feat_factor = feat_class.main()
            
            whole_factor = neigh_factor + feat_factor
    
            result_df.loc[i] = [fname.split('.')[0], rs_idx, rs, fidx]+ whole_factor
            i+=1
        if time.time()-rs_time > 30:
            print(fname.split('.')[0], rs_idx, rs, time.time() - rs_time)
    print(fname.split('.')[0], time.time() - start_time)
    result_df.to_csv(r'%s'%(fname), index=False)




    
































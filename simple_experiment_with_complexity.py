# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:51:26 2024

@author: dohyeon
"""
import os

import time

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, recall_score, precision_score,roc_auc_score
from imblearn.metrics import geometric_mean_score,sensitivity_specificity_support

from oversampling import SampleGenerator
from classifier import With_Boost






def get_metric(y__pred, y__proba, test__y):
    acc=np.sum(test__y == y__pred)/len(y__pred)
    f1 = f1_score(test__y, y__pred, average='binary')
    pre=precision_score(test__y, y__pred, zero_division=0)
    recall=recall_score(test__y, y__pred, zero_division=0)
    spec = sensitivity_specificity_support(test__y, y__pred, average='binary')[1]
    auc_score=roc_auc_score(test__y, y__proba[:,-1])
    g_mean = geometric_mean_score(test__y, y__pred, average='binary')

    return [acc, f1, pre, recall, auc_score, g_mean, spec]




if __name__ == '__main__' :

    # 데이터 생성: 불균형한 이진 분류 데이터셋


    random_states = np.sort([0, 4, 18, 19, 32, 38, 44, 49, 54, 58, 61, 65, 69, 70, 71, 74, 76, 83, 88, 97, 20, 31, 59, 39, 23, 82, 1, 30, 26, 86])
    dataset_dir = r'resources/dataset/prep'
    dataset_fnames = os.listdir(dataset_dir)

    minority_label = 1
    majority_label = 0
    k_value = 5

    classifiers = ['smote_boost','ramo_boost','wot_boost']
    #Ts = np.arange(0.2, 1.2, 0.2)


    for fname in dataset_fnames:
        start_time = time.time()
        temp_df = pd.read_csv(r'%s/%s'%(dataset_dir, fname))
    
        X = temp_df.drop(columns=temp_df.columns[-1]).values
        y = temp_df[temp_df.columns[-1]].values
    
        result_df = pd.DataFrame(columns=['dataset','rel_rs','rs','fold','model','sampling_ratio',
                                          'acc', 'f1', 'pre', 'recall', 'auc_score', 'g_mean', 'spec'])

        complexity_df = pd.read_csv(r'resources\complexity\%s'%(fname)).groupby(['dataset','rel_rs','rs','fold']).mean()

        i=0
        for rs_idx, rs in enumerate(random_states):
            rs_time = time.time()
            # 5-fold 교차 검증 설정
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        
            # 교차 검증 시작
            for fidx, (train_index, test_index) in enumerate(kf.split(X, y)):
                fold_time = time.time()
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                for uni_model in classifiers:

                    sampling = SampleGenerator(sampling_method=uni_model)
    
                    sampling.fit(X=X_train, y=y_train, minority_class=minority_label)

                    Ts = complexity_df.loc[fname.split('.')[0], rs_idx, rs, fidx]
                    for uni_complexity, uni_ratio in Ts.items():

                        boosting = With_Boost()

                        num_majority = len(y_train[y_train==majority_label])
                        num_minority = len(y_train[y_train==minority_label])
                        n_samples = int((num_majority - num_minority)*uni_ratio)

                        boosting.fit(X=X_train, y=y_train,
                                     n_samples=n_samples, smote=sampling)
                    
                        y_pred = boosting.predict(X_test)
                        y_proba = boosting.predict_proba(X_test)
                    
                        metrics = get_metric(y_pred, y_proba, y_test)
                    
                        result_df.loc[i] = [fname.split('.')[0], rs_idx, rs, fidx, uni_model, uni_complexity] + metrics
                        i+=1

                        boosting = None
                    sampliing = None
                if time.time() - fold_time > 30:
                    print(fname.split('.')[0], rs_idx, rs, fidx, time.time() - fold_time)
            if time.time()-rs_time > 30:
                print(fname.split('.')[0], rs_idx, rs, time.time() - rs_time)
        print(fname.split('.')[0], time.time() - start_time)
        result_df.to_csv(r'result/prediction_performance/proposed/others/%s'%(fname), index=False)
    


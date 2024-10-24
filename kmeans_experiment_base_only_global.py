# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:40:19 2024

@author: dohyeon
"""


import os
from kmeans_classifier import With_Boost_Clustering as classifier
import cal_cluster_complexity as lc
from kmeans_smote import GlobalLocalSMOTE as smote
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import itertools
import time
from sklearn.metrics import f1_score, recall_score, precision_score,roc_auc_score
from imblearn.metrics import geometric_mean_score,sensitivity_specificity_support
# 모델 파라미터에서 복잡도 파일을 읽어 N1, N2 값들을 가져옴




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



    random_states = np.sort([0, 4, 18, 19, 32, 38, 44, 49, 54, 58, 61, 65, 69, 70, 71, 74, 76, 83, 88, 97, 20, 31, 59, 39, 23, 82, 1, 30, 26, 86])
    dataset_dir = r'resources/dataset/prep'
    dataset_fnames = os.listdir(dataset_dir)


    ir_list = [1, np.inf]
    n_clusters_list = [2, 20, 50, 100, 250, 500]
    minority_label = 1
    majority_label = 0
    k_value = 5
    c_names = np.around(np.arange(0.2, 1.2, 0.2),1)


    for fname in dataset_fnames:
        start_time = time.time()
        temp_df = pd.read_csv(r'%s/%s'%(dataset_dir, fname))

        X = temp_df.drop(columns=temp_df.columns[-1]).values
        y = temp_df[temp_df.columns[-1]].values
    
        result_df = pd.DataFrame(columns=['dataset','rel_rs','rs','fold','cls_try','sampling_ratio','num_clusters','ir',
                                          'acc', 'f1', 'pre', 'recall', 'auc_score', 'g_mean', 'spec'])
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

                # 폴드 데이터를 준비
                #fold_T = np.append(fold_T, 1)
                fold_trn_minidx = np.where(y_train == minority_label)[0]
                fold_trn_majidx = np.where(y_train != minority_label)[0]
            
                fold_x_min = X_train[fold_trn_minidx]
                fold_x_maj = X_train[fold_trn_majidx]
            
                fold_min_num = len(fold_x_min)
                fold_maj_num = len(y_train) - fold_min_num
                fold_size_diff = fold_maj_num - fold_min_num
            
                # NearestNeighbors로 학습
                fold_neigh = NearestNeighbors(n_neighbors=k_value + 1).fit(fold_x_min)
                fold_neigh1 = NearestNeighbors(n_neighbors=k_value + 1).fit(fold_x_maj)
                fold_neigh2 = NearestNeighbors(n_neighbors=k_value + 1).fit(X_train)
            
                dist, ind = lc.get_neighbor_distances_indices(fold_neigh, fold_x_min)
                dist1, ind1 = lc.get_neighbor_distances_indices(fold_neigh1, fold_x_min)
                dist2, ind2 = lc.get_neighbor_distances_indices(fold_neigh2, fold_x_min)
            
                N1_base_intra = np.sum(dist, axis=1)
                N1_base_extra = np.sum(dist1, axis=1)
                N2_base = np.sum(y_train[ind2] != 1, axis=1)
            
                smote_list = []
                
                # 클러스터별로 SMOTE 적용
                for n_cls, ir in itertools.product(n_clusters_list, ir_list):
                    if len(y_train) >= n_cls:
                        smote_object = smote(n_clusters=n_cls, ir=ir, k_neighbors=k_value, neigh=fold_neigh)
                        smote_object.fit(X_train, y_train)
                        if smote_object.test_ok:
                            smote_list.append(smote_object)
                
                if not smote_list:
        
                    print('no_smote_list')
                    continue  # SMOTE를 적용할 클러스터가 없으면 다음 폴드로 넘어감
            
                # SMOTE 결과를 바탕으로 부스팅 수행
                for uni_ok_smote in smote_list:
                    for p_n, p_val in zip(c_names, c_names):
                        base_size = int(fold_size_diff * p_val)
                        global_cls_size = np.around(uni_ok_smote.cls_sparsity * base_size)

                        # checking valid 조건
                        if np.all(global_cls_size > 0):
                            uni_case = 'global'
                            learner = classifier()
                            learner.fit(X=X_train, y=y_train,
                                        n_samples=base_size, smote=uni_ok_smote,
                                        is_global=True, cls_T=None)

                            y_pred = learner.predict(X_test)
                            y_proba = learner.predict_proba(X_test)
        
                            # 평가 결과 계산
                            s_val = get_metric(y_pred, y_proba, y_test)
                            s_info = [fidx, p_n, uni_case, uni_ok_smote.n_clusters, uni_ok_smote.ir]
                            # 결과 저장
                            result_df.loc[i] = [fname.split('.')[0], rs_idx, rs, fidx,
                                                uni_case,p_n,uni_ok_smote.n_clusters,
                                                uni_ok_smote.ir] + s_val
                            i+=1
            if time.time()-rs_time > 30:
                print(fname.split('.')[0], rs_idx, rs, time.time() - rs_time)
        print(fname.split('.')[0], time.time() - start_time)
        result_df.to_csv(r'result/prediction_performance/base/kmeans/%s'%(fname), index=False)

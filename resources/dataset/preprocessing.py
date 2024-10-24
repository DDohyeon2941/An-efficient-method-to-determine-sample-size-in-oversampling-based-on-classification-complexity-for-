# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:21:52 2024

@author: dohyeon
"""



import os
import pandas as pd
import numpy as np


def load_dataset(root_dir, file_name):
    return pd.read_csv(os.path.join(root_dir, file_name))

def drop_samples(raw_df):
    data = raw_df.drop_duplicates(keep='first').reset_index(drop = True)
    data = data.dropna(axis = 0, how = 'any').reset_index(drop = True)
    return data

def check_binary(y_arr, minority_label=1, majority_label=0):
    labels, counts = np.unique(y_arr, return_counts=True)
    num_class = len(labels)
    assert num_class == 2

    min_lab, maj_lab = labels[np.argsort(counts)]

    new_y = y_arr.copy()
    if not min_lab == minority_label:
        new_y[np.where(new_y==min_lab)[0]] = minority_label

    if not maj_lab == majority_label:
        new_y[np.where(new_y==maj_lab)[0]] = majority_label

    return new_y


def main(root_dir, file_name):
    dropped_arr = drop_samples(load_dataset(root_dir, file_name)).values
    x_arr, y_arr = dropped_arr[:,:-1], dropped_arr[:,-1]
    new_y_arr = check_binary(y_arr)
    return pd.DataFrame(data=np.hstack((x_arr, new_y_arr[:,np.newaxis])))


def get_ir(y_arr, minority_label=1, majority_label=0):
    return len(y_arr[y_arr == majority_label]) / len(y_arr[y_arr == minority_label])
#%%
if __name__ == '__main__' :

    dataset_list_dir = r'D:\project_repository\oversampling_size_determination\experiment\analysis\commons\resources\dataset\preprocess_16'
    base_dir = r'raw'
    save_dir = r'prep'
    dataset_list = os.listdir(dataset_list_dir)

    raw_li = []
    prep_li=[]
    for fname in os.listdir(dataset_list_dir):
        raw_df = pd.read_csv(r'raw\%s'%(fname))
        temp_df=main(base_dir,fname)

        raw_li.append(raw_df)
        prep_li.append(temp_df)
        temp_df.to_csv(os.path.join(save_dir, fname), index=False)

#%%

idx_val = 15
print(dataset_list[idx_val])
print(raw_li[idx_val])
print(prep_li[idx_val])
print(prep_li[idx_val].shape)
print(get_ir(prep_li[idx_val].values[:,-1]))


#%%

data_df = pd.read_csv(r'raw\ecoli.csv', index_col=0)
data_df.to_csv(r'ecoli.csv', index=False)


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:15:19 2024

@author: dohyeon
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__' :


    complexity_dir = r'resources/complexity'
    dataset_dir = r'resources/dataset/prep'
    dataset_fnames = os.listdir(dataset_dir)

    result_df = pd.DataFrame()
    for fname in dataset_fnames:
        start_time = time.time()
        temp_df = pd.read_csv(r'%s/%s'%(complexity_dir, fname))
        result_df = pd.concat([result_df, temp_df.groupby(['dataset']).mean()[['F1','F2','N1','N2']].reset_index()])
    result_df = result_df.reset_index(drop=True)
    #result_df.to_csv(r'average_complexity.csv',index=False)

    des_df = pd.read_csv(r'resources\dataset\Dataset Description.csv')

    result_df.loc[:, 'IR'] = des_df['IR']
    result_df.loc[:, 'dataset'] = des_df['Data']
    result_df = result_df.sort_values(by='IR').reset_index(drop=True)

    #%%
    fig1, axes1 = plt.subplots(1,1)

    axes1.bar(result_df['dataset'],result_df['IR'])
    axes2 = axes1.twinx()
    axes2.scatter(np.arange(result_df.shape[0]), result_df['F1'], marker='o', c='b', alpha=0.5)
    axes2.scatter(np.arange(result_df.shape[0]), result_df['F2'], marker='s', c='b', alpha=0.5)
    axes2.scatter(np.arange(result_df.shape[0]), result_df['N1'], marker='d', c='purple', alpha=0.5)
    axes2.scatter(np.arange(result_df.shape[0]), result_df['N2'], marker='p', c='purple', alpha=0.5)
    axes2.scatter(np.arange(result_df.shape[0]), result_df[['F1','F2','N1','N2']].mean(axis=1), marker='X', c='k')

    axes1.tick_params(axis='x', labelrotation=90, labelsize=12)

    axes1.set_ylabel('IR', fontsize=15)
    axes2.set_ylabel('Classification Complexity', fontsize=15)

    #%%


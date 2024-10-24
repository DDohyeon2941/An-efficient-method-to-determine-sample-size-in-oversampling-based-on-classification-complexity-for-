# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:09:01 2024

@author: dohyeon
"""

import numpy as np
import os
import pandas as pd

class complexity_neighbors(object):
    
    def __init__(self, min_all_ind, min_min_dist, min_maj_dist, y_array , k_value = 5, minority_label = 1):
        
        self.ind= min_all_ind
        self.dist1 = min_min_dist
        self.dist2 = min_maj_dist
        self.y_array = y_array
        
        self.k_value = k_value
        self.minority_label = minority_label
        self.eps = 0.2
        
        
    def main(self):
        
        N1 = self.get_N1()
        N2 = self.get_N2()
        
        return [N1, N2]
    
    ###여기서 부터 N2, N3
    
    def get_N1(self):
        '''
        definition : ratio_of_intra_extra_ditance
        '''
        
        min_min = self.dist1[:,:self.k_value]
        min_maj = self.dist2[:,:self.k_value]
        
        intra_indi = np.sum(min_min, axis=1)
        
        extra_indi = np.sum(min_maj, axis =1)
        
        if not np.where(extra_indi == 0)[0].shape == (0,):
            
            intra_extra = np.sum(intra_indi) / np.sum(extra_indi)
            
        else:
            
            non_zero_idx = np.where(extra_indi >0)[0]
            not_zero_val = np.min(extra_indi[non_zero_idx]) * self.eps
            np.place(extra_indi, extra_indi == 0, not_zero_val)
        
            intra_extra = np.sum(intra_indi) / np.sum(extra_indi)
        
        N1 = intra_extra / (1 + intra_extra)
        
        return N1
    
    def get_N2(self):
        
        N2_neighbors = self.ind[:, :self.k_value]
        
        n_error = np.sum(self.y_array[N2_neighbors] != self.minority_label)
        
        N2 = n_error / (self.ind.shape[0] * self.ind.shape[1])
        
        return N2
    
class complexity_feature_based(object):
    
    def __init__(self, X_array, y_array, minority_label = 1):

        self.X_array = X_array
        self.y_array = y_array
        self.minority_label = minority_label
        self.eps = 0.2
    
    
    def split_X_array(self, X_array, y_array):
    
        class0 = X_array [y_array != self.minority_label]
        class1 = X_array [y_array == self.minority_label]
        
        return class0, class1
    
    def main(self):
        
        F1 = 1 / (1 + np.nanmax(self.get_f1_array()))
        F2 = self.min_f2()

        return [F1, F2]

    def get_f1_array(self):
        
        f_array = np.apply_along_axis(self.f1, 0, self.X_array, self.y_array)
        
        f_array = np.nan_to_num(f_array, nan = np.nanmin(f_array) * self.eps)
        
        return f_array
    
    def f1(self, X_array, y_array):
        
        array_0, array_1 = self.split_X_array(X_array, y_array)
        
        m0 = np.mean(array_0)
        m1 = np.mean(array_1)
        
        std0 = np.std(array_0)
        std1 = np.std(array_1)
        
        #분자
        numerator = (m1 - m0) ** 2
        
        #분모
        denominator = std1**2 + std0 **2
            
        if denominator == 0:
            
            f = np.nan
        
        else:
        
            f =  numerator / denominator
        
        return f

        
    def min_f2(self):
        
        f_array = self.get_f2_array(self.X_array, self.y_array)
        f_array_ = [float('inf') if len(np.unique(self.X_array[: ,idx])) == 0 else f_array[idx] for idx in range(len(f_array))]
        
        min_idx = np.argmin(f_array_)
        min_value = f_array_[min_idx]
            
        return min_value / len(self.X_array)
    
    def get_f2_array(self, X_array, y_array):
        
        f_array = np.apply_along_axis(self.f2, 0, X_array, y_array)
        
        return f_array
    
    
    def f2(self, X_array, y_array):
        
        minmax, maxmin = self.maxmin_minmax(X_array, y_array)
        
        num_sample = np.sum((maxmin <= X_array) & (minmax >= X_array))
        
        return num_sample
    
    
    def maxmin_minmax(self,X_array, y_array):
    
        array_0 , array_1 = self.split_X_array(X_array, y_array)
        
        min0, max0 = np.min(array_0), np.max(array_1)
        min1, max1 = np.min(array_1), np.max(array_1)
        
        maxmin = np.max([min0, min1])
        minmax = np.min([max0, max1])
    
        return minmax, maxmin
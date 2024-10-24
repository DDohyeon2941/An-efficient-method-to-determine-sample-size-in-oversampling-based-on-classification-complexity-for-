# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:09:24 2024

@author: dohyeon
"""


import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import time
from sklearn.cluster import AgglomerativeClustering
import warnings

class GlobalLocalSMOTE:
    """
    Implementation of k-means Synthetic Minority Over-Sampling Technique (SMOTE).

    Parameters
    ----------
    n_clusters : int
        Number of clusters (k-means)
    ir : float
        Imbalance ratio threshold
    k_neighbors : int
        Number of nearest neighbors.
    random_state : int or None
        Seed for random number generation.

    References
    ----------
    Douzas, Georgios, Fernando Bacao, and Felix Last. "Improving imbalanced learning through a heuristic
    oversampling method based on k-means and SMOTE." Information Sciences 465 (2018): 1-20.
    """

    def __init__(self, n_clusters=5, ir=1, k_neighbors=5, neigh=None, random_state=None):
        self.n_clusters = n_clusters
        self.ir = ir
        self.k = k_neighbors
        self.neigh = neigh
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=-1)

    def local_sample(self, n_samples, cls_T=None, random_state=None):
        """Generate synthetic samples for local clusters."""
        np.random.seed(seed=random_state or self.random_state)

        S = np.empty(shape=(0, self.n_features))
        assert len(cls_T) == len(self.cls_sparsity), "Mismatched length: cls_T and cls_sparsity"

        cls_sample_size = np.nan_to_num(np.round(n_samples * self.cls_sparsity * cls_T)).astype(int)
        assert np.all(cls_sample_size >= 0), "Negative sample size detected"
        self.cls_sample_size = cls_sample_size

        for c, s in zip(self.sel_cls, cls_sample_size):
            if s == 0:
                continue

            sel_ind = np.intersect1d(self.min_all_idx, c)
            selX = self.X[sel_ind]
            j = np.random.randint(0, selX.shape[0], size=s)

            # Nearest neighbors excluding the sample itself
            nn = self.neigh.kneighbors(selX[j], return_distance=False)[:, 1:]
            nn_index = nn[np.arange(s), np.random.choice(self.k, size=s)]

            dif = self.Xmin[nn_index] - selX[j]
            gap = np.random.random(s)

            S = np.concatenate((S, selX[j] + gap[:, None] * dif))

        return S

    def global_sample(self, n_samples, random_state=None):
        """Generate synthetic samples for global clusters."""
        np.random.seed(seed=random_state or self.random_state)

        S = np.empty(shape=(0, self.n_features))
        cls_sample_size = np.round(n_samples * self.cls_sparsity).astype(int)

        # Adjust sample size to match n_samples
        while sum(cls_sample_size) != n_samples:
            if sum(cls_sample_size) < n_samples:
                cls_sample_size[np.argmin(cls_sample_size)] += 1
            else:
                cls_sample_size[np.argmax(cls_sample_size)] -= 1

        assert np.all(cls_sample_size >= 0), "Negative sample size detected"

        for c, s in zip(self.sel_cls, cls_sample_size):
            if s == 0:
                continue

            sel_ind = np.intersect1d(self.min_all_idx, c)
            selX = self.X[sel_ind]
            j = np.random.randint(0, selX.shape[0], size=s)

            nn = self.neigh.kneighbors(selX[j], return_distance=False)[:, 1:]
            nn_index = nn[np.arange(s), np.random.choice(self.k, size=s)]

            dif = self.Xmin[nn_index] - selX[j]
            gap = np.random.random(s)

            S = np.concatenate((S, selX[j] + gap[:, None] * dif))

        return S

    def get_cls_ir(self, num_min, num_maj):
        """Calculate class imbalance ratio."""
        return (num_maj + 1) / (num_min + 1)

    def get_candi_ir(self, minor_idx, indi_cls_idx):
        """Get imbalance ratio for a given class candidate."""
        num_min = len(np.intersect1d(minor_idx, indi_cls_idx))
        num_maj = len(indi_cls_idx) - num_min
        return num_min, self.get_cls_ir(num_min, num_maj)

    def get_sel_cls(self, minor_idx, candi_list, ir):
        """Select clusters satisfying imbalance ratio threshold."""
        for indi_cls_idx in candi_list:
            if isinstance(indi_cls_idx, set):
                indi_cls_idx = np.array(list(indi_cls_idx))

            num_min, candi_ir = self.get_candi_ir(minor_idx, indi_cls_idx)
            if candi_ir < ir and num_min > 1:
                yield indi_cls_idx

    def get_alternative_cls(self, minor_idx, candi_list):
        """Select clusters with at least two minority samples."""
        for indi_cls_idx in candi_list:
            num_min = len(np.intersect1d(minor_idx, indi_cls_idx))
            if num_min > 1:
                yield indi_cls_idx

    def remove_all_major(self, major_idx, cls_list):
        """Remove clusters containing only majority samples."""
        for indi_cls_idx in cls_list:
            if isinstance(indi_cls_idx, set):
                indi_cls_idx = np.array(list(indi_cls_idx))

            if len(np.intersect1d(major_idx, indi_cls_idx)) != len(indi_cls_idx):
                yield indi_cls_idx

    def make_cls_dict_list(self, cls_object):
        """Create a dictionary of clusters."""
        unique_labels = np.unique(cls_object.labels_)
        for label in unique_labels:
            yield set(np.where(cls_object.labels_ == label)[0])

    def fit(self, X, y, minority_target=1, m=None):
        """Fit the model and prepare clusters for SMOTE."""
        self.X = X
        self.y = y
        self.minority_target = minority_target or min(Counter(y), key=Counter(y).get)
        self.Xmin = X[y == self.minority_target]
        self.n_features = X.shape[1]
        self.min_all_idx = np.where(y == self.minority_target)[0]
        self.maj_all_idx = np.where(y != self.minority_target)[0]

        self.kmeans.fit(X)
        cls_list = list(self.make_cls_dict_list(self.kmeans))
        candi_cls = [*self.remove_all_major(self.maj_all_idx, cls_list)]
        sel_cls = [*self.get_sel_cls(self.min_all_idx, candi_cls, self.ir)]

        if not sel_cls:
            self.no_cls = True
            sel_cls = [*self.get_alternative_cls(self.min_all_idx, candi_cls)]
        self.sel_cls = sel_cls

        if sel_cls:
            _, m = X.shape if m is None else m
            avg_dist_n = np.array([*self.get_cls_avgdist(sel_cls, X)])
            density, m = self._compute_density(avg_dist_n, m)

            self.cls_sparsity = np.array(1 / density) / sum(1 / density)
            self.test_ok = True
        else:
            print('All clusters contain less than 2 minority samples')
            self.test_ok = False

        return self

    def _compute_density(self, avg_dist_n, m):
        """Helper function to compute cluster density."""
        tag = True
        while tag:
            numerator = avg_dist_n[:, 0]
            denominator = np.power(avg_dist_n[:, 1], m) + 1e-10
            density = numerator / denominator
            if not np.any(density == 0.0):
                tag = False
            else:
                m //= 2
        return density, m

    def get_cls_avgdist(self, sel_cls, X):
        """Calculate average distance within selected clusters."""
        for cls_idx in sel_cls:
            selX = X[np.intersect1d(self.min_all_idx, cls_idx)]
            avg_dist = np.mean(pdist(selX))
            yield len(selX), avg_dist


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:17:43 2024

@author: dohyeon
"""

import numpy as np

def get_neighbor_distances_indices(neighbor_classifier, data_samples):
    """
    Get the distances and indices of the nearest neighbors for the given data samples.
    Skips the first neighbor (itself).
    """
    distances, indices = neighbor_classifier.kneighbors(data_samples, return_distance=True)
    return distances[:, 1:], indices[:, 1:]


def get_intersection_indices(base_indices, selected_indices_):
    """
    Find the intersection of two index arrays and return the indices of the intersection.
    """
    #print(np.intersect1d(base_indices, selected_indices_, return_indices=True))
    intersected_indices = np.intersect1d(base_indices, selected_indices_, return_indices=True)[1]
    return intersected_indices


def compute_N1(intra_class_distances, extra_class_distances, selected_indices, epsilon=0.2):
    """
    Compute the N1 metric, comparing intra-class and extra-class distances.
    """
    intra_sum = np.sum(intra_class_distances[selected_indices])
    extra_values = extra_class_distances[selected_indices]

    if np.all(extra_values != 0):
        intra_extra_ratio = intra_sum / np.sum(extra_values)
    else:
        min_non_zero = np.min(extra_values[extra_values > 0]) * epsilon
        np.place(extra_values, extra_values == 0, min_non_zero)
        intra_extra_ratio = intra_sum / np.sum(extra_values)

    return intra_extra_ratio / (1 + intra_extra_ratio)


def compute_N2(different_class_neighbor_count, selected_indices, k_value=5):
    """
    Compute the N2 metric based on the count of neighbors whose labels differ 
    from the minority class.
    """
    return np.sum(different_class_neighbor_count[selected_indices]) / (k_value * len(selected_indices))


def calculate_T_values(selected_cluster_list, metric_mode, minority_class_indices, neighbor_data):
    """
    Calculate T values (N1, N2, or balance) for a given list of selected clusters.

    Args:
        selected_cluster_list: List of cluster indices for which T values will be calculated.
        metric_mode: A string representing the mode ('N1', 'N2', or 'balance').
        minority_class_indices: Indices of the minority class samples.
        neighbor_data: Tuple containing necessary data for T value calculation 
                       (intra-class distances, extra-class distances, different class counts).

    Returns:
        An array of calculated T values based on the mode.
    """
    T_values = []
    
    for cluster_indices in selected_cluster_list:
        #print(minority_class_indices, cluster_indices)
        selected_indices = get_intersection_indices(minority_class_indices, cluster_indices)

        if metric_mode == 'N1':
            T_values.append(compute_N1(neighbor_data[0], neighbor_data[1], selected_indices))
        elif metric_mode == 'N2':
            T_values.append(compute_N2(neighbor_data[2], selected_indices))
        elif metric_mode == 'balance':
            T_values.append(1)

    return np.array(T_values)

























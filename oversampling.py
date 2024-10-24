# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:07:16 2024

@author: dohyeon
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class SMOTESampler:
    def __init__(self, 
                 nearest_neighbors_1, 
                 nearest_neighbors_2, 
                 labels, 
                 method='origin', 
                 minority_class=1):
        """
        SMOTESampler 클래스 초기화

        Parameters:
        nearest_neighbors_1: 첫 번째 최근접 이웃 정보
        nearest_neighbors_2: 두 번째 최근접 이웃 정보
        labels: 레이블 배열
        method: SMOTE 적용 방법 (기본값은 'origin')
        minority_class: 소수 클래스 레이블 (기본값은 1)
        """
        self.nearest_neighbors_1 = nearest_neighbors_1
        self.nearest_neighbors_2 = nearest_neighbors_2
        self.labels = labels
        self.method = method
        self.minority_class = minority_class

        # method는 'origin'만 사용 가능
        assert self.method == 'origin'

    def main(self, version, num_samples, minority_samples, sample_weights):
        """
        샘플 생성 메서드

        Parameters:
        version: 'sampling' 또는 'weight'
        num_samples: 생성할 샘플 수
        minority_samples: 소수 클래스 샘플 배열
        sample_weights: 샘플 가중치

        Returns:
        생성된 샘플과 해당 인덱스
        """
        assert version in ['sampling', 'weight']

        if version == 'sampling':
            synthetic_indices = self._get_sampling_indices(num_samples, sample_weights)
        else:
            synthetic_indices = self._get_weighted_indices(num_samples, sample_weights)
        
        if synthetic_indices is None:
            return None, None
        else:
            synthetic_samples = self._create_synthetic_samples(synthetic_indices, num_samples, minority_samples)
            return synthetic_samples, synthetic_indices

    def _get_minority_indices(self, sample_weights=None):
        """
        소수 클래스 인덱스와 가중치를 반환하는 함수
        """
        num_minority_samples, num_neighbors = self.nearest_neighbors_2.shape

        # 소수 클래스 인덱스
        minority_indices = np.arange(num_minority_samples)

        if sample_weights is None:
            weights = np.ones(num_minority_samples) / num_minority_samples
        else:
            if len(minority_indices) == 0:
                return None, None
            weights = np.squeeze(normalize(sample_weights[minority_indices].reshape(1, -1), axis=1, norm='l1'))

        return minority_indices, weights

    def _get_sampling_indices(self, num_samples, sample_weights=None):
        """
        샘플링 기반 synthetic 샘플 인덱스를 생성
        """
        minority_indices, weights = self._get_minority_indices(sample_weights)
        
        if minority_indices is None or len(weights) == 0 or num_samples == 0:
            return None

        if len(weights) == 1:
            synthetic_indices = minority_indices.tolist() * num_samples
        else:
            synthetic_indices = np.random.choice(minority_indices, size=num_samples, p=weights)
        
        return synthetic_indices

    def _get_weighted_indices(self, num_samples, sample_weights=None):
        """
        가중치 기반 synthetic 샘플 인덱스를 생성
        """
        minority_indices, weights = self._get_minority_indices(sample_weights)

        if minority_indices is None or num_samples == 0 or np.all(weights == 0):
            return None

        # 각 인덱스에 대해 생성할 샘플 수 계산
        num_synthetic_samples = np.around(num_samples * weights).astype(int)

        # 부족한 경우 샘플 추가
        if np.sum(num_synthetic_samples) < num_samples:
            extra_samples = num_samples - np.sum(num_synthetic_samples)
            extra_indices = np.random.choice(len(minority_indices), size=extra_samples, p=weights, replace=False)
            num_synthetic_samples[extra_indices] += 1
        # 초과한 경우 샘플 삭제
        elif np.sum(num_synthetic_samples) > num_samples:
            excess_samples = np.sum(num_synthetic_samples) - num_samples
            valid_indices = np.where(num_synthetic_samples > 0)[0]
            remove_indices = np.random.choice(valid_indices, size=excess_samples, p=(weights[valid_indices] / np.sum(weights[valid_indices])), replace=False)
            num_synthetic_samples[remove_indices] -= 1

        temp_indices = np.repeat(range(len(minority_indices)), num_synthetic_samples)
        np.random.shuffle(temp_indices)
        synthetic_indices = minority_indices[temp_indices]

        return synthetic_indices

    def _get_nearest_neighbors_indices(self, minority_sample_indices, num_samples):
        """
        최근접 이웃 인덱스를 반환하는 함수
        """
        num_minority_samples, num_neighbors = self.nearest_neighbors_2.shape
        neighbor_indices = np.random.choice(num_neighbors, size=num_samples)

        minority_neighbor_indices = self.nearest_neighbors_2[minority_sample_indices, neighbor_indices]

        return neighbor_indices, minority_neighbor_indices

    def _calculate_differences(self, minority_neighbor_indices, minority_sample_indices, minority_samples):
        """
        샘플 간 차이 값을 계산하는 함수
        """
        differences = minority_samples[minority_neighbor_indices] - minority_samples[minority_sample_indices]
        return differences

    def _create_synthetic_samples(self, synthetic_indices, num_samples, minority_samples):
        """
        synthetic 샘플을 생성하는 함수
        """
        neighbor_indices, minority_neighbor_indices = self._get_nearest_neighbors_indices(synthetic_indices, num_samples)
        differences = self._calculate_differences(minority_neighbor_indices, synthetic_indices, minority_samples)

        # gap 값을 임의로 설정
        gap_values = np.random.random(len(synthetic_indices))

        # synthetic 샘플 생성
        synthetic_samples = minority_samples[synthetic_indices] + gap_values[:, None] * differences

        return synthetic_samples




class SampleGenerator:
    def __init__(self, 
                 sampling_method, 
                 k_neighbors_1=5, 
                 k_neighbors_2=5, 
                 random_state=None, 
                 n_jobs=1):
        """
        SampleGenerator 클래스 초기화

        Parameters:
        sampling_method: 샘플링 방법 ('smote', 'adasyn', 'ramo_boost', 'wot_boost' 등)
        k_neighbors_1: 첫 번째 최근접 이웃 개수 (기본값은 5)
        k_neighbors_2: 두 번째 최근접 이웃 개수 (기본값은 5)
        random_state: 난수 시드 (기본값은 None)
        n_jobs: 병렬 처리를 위한 CPU 수 (기본값은 1)
        """
        self.k_neighbors_1 = k_neighbors_1
        self.k_neighbors_2 = k_neighbors_2
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.smote_method = 'origin'  # SMOTE는 기본적으로 'origin' 방식 사용
        self.sampling_method = sampling_method

    def fit(self, X, y, minority_class=None):
        """
        소수 클래스 데이터를 학습하고 최근접 이웃을 계산하는 함수

        Parameters:
        X: 입력 데이터
        y: 클래스 레이블
        minority_class: 소수 클래스 레이블 (기본값은 자동으로 결정)
        """
        # 소수 클래스 결정
        if minority_class is None:
            class_counts = Counter(y)
            majority_class = max(class_counts, key=class_counts.get)
            minority_class = min(class_counts, key=class_counts.get)

        self.minority_class = minority_class
        self.X_minority = X[y == self.minority_class]
        self.X = X
        self.y = y

        # 최근접 이웃 학습
        if self.sampling_method == 'ramo_boost':
            # 다수 및 소수 클래스 모두에 대해 최근접 이웃 계산
            nn_majority = NearestNeighbors(n_neighbors=self.k_neighbors_1 + 1, n_jobs=self.n_jobs).fit(self.X)
            nn_minority = NearestNeighbors(n_neighbors=self.k_neighbors_2 + 1, n_jobs=self.n_jobs).fit(self.X_minority)
            self.nearest_neighbors_majority = nn_majority.kneighbors(self.X_minority, return_distance=False)[:, 1:]
            self.nearest_neighbors_minority = nn_minority.kneighbors(self.X_minority, return_distance=False)[:, 1:]
        else:
            # 소수 클래스에 대해서만 최근접 이웃 계산
            nn_minority = NearestNeighbors(n_neighbors=self.k_neighbors_2 + 1, n_jobs=self.n_jobs).fit(self.X_minority)
            self.nearest_neighbors_minority = nn_minority.kneighbors(self.X_minority, return_distance=False)[:, 1:]

        self.sampler = SMOTESampler(self.nearest_neighbors_majority if self.sampling_method == 'ramo_boost' else None,
                                    self.nearest_neighbors_minority, labels=self.y, method=self.smote_method)
        self.index_info = None

        return self

    def sample(self, n_samples, sample_weights=None):
        """
        지정된 샘플링 방법에 따라 샘플을 생성하는 함수

        Parameters:
        n_samples: 생성할 샘플 수
        sample_weights: 각 샘플의 가중치 (기본값은 None)

        Returns:
        생성된 샘플
        """
        if self.sampling_method == 'smote_boost':
            return self._smote(n_samples)

        elif self.sampling_method == 'ramo_boost':
            return self._ramo_boost(n_samples, sample_weights)

        elif self.sampling_method == 'wot_boost':
            return self._wot_boost(n_samples, sample_weights)

    def _smote(self, n_samples):
        """
        SMOTE 방식으로 샘플을 생성하는 함수
        """
        np.random.seed(self.random_state)
        synthetic_samples, syn_index = self.sampler.main(
            version='sampling',
            num_samples=n_samples,
            minority_samples=self.X_minority,
            sample_weights=None
        )
        self.index_info = np.unique(syn_index, return_counts=True)
        return synthetic_samples

    def _ramo_boost(self, n_samples, sample_weights=None, alpha=0.3):
        """
        RAMOBoost 방식으로 샘플을 생성하는 함수
        """
        np.random.seed(self.random_state)

        if sample_weights is None:
            # 소수 클래스 샘플의 가중치를 균등하게 설정
            sample_weights = np.ones(len(self.X_minority)) / len(self.X_minority)
        else:
            # sample_weights는 소수 클래스에 해당하는 가중치만 추출해야 함
            # 전체 데이터셋이 아닌 소수 클래스에 대한 가중치로 인덱싱해야 함
            if len(sample_weights) != len(self.y):
                raise ValueError("sample_weights 크기가 self.y와 일치하지 않습니다.")
            
            sample_weights = sample_weights[self.y == self.minority_class]

        n_majority_neighbors = np.sum(self.y[self.nearest_neighbors_majority] != self.minority_class, axis=1)
        adjusted_weights = 1 / (1 + np.exp(-alpha * n_majority_neighbors))
        sample_weights = adjusted_weights * sample_weights

        synthetic_samples, syn_index = self.sampler.main(
            version='sampling',
            num_samples=n_samples,
            minority_samples=self.X_minority,
            sample_weights=sample_weights
        )
        self.index_info = np.unique(syn_index, return_counts=True)
        return synthetic_samples

    def _wot_boost(self, n_samples, sample_weights=None):
        """
        WOTBoost 방식으로 샘플을 생성하는 함수
        """
        np.random.seed(self.random_state)

        if sample_weights is None:
            sample_weights = np.ones(len(self.X_minority)) / len(self.X_minority)
        else:
            assert len(self.y) == len(sample_weights)
            sample_weights = sample_weights[self.y == self.minority_class]

        synthetic_samples, syn_index = self.sampler.main(
            version='weight',
            num_samples=n_samples,
            minority_samples=self.X_minority,
            sample_weights=sample_weights
        )

        if synthetic_samples is None:
            return None
        else:
            self.index_info = np.unique(syn_index, return_counts=True)
            return synthetic_samples


#%%

if __name__ == '__main__' :

    # 데이터 생성: 불균형한 이진 분류 데이터셋
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)
    
    # 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #%%
    # 원본 데이터의 클래스 비율 출력
    print("Original dataset shape:", Counter(y_train))
    
    # SampleGenerator 클래스 사용 예시
    sampling_method = 'wot_boost'  # 'ramo_boost', 'wot_boost'도 가능합니다.
    sample_gen = SampleGenerator(sampling_method=sampling_method, k_neighbors_1=5, k_neighbors_2=5, random_state=42)
    
    # 데이터 학습 (최근접 이웃 계산)
    sample_gen.fit(X_train, y_train)
    
    # 샘플 생성 (예: 소수 클래스 샘플 100개 생성)
    n_samples_to_generate = 100
    synthetic_samples = sample_gen.sample(n_samples=n_samples_to_generate)
    
    # 생성된 샘플 확인
    print("Generated synthetic samples shape:", synthetic_samples.shape)
    
    # 추가된 샘플을 기존 데이터에 추가하는 등 후속 처리 가능


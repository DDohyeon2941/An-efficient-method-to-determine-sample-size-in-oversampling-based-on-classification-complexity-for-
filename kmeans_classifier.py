# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:11:05 2024

@author: dohyeon
"""


from collections import Counter
from scipy.special import xlogy
import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, check_X_y, check_array
from sklearn.preprocessing import normalize


class With_Boost_Clustering(AdaBoostClassifier):
    """
    Implementation of SMOTEBoost.

    SMOTEBoost introduces SMOTE into the AdaBoost algorithm by
    oversampling the minority class during each boosting iteration.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of synthetic samples to generate per boosting step.
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator for the boosted ensemble.
    n_estimators : int, optional (default=25)
        The maximum number of boosting iterations.
    learning_rate : float, optional (default=1.0)
        Learning rate for shrinking the contribution of each classifier.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        Algorithm to use for boosting.
    random_state : int or None, optional (default=None)
        Random state for reproducibility.

    References
    ----------
    .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
           "SMOTEBoost: Improving Prediction of the Minority Class in
           Boosting." European Conference on Principles of Data Mining and
           Knowledge Discovery (PKDD), 2003.
    """

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=25,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            algorithm=algorithm)

    def fit(self, X, y, n_samples, smote, sample_weight=None,
            minority_target=None, is_global=True, cls_T=None):
        """
        Fit the boosting classifier while performing SMOTE during each boosting step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Class labels.
        n_samples : int
            Number of synthetic samples to generate per boosting step.
        smote : GlobalLocalSMOTE
            SMOTE object for oversampling.
        sample_weight : array-like, optional
            Initial sample weights. If None, weights are set to uniform.
        minority_target : int, optional
            Label of the minority class.
        is_global : bool, optional
            Whether to use global or local sampling for SMOTE.
        cls_T : array, optional
            T values used for local oversampling (if applicable).

        Returns
        -------
        self : object
            Returns the fitted model.
        """
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("Unsupported algorithm: %s" % self.algorithm)

        X, y = check_X_y(X, y, accept_sparse='csc')
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            sample_weight /= np.sum(sample_weight)

        if sample_weight.sum() <= 0:
            raise ValueError("Non-positive weighted number of samples.")

        if minority_target is None:
            self.minority_target = min(Counter(y), key=Counter(y).get)
        else:
            self.minority_target = minority_target

        self._validate_estimator()
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        X_min = X[y == self.minority_target]

        for iboost in range(self.n_estimators):
            if len(X_min) >= smote.k:
                X_syn = (smote.global_sample(n_samples) if is_global
                         else smote.local_sample(n_samples, cls_T=cls_T))

                y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target, dtype=np.int64)

                syn_sample_weight = np.ones(X_syn.shape[0]) / X.shape[0]
                newX = np.vstack([X, X_syn])
                newy = np.append(y, y_syn)
                new_sample_weight = np.append(sample_weight, syn_sample_weight)
                new_sample_weight = normalize(new_sample_weight[:, np.newaxis], axis=0).ravel()

                sample_weight, estimator_weight, estimator_error = self._boost(
                    iboost, newX, newy, X, y, new_sample_weight, sample_weight, self.random_state)

                if sample_weight is None:
                    break

                self.estimator_weights_[iboost] = estimator_weight
                self.estimator_errors_[iboost] = estimator_error

                if estimator_error == 0 or sample_weight.sum() <= 0:
                    break

                if iboost < self.n_estimators - 1:
                    sample_weight /= sample_weight.sum()

        return self

    def _boost(self, iboost, synX, syny, orgX, orgy, syn_sample_weight, sample_weight, random_state):
        """
        Perform a single boosting iteration using either SAMME or SAMME.R.
        """
        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, synX, syny, orgX, orgy, syn_sample_weight, sample_weight, random_state)
        return self._boost_discrete(iboost, synX, syny, orgX, orgy, syn_sample_weight, sample_weight, random_state)

    def _boost_real(self, iboost, synX, syny, orgX, orgy, syn_sample_weight, sample_weight, random_state):
        """
        Perform a single boosting iteration using the SAMME.R real algorithm.
        """
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(synX, syny, sample_weight=syn_sample_weight)

        y_proba = estimator.predict_proba(orgX)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_pred = self.classes_.take(np.argmax(y_proba, axis=1), axis=0)
        incorrect = y_pred != orgy
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        y_coding = np.array([-1. / (n_classes - 1), 1.]).take(self.classes_ == orgy[:, np.newaxis])

        np.clip(y_proba, np.finfo(y_proba.dtype).eps, None, out=y_proba)
        estimator_weight = (-1. * self.learning_rate *
                            ((n_classes - 1.) / n_classes) *
                            xlogy(y_coding, y_proba).sum(axis=1))

        if iboost != self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight * (sample_weight > 0))

        return sample_weight, 1., estimator_error

    def _boost_discrete(self, iboost, synX, syny, orgX, orgy, syn_sample_weight, sample_weight, random_state):
        """
        Perform a single boosting iteration using the SAMME discrete algorithm.
        """
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(synX, syny, sample_weight=syn_sample_weight)

        y_pred = estimator.predict(orgX)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_pred != orgy
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            return None, None, None

        estimator_weight = (self.learning_rate *
                            (np.log((1. - estimator_error) / estimator_error) +
                             np.log(n_classes - 1.)))

        if iboost != self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight * incorrect * (sample_weight > 0))

        return sample_weight, estimator_weight, estimator_error

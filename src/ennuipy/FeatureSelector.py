'''
Created on 30 Aug 2019

@author: Koosie DeMoer
'''

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin



class FeatureSelector(BaseEstimator, SelectorMixin):
    '''
    classdocs
    '''


    def __init__(self, feature_corr_min=0.45, feature_x_corr_max=0.5):
        '''
        Constructor
        Parameters
        ----------
        feature_corr_min : <class 'float'>
            Any feature with less than this level of correlation to the target will be ignored
        feature_x_corr_max : <class 'float'>
            Any features that have a x_correlation with a selected feature higher than this will be eliminated.
        '''

        self.feature_corr_min = feature_corr_min
        
        self.feature_x_corr_max = feature_x_corr_max

    def fit(self, X, y):
        """Fit the FeatureSelector model and then the underlying estimator on the selected
           features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        
        data = pd.concat([y, X], axis=1, sort=False)

        corr = data.corr()

        result = corr.reindex(corr['Target'].abs().sort_values( ascending=False ).index)
        result.drop('Target', inplace=True)

        features = []

        while not result.empty:
            if ((result[result['Target'].abs() > self.feature_corr_min]).empty):
                break
            feature_name = result.index.values[0]
            features.append(feature_name)
            result.drop(feature_name, inplace=True)
            result = result[result[feature_name].abs() < self.feature_x_corr_max]
            
        support_ = np.full((len(X.columns)), False)
        
        support_[np.in1d(X.columns.values, features)] = True

        self.features_ = features
        self.support_ = support_
        
        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
        return self.support_
    
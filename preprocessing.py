"""Transformador entrenable: el percentil se aprende solo del entrenamiento."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FareCapper(TransformerMixin, BaseEstimator):
    def __init__(self, quantile=0.99):
        self.quantile = quantile

    def fit(self, X, y=None):
        if not 0 < self.quantile <= 1:
            raise ValueError("quantile debe estar en (0, 1].")
        self.cap_ = float(np.nanquantile(X["Fare"], self.quantile))
        return self

    def transform(self, X):
        check_is_fitted(self, "cap_")
        result = X.copy()
        result["Fare"] = result["Fare"].clip(upper=self.cap_)
        return result

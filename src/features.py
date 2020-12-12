from typing import Sequence

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import KNNImputer

from src.ops import batch

statistics = {
    'mean': lambda x: x.mean(axis=1),
    'std': lambda x: x.std(axis=1),
    'median': lambda x: x.median(axis=1),
    'skew': lambda x: x.skew(axis=1),
}


class NACounter(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['na_counts'] = X[self.columns].isna().sum(axis=1)
        return X


class StatisticsExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        assert all(col in X.columns for col in self.columns)
        return self

    def transform(self, X, y=None):
        for key, fn in statistics.items():
            X[key] = fn(X[self.columns])
        return X


class Imputer(TransformerMixin, BaseEstimator):
    def __init__(self, impute_cols: list, imputer_kwargs: dict):
        self.impute_cols = impute_cols
        self.imputer_kwargs = imputer_kwargs
        self.imputer = KNNImputer(**imputer_kwargs)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.impute_cols], y)
        return self

    def transform(self, X, y=None):
        X[self.impute_cols] = self.imputer.transform(X[self.impute_cols])
        return X


class PeriodStatistics(TransformerMixin, BaseEstimator):
    _prefix = "period_statistics"

    def __init__(self,
                 columns: Sequence[str],
                 period: int,
                 stats: Sequence[str]):
        self.columns = columns
        self.period = period
        self.stats = stats

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for i, columns in enumerate(batch(self.columns, self.period)):
            for stat in self.stats:
                X[f"{self._prefix}_{i}_{self.period}_{stat}"] = statistics[stat](X[columns])

        return X

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features import StatisticsExtractor, Imputer, PeriodStatistics, statistics, NACounter
from src.ops import read_params

if __name__ == '__main__':
    params = read_params("params.yaml")
    seed = params['randomness']['seed']

    root_path = Path(params['paths']['root_path'])
    data_path = root_path / Path(params['paths']['data_dir'])
    train = pd.read_csv(data_path / params['paths']['train_path'])
    test = pd.read_csv(data_path / params['paths']['test_path'])

    day_columns = [c for c in train.columns if 'Day' in c]
    day_columns = list(sorted(day_columns, key=lambda x: int(x.split()[1])))

    target_column = params['columns']['target']
    feature_columns = [col for col in train.columns if col != target_column]

    feature_extraction_pipeline = Pipeline([
        ('na_counter', NACounter(day_columns)),
        ('knn_imputer', Imputer(day_columns, dict())),
        ('statistics_extractor', StatisticsExtractor(day_columns)),
        ('month_statistics', PeriodStatistics(day_columns, 30, list(statistics.keys()))),
    ])

    model = Pipeline([
        ('lgb', lgb.LGBMClassifier(seed=seed, **params['model']['params'])),
    ])

    pipeline = Pipeline([
        ('feature_extraction', feature_extraction_pipeline),
        ('model', model)
    ])

    cv = StratifiedKFold(random_state=seed, **params['cv'])
    scores = cross_val_score(pipeline,
                             train[feature_columns], train[target_column],
                             scoring=lambda est, x, y: f1_score(y, est.predict(x), average='weighted'),
                             cv=cv)
    print("mean score:", np.mean(scores), "+-", np.std(scores))

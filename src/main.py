from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.features import StatisticsExtractor, Imputer, PeriodStatistics, statistics, NACounter

if __name__ == '__main__':
    data_path = Path("../data/raw")
    train = pd.read_csv(data_path / 'train.csv')
    test = pd.read_csv(data_path / 'test.csv')

    day_columns = [c for c in train.columns if 'Day' in c]
    day_columns = list(sorted(day_columns, key=lambda x: int(x.split()[1])))

    target_column = 'Culture'
    feature_columns = [col for col in train.columns if col != target_column]

    feature_extraction_pipeline = Pipeline([
        ('na_counter', NACounter(day_columns)),
        ('knn_imputer', Imputer(day_columns, dict())),
        ('statistics_extractor', StatisticsExtractor(day_columns)),
        ('month_statistics', PeriodStatistics(day_columns, 30, list(statistics.keys()))),
    ])

    model = VotingClassifier([
        ('rf', RandomForestClassifier(random_state=42)),
        ('lgb', lgb.LGBMClassifier(max_depth=4, n_estimators=400, random_state=42)),
        ('svr', make_pipeline(StandardScaler(),
                              LinearSVC(random_state=42))),
        ('log_reg', make_pipeline(StandardScaler(),
                                  LogisticRegression(random_state=42))),
    ])

    pipeline = Pipeline([
        ('feature_extraction', feature_extraction_pipeline),
        ('model', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline,
                             train[feature_columns], train[target_column],
                             scoring=lambda est, x, y: f1_score(y, est.predict(x), average='weighted'),
                             cv=cv)
    print("mean score:", np.mean(scores), "+-", np.std(scores))

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.ops import ROOT_DIR

__step_name = "model"


def eval_model(params: dict):
    seed = params["constants"]["seed"]

    with open(ROOT_DIR / params["feature-extractor"]["pipeline_path"], "rb") as f:
        features_extractor = pickle.load(f)

    with open(ROOT_DIR / params["model"]["path"]["base"], "rb") as f:
        model = pickle.load(f)

    pipeline = Pipeline([
        ('fe', features_extractor),
        ('model', model),
    ])

    features = pd.read_csv(ROOT_DIR / params["feature-extractor"]["features_path"])

    target_col = params['data-reader']['target_col']
    print(features.columns.isin([target_col, ]).shape)
    y_train = features[target_col]
    X_train = features[features.columns.difference([target_col, ])]

    cv = StratifiedKFold(n_splits=params[__step_name]["eval"]["k"])
    scores = cross_val_score(model,
                             X_train, y_train,
                             scoring=lambda est, x, y: f1_score(y, est.predict(x), average='weighted'),
                             cv=cv)

    print("Mean score:", np.mean(scores))

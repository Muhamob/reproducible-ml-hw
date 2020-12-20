import pickle

import lightgbm as lgb
from sklearn.pipeline import Pipeline

from src.ops import ROOT_DIR


def create_model(params: dict):
    seed = params["constants"]["seed"]
    model = Pipeline([
        ('lgb', lgb.LGBMClassifier(seed=seed, **params['model']['params'])),
    ])

    with open(ROOT_DIR / params["model"]["path"]["base"], "wb") as f:
        pickle.dump(model, f)

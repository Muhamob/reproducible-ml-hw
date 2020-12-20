import pickle

import lightgbm as lgb
from sklearn.pipeline import Pipeline

from src.ops import ROOT_DIR, read_params


def create_model(params: dict):
    print("run create model")
    seed = params["constants"]["seed"]
    model = Pipeline([
        ('lgb', lgb.LGBMClassifier(seed=seed, **params['model']['params'])),
    ])

    with open(ROOT_DIR / params["model"]["path"]["base"], "wb") as f:
        pickle.dump(model, f)
    print("done create model")


if __name__ == "__main__":
    params = read_params("src/params.yaml")
    create_model(params)

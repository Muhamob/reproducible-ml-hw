import pickle

import click
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.ops import ROOT_DIR, read_params

__step_name = "model"


@click.command()
@click.option("-p", "--params",
              help="path to params.yaml file",
              type=click.Path(exists=True),
              default=lambda: ROOT_DIR.joinpath("src/params.yaml").as_posix())
def eval_model(params: str):
    params_dict = read_params(params)
    print("run model eval")
    seed = params_dict["constants"]["seed"]

    with open(ROOT_DIR / params_dict["feature-extractor"]["pipeline_path"], "rb") as f:
        features_extractor = pickle.load(f)

    with open(ROOT_DIR / params_dict["model"]["path"]["base"], "rb") as f:
        model = pickle.load(f)

    pipeline = Pipeline([
        ('fe', features_extractor),
        ('model', model),
    ])

    features = pd.read_csv(ROOT_DIR / params_dict["feature-extractor"]["features_path"])

    target_col = params_dict['data-reader']['target_col']
    print(features.columns.isin([target_col, ]).shape)
    y_train = features[target_col]
    X_train = features[features.columns.difference([target_col, ])]

    cv = StratifiedKFold(n_splits=params_dict[__step_name]["eval"]["k"])
    scores = cross_val_score(model,
                             X_train, y_train,
                             scoring=lambda est, x, y: f1_score(y, est.predict(x), average='weighted'),
                             cv=cv)

    with mlflow.start_run():
        mlflow.log_metric("f1-weighted-cv-mean", np.mean(scores))
        mlflow.log_metric("f1-weighted-cv-std", np.std(scores))
        mlflow.log_params(params_dict)

    print("Mean score:", np.mean(scores))
    print("done model eval")


if __name__ == "__main__":
    eval_model()

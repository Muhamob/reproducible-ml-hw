import pickle

import pandas as pd

from src.ops import ROOT_DIR, read_params


def fit_model(params: dict):
    print("run fit model")
    features = pd.read_csv(ROOT_DIR / params["feature-extractor"]["features_path"])
    seed = params["constants"]["seed"]

    target_col = params['data-reader']['target_col']
    y_train = features[target_col]
    X_train = features[features.columns.difference([target_col, ])]

    with open(ROOT_DIR / params["model"]["path"]["base"], "rb") as f:
        model = pickle.load(f)

    model.fit(X_train, y_train)
    with open(ROOT_DIR / params["model"]["path"]["fitted"], "wb") as f:
        pickle.dump(model, f)
    print("done fit model")


if __name__ == "__main__":
    params = read_params("src/params.yaml")
    fit_model(params)

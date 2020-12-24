import pickle

import click
import pandas as pd

from src.ops import ROOT_DIR, read_params


@click.command()
@click.option("-p", "--params",
              help="path to params.yaml file",
              type=click.Path(exists=True),
              default=lambda: ROOT_DIR.joinpath("src/params.yaml").as_posix())
def fit_model(params: str):
    params_dict = read_params(params)
    print("run fit model")
    features = pd.read_csv(ROOT_DIR / params_dict["feature-extractor"]["features_path"])
    seed = params_dict["constants"]["seed"]

    target_col = params_dict['data-reader']['target_col']
    y_train = features[target_col]
    X_train = features[features.columns.difference([target_col, ])]

    with open(ROOT_DIR / params_dict["model"]["path"]["base"], "rb") as f:
        model = pickle.load(f)

    model.fit(X_train, y_train)
    with open(ROOT_DIR / params_dict["model"]["path"]["fitted"], "wb") as f:
        pickle.dump(model, f)
    print("done fit model")


if __name__ == "__main__":
    fit_model()

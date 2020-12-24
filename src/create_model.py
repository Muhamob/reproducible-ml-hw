import pickle

import click
import lightgbm as lgb
from sklearn.pipeline import Pipeline

from src.ops import ROOT_DIR, read_params


@click.command()
@click.option("-p", "--params",
              help="path to params.yaml file",
              type=click.Path(exists=True),
              default=lambda: ROOT_DIR.joinpath("src/params.yaml").as_posix())
def create_model(params: str):
    params_dict = read_params(params)
    print("run create model")
    seed = params_dict["constants"]["seed"]
    model = Pipeline([
        ('lgb', lgb.LGBMClassifier(seed=seed, **params_dict['model']['params'])),
    ])

    with open(ROOT_DIR / params_dict["model"]["path"]["base"], "wb") as f:
        pickle.dump(model, f)
    print("done create model")


if __name__ == "__main__":
    create_model()

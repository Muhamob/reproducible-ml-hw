from src.create_model import create_model
from src.evaluate_model import eval_model
from src.feature_extraction import extract_features
from src.fit_model import fit_model
from src.ops import read_params


def eval_pipeline(params: dict):
    extract_features(params)
    create_model(params)
    eval_model(params)


def fit_pipeline(params: dict):
    extract_features(params)
    create_model(params)
    fit_model(params)


if __name__ == "__main__":
    params = read_params("params.yaml")
    eval_pipeline(params)
    # fit_model(params)
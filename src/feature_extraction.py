import pickle

import click
from sklearn.pipeline import Pipeline

from src.features import NACounter, Imputer, StatisticsExtractor, PeriodStatistics, statistics
from src.load_data import load_data
from src.ops import ROOT_DIR, read_params

__step_name: str = "feature-extractor"


@click.command()
@click.option("-p", "--params",
              help="path to params.yaml file",
              type=click.Path(exists=True),
              default=lambda: ROOT_DIR.joinpath("src/params.yaml").as_posix())
def extract_features(params: str):
    params_dict = read_params(params)
    print("run feature extraction")
    data = load_data(params_dict)
    day_columns = data['day_columns']

    feature_extraction_pipeline = Pipeline([
        ('na_counter', NACounter(day_columns)),
        ('knn_imputer', Imputer(day_columns, dict())),
        ('statistics_extractor', StatisticsExtractor(day_columns)),
        ('month_statistics', PeriodStatistics(day_columns, 30, list(statistics.keys()))),
    ])

    features = feature_extraction_pipeline.fit_transform(
        data['train_df'][data['feature_columns']],
        data['train_df'][data['target_column']]
    )

    feature_extractor_path = ROOT_DIR / params_dict[__step_name]["pipeline_path"]
    with open(feature_extractor_path, "wb") as f:
        pickle.dump(feature_extraction_pipeline, f)

    features_path = ROOT_DIR / params_dict[__step_name]['features_path']
    features[data['target_column']] = data['train_df'][data['target_column']]
    features.to_csv(features_path, index=False)
    print("done feature extraction")


if __name__ == "__main__":
    extract_features()

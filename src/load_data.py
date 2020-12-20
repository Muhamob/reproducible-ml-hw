from pathlib import Path

import pandas as pd

from src.ops import ROOT_DIR


def load_data(params: dict):
    train_data_path = ROOT_DIR / Path(params['data-reader']['train_csv'])
    test_data_path = ROOT_DIR / Path(params['data-reader']['test_csv'])
    
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    day_columns = [c for c in train.columns if 'Day' in c]
    day_columns = list(sorted(day_columns, key=lambda x: int(x.split()[1])))

    target_column = params['data-reader']['target_col']
    feature_columns = [col for col in train.columns if col != target_column]

    return {
        'train_df': train,
        'test_df': test,
        'day_columns': day_columns,
        'feature_columns': feature_columns,
        'target_column': target_column
    }

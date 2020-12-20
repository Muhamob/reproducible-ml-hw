import os
from pathlib import Path

import yaml

ROOT_DIR = Path(os.path.abspath(__file__)).parents[1]


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def read_params(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)

    return params

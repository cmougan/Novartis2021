from os import listdir
import os

import pandas as pd

def get_filenames(path: str, suffix: str):
    filenames = listdir(path)
    return [filename for filename in filenames if filename.endswith(suffix)]


def get_data_from_path(path: str, suffix: str = 'csv'):
    filenames = get_filenames(path=path, suffix=suffix)
    data = dict()
    for file in filenames:
        name = file.split(".")[0]
        data_path = os.path.join(path, file)
        data[name] = pd.read_csv(data_path)
    return data

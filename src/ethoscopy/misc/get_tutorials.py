import pandas as pd
from pathlib import PurePath

def get_tutorial(type):
    """ Get a dataset for use with the tutorials avaibale at : http://lab.gilest.ro/ethoscopy"""
    path = PurePath(__file__)
    this_dir = path.parent

    data_path = this_dir / f'tutorial_data/{type}_data.pkl'
    meta_path = this_dir / f'tutorial_data/{type}_meta.pkl'

    return pd.read_pickle(data_path), pd.read_pickle(meta_path)
import pandas as pd
from pathlib import PurePath

def get_tutorial(data_type):
    """ 
    Get a dataset for use with the tutorials in the tutorial notebook folder.

        Args:
            data_type (str): Choose from two datasets, 'overview' and 'circadian'.
                Circadian is specifically for the circadian tutorial notebook as 
                it contains circadian mutant data.
    
    Returns:
        Two pandas dataframes, data and metadata
    """

    data_types = ['overview', 'circadian']
    if data_type not in data_types:
        raise KeyError(f'data_type argument must be one of {*data_types,}')

    path = PurePath(__file__)
    this_dir = path.parent

    data_path = this_dir / f'tutorial_data/{data_type}_data.pkl'
    meta_path = this_dir / f'tutorial_data/{data_type}_meta.pkl'

    return pd.read_pickle(data_path), pd.read_pickle(meta_path)
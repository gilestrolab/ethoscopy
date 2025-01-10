import pandas as pd
from pathlib import PurePath

def get_tutorial(data_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Load example datasets for tutorial notebooks.
    
    Provides access to pre-packaged datasets for learning ethoscopy functionality.

    Args:
        data_type (str): Dataset to load ('overview' or 'circadian')
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (data, metadata) DataFrames

    Raises:
        KeyError: If data_type is not 'overview' or 'circadian'
    """

    data_types = ['overview', 'circadian']
    if data_type not in data_types:
        raise KeyError(f'data_type argument must be one of {*data_types,}')

    path = PurePath(__file__)
    this_dir = path.parent

    data_path = this_dir / f'tutorial_data/{data_type}_data.pkl'
    meta_path = this_dir / f'tutorial_data/{data_type}_meta.pkl'

    return pd.read_pickle(data_path), pd.read_pickle(meta_path)
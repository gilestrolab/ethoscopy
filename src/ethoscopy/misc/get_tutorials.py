import pandas as pd
from pathlib import Path

def get_tutorial(data_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Load example datasets for tutorial notebooks.
    
    Provides access to pre-packaged datasets for learning ethoscopy functionality.
    Available datasets:
    - 'overview': Basic movement and sleep data
    - 'circadian': Extended recording for circadian analysis

    Args:
        data_type (str): Dataset to load ('overview' or 'circadian')
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (data, metadata) DataFrames

    Raises:
        KeyError: If data_type is not 'overview' or 'circadian'
        FileNotFoundError: If tutorial data files cannot be found
    """
    # Use resolve() to handle symlinks and get absolute path
    path = Path(__file__).absolute()
    this_dir = path.parent
    
    # Normalize input
    data_type = data_type.lower()
    valid_types = {'overview', 'circadian'}
    
    if data_type not in valid_types:
        raise KeyError(f'data_type must be one of: {", ".join(valid_types)}')

    # Use path joining for cross-platform compatibility
    data_path = this_dir / 'tutorial_data' / f'{data_type}_data.pkl'
    meta_path = this_dir / 'tutorial_data' / f'{data_type}_meta.pkl'

    try:
        data = pd.read_pickle(data_path)
        meta = pd.read_pickle(meta_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Tutorial data files not found in: {this_dir / 'tutorial_data'}")

    return data, meta
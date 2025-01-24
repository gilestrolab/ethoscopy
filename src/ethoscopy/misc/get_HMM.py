from hmmlearn.hmm import CategoricalHMM
from pathlib import Path
import pickle

def get_HMM(sex: str) -> 'CategoricalHMM':
    """
    Load pre-trained Hidden Markov Models for Drosophila behavior states.
    
    Provides access to 4-state HMM models (Deep Sleep, Active Awake, Quiet Awake, Active Awake)
    trained separately for male and female flies.

    Args:
        sex (str): Sex of the model to load ('M' for male or 'F' for female)

    Returns:
        CategoricalHMM: Trained HMM model object from hmmlearn

    Raises:
        KeyError: If sex argument is not 'M' or 'F'
        FileNotFoundError: If HMM model file cannot be found
    """
    # Get absolute path that works on all OS
    path = Path(__file__).absolute()
    this_dir = path.parent

    # Normalize input
    sex = sex.upper()
    if sex not in {'M', 'F'}:
        raise KeyError('The argument for "sex" must be "M" or "F"')

    # Use Path's / operator for cross-platform path joining
    hmm_path = this_dir / 'tutorial_data' / f'4_states_{sex}_WT.pkl'

    try:
        with open(hmm_path, 'rb') as file:
            h = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"HMM model file not found at: {hmm_path}\n"
            f"Current directory is: {this_dir}"
        )

    return h



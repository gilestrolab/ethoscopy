from hmmlearn.hmm import CategoricalHMM
from pathlib import PurePath
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
    """

    path = PurePath(__file__).resolve()
    this_dir = path.parent

    sex = sex.upper()
    if sex not in {'M', 'F'}:
        raise KeyError('The argument for "sex" must be "M" or "F"')

    hmm_path = this_dir / 'tutorial_data' / f'4_states_{sex}_WT.pkl'

    try:
        with open(hmm_path, 'rb') as file:
            h = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"HMM model file not found at: {hmm_path}")

    return h



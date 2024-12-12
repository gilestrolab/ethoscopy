from pathlib import PurePath
import pickle

def get_HMM(sex):
    """Get trained hidden markov models for female and male drosophila -- a 4 state model '
    Deep Sleep, Active Awake, Quiet awake, Active Awake 
    
        Args:
            sex (str): Choose which trained model you wold like. 
                Either 'M' for male or 'F' for female. 

    Returns:
        a hmmlearn.hmm.CategoricalHMM object
    """

    path = PurePath(__file__)
    this_dir = path.parent

    if sex != 'M' and sex != 'F':
        raise KeyError('The argument for "sex" must be "F" or "M"')

    hmm_path = this_dir / f'tutorial_data/4_states_{sex}_WT.pkl'

    with open(hmm_path, 'rb') as file: 
        h = pickle.load(file)

    return h
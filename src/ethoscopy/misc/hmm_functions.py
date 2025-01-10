import pandas as pd 
import numpy as np
from ethoscopy.misc.rle import rle

def hmm_pct_transition(state_array: np.ndarray, total_states: np.ndarray) -> pd.DataFrame:
    """
    Calculate proportion of occurrences for each behavioral state.
    
    Analyzes HMM-decoded state sequences to determine state distribution.

    Args:
        state_array (np.ndarray): Array of HMM-decoded states
        total_states (np.ndarray): Array of possible state values

    Returns:
        pd.DataFrame: Proportions of each state occurrence
    """

    v, s, l = rle(state_array)

    states_dict = {}

    def average(a):
        total = a.sum()
        count = len(a)
        av = total / count
        return av

    for i in total_states:
        states_dict[i] = average(np.where(v == i, 1, 0))

    state_list = [states_dict]
    df = pd.DataFrame(state_list)

    return df

def hmm_mean_length(state_array: np.ndarray, delta_t: int = 60, 
                   raw: bool = False, func: str = 'mean') -> pd.DataFrame:
    """
    Calculate mean duration of behavioral state runs.
    
    Analyzes continuous runs of HMM states to determine typical durations.

    Args:
        state_array (np.ndarray): Array of HMM-decoded states
        delta_t (int, optional): Time difference between points in seconds. Default is 60.
        raw (bool, optional): Return all run lengths instead of means. Default is False.
        func (str, optional): Aggregation function for lengths. Default is 'mean'.

    Returns:
        pd.DataFrame: Mean lengths or raw runs of each state
    """
    assert(isinstance(raw, bool))
    delta_t_mins = delta_t / 60

    v, s, l = rle(state_array)

    df = pd.DataFrame(data = zip(v, l), columns = ['state', 'length'])
    df['length_adjusted'] = df['length'].map(lambda l: l * delta_t_mins)
    
    if raw == True:
        return df
    else:
        gb_bout = df.groupby('state').agg(**{
                            'mean_length' : ('length_adjusted', func)
        })
        gb_bout.reset_index(inplace = True)

        return gb_bout

def hmm_pct_state(state_array: np.ndarray, time: np.ndarray, 
                  total_states: np.ndarray, avg_window: int = 30) -> pd.DataFrame:
    """
    Calculate state percentages within sliding windows.
    
    Computes proportion of each state within specified time windows.

    Args:
        state_array (np.ndarray): Array of HMM-decoded states
        time (np.ndarray): Array of timestamps
        total_states (np.ndarray): Array of possible state values
        avg_window (int, optional): Window size in time units. Default is 30.

    Returns:
        pd.DataFrame: State percentages per window with columns [t, state_1, state_2, ...]
    """
    states_dict = {}

    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    for i in total_states:
        states_dict['state_{}'.format(i)] = moving_average(np.where(state_array == i, 1, 0), n = avg_window)

    adjusted_time = time[avg_window-1:]

    df = pd.DataFrame.from_dict(states_dict)
    df.insert(0, 't', adjusted_time)
                        
    return df
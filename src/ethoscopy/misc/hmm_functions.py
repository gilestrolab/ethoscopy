import pandas as pd 
import numpy as np
from ethoscopy.misc.general_functions import rle

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

    v, _, _ = rle(state_array)
    
    # Vectorized calculation of average
    states_dict = {
        state: np.mean(v == state) 
        for state in total_states
    }
    
    return pd.DataFrame([states_dict])

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
    assert isinstance(raw, bool)
    delta_t_mins = delta_t / 60

    v, _, l = rle(state_array)
    
    df = pd.DataFrame({
        'state': v,
        'length': l,
        'length_adjusted': l * delta_t_mins 
    })
    
    if raw:
        return df
    
    return (df.groupby('state')
             .agg(**{'mean_length': ('length_adjusted', func)})
             .reset_index())

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
    # Pre-calculate window size for efficiency
    n = avg_window
    
    def moving_average(a: np.ndarray) -> np.ndarray:
        """Vectorized moving average calculation"""
        cumsum = np.cumsum(a, dtype=float)
        cumsum[n:] = cumsum[n:] - cumsum[:-n]
        return cumsum[n - 1:] / n
    
    states_dict = {
        f'state_{i}': moving_average(state_array == i) 
        for i in total_states
    }
    
    df = pd.DataFrame(states_dict)
    df.insert(0, 't', time[n-1:])
    
    return df
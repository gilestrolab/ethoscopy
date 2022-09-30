import pandas as pd 
import numpy as np
from ethoscopy.misc.rle import rle

def hmm_pct_transition(state_array, total_states):
    """
    Finds the proportion of instances of runs of each state per array/fly
    params:
    @state_array =  1D numpy array produced from a HMM decoder
    @total_states = numerical array denoting the states in 'state_array'
    """

    v, s, l = rle(state_array)

    states_dict = {}

    def average(a):
        total = a.sum()
        count = len(a)
        av = total / count
        return av

    for i in total_states:
        states_dict[f'{i}'] = average(np.where(v == i, 1, 0))

    state_list = [states_dict]
    df = pd.DataFrame(state_list)

    return df

def hmm_mean_length(state_array, delta_t = 60, raw = False):
    """
    Finds the mean length of each state run per array/fly 
    returns a dataframe with a state column containing the states id and a mean_length column
    params:
    @state_array =  1D numpy array produced from a HMM decoder
    @delta_t = the time difference between each element of the array
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
                            'mean_length' : ('length_adjusted', 'mean')
        })
        gb_bout.reset_index(inplace = True)

        return gb_bout

def hmm_pct_state(state_array, time, total_states, avg_window = 30):
    """
    Takes a window of n and finds what percentage each state is present within that window
    returns a dataframe with columns t, and states with their corresponding percentage per window
    params:
    @state_array =  1D numpy array produced from a HMM decoder
    @time = 1D numpy array of the timestamps of state_array of equal length and same order
    @total_states = numerical array denoting the states in 'state_array'
    @avg_window = length of window given as elements of the array
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
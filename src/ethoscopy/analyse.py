import pandas as pd
import numpy as np 
from random import shuffle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import copy

from math import floor
from sys import exit
from scipy.stats import zscore
from ethoscopy.misc.rle import rle

def max_velocity_detector(data, 
                        time_window_length,
                        velocity_correction_coef = 3e-3,
                        masking_duration = 6,
                        optional_columns = 'has_interacted'                        
                        ):
    """ 
    Max_velocity_detector is the default movement classification for real-time ethoscope experiments.
    It is benchmarked against human-generated ground truth.

        Args:
            data (pd.DataFrame): A dataframe containing behavioural variables of a single animal (no id)    
            time_window_length (int, optional): The period of time the data is binned and sampled to, i.e. if 60 the timestep will per row will be 60 seconds.
            velocity_correction_coef (float, optional):  A coefficient to correct the velocity data (change for different length tubes). For 'small' tubes (20 per ethoscope) =
                3e-3, for 'long' tubes (10 per ethoscope) = 15e-4. Default is 3e-3.
            masking_duration (int, optional): The number of seconds during which any movement is ignored (velocity is set to 0) after a stimulus is delivered (a.k.a. interaction).
                If using the AGO set to 0. Default is 6.
            optional_columns (str, optional): The columns other than ['t', 'x', 'velocity'] that you want included post analysis. Default is 'has_interacted'.

    returns:
        A pandas dataframe object with columns such as 't', 'moving', 'max_velocity', 'mean_velocity' and 'beam_cross'
    """

    if len(data.index) < 100:
        return None

    needed_columns = ['t', 'x', 'y', 'w', 'h', 'phi', 'xy_dist_log10x1000']

    dt = prep_data_motion_detector(data,
                                needed_columns = needed_columns,
                                time_window_length = time_window_length,
                                optional_columns = optional_columns)

    dt['deltaT'] = dt.t.diff()
    dt['dist'] = 10 ** (dt.xy_dist_log10x1000 / 1000)
    # in rethomics it was v = dist / deltaT and then vc = v * (deltaT / vcoef), subrating in the first equation ablates the deltaT leavinf vc = dist / vcoef
    dt['velocity'] = dt.dist / velocity_correction_coef

    dt['beam_cross'] = abs(np.sign(0.5 - dt['x']).diff())
    dt['beam_cross'] = np.where(dt['beam_cross'] == 2.0, True, False)

    if 'has_interacted' not in dt.columns:
        if masking_duration > 0:
            masking_duration = 0
        dt['has_interacted'] = 0

    dt['interaction_id'] = dt['has_interacted'].cumsum()
    dt['mask'] = dt.groupby('interaction_id', group_keys = False)['t'].apply(lambda x: pd.Series(np.where(x < (x.min() + masking_duration), True, False), index=x.index))
    dt['beam_cross'] = dt['beam_cross'] & ~dt['mask']
    dt = dt.drop(columns = ['interaction_id', 'mask'])

    d_small = dt.groupby('t_round').agg(
    x = pd.NamedAgg(column='x', aggfunc='mean'),
    y = pd.NamedAgg(column='y', aggfunc='mean'),
    w = pd.NamedAgg(column='w', aggfunc='mean'),
    h = pd.NamedAgg(column='h', aggfunc='mean'),
    phi = pd.NamedAgg(column='phi', aggfunc='mean'),
    max_velocity = pd.NamedAgg(column='velocity', aggfunc='max'),
    mean_velocity = pd.NamedAgg(column='velocity', aggfunc='mean'),
    distance = pd.NamedAgg(column='dist', aggfunc='sum'),
    interactions = pd.NamedAgg(column='has_interacted', aggfunc='sum'),
    beam_crosses = pd.NamedAgg(column='beam_cross', aggfunc= 'sum')
    )

    d_small['moving'] = np.where(d_small['max_velocity'] > 1, True, False)
    d_small['micro'] = np.where((d_small['max_velocity'] > 1) & (d_small['max_velocity'] < 2.5), True, False)
    d_small['walk'] = np.where(d_small['max_velocity'] > 2.5, True, False)
    d_small.rename_axis('t', inplace = True)
    d_small.reset_index(level=0, inplace=True)

    return d_small

def prep_data_motion_detector(data,
                            needed_columns,
                            time_window_length = 10,
                            optional_columns = None 
                            ):
    """ 
    This function bins all points of the time series column into a specified window.
    Also checks optional columns provided in max_velocity_detector are present.
    
        Args:
            data (pandas dataframe): The dataframe as entered into the max_velocity_detector function
            needed_columns (str): Columns to be kept and the function enacted upon
            time_window_length (int): The period of time the data is binned and sampled to, default is 10
            optional_columns (str): Columns other than ['t', 'x', 'xy_dist_log10x1000'] that you want included post analysis, default is None
    
    returns:
        The same object as entered into the function
    """
    
    if all(elem in data.columns.values for elem in needed_columns) is not True:
        warnings.warn('data from ethoscope should have columns named {}!'.format(needed_columns))
        exit()

    # check optional columns input are column headings
    if optional_columns != None:
    
        if isinstance(optional_columns, str):
            check_optional_columns = set(data.columns.tolist()).intersection(list([optional_columns]))
            needed_columns = list(set(list(check_optional_columns) + needed_columns)) 
        else:
            check_optional_columns = set(data.columns.tolist()).intersection(optional_columns)
            needed_columns = list(set(list(check_optional_columns) + needed_columns))

    dc = copy.deepcopy(data[needed_columns])
    dc['t_round'] = dc['t'].map(lambda t: time_window_length * floor(t / time_window_length)) 
    
    def curate_sparse_roi_data(data,
                            window = 60,
                            min_p = 20
                            ):
        """ 
        Remove rows from table when there are not enough data points per the given window

        Params:
        @data =  pandas dataframe, dataframe containing ethoscope raw data with column 't' containing time series data
        @window = int, the size of the window to search for minimum points, default is 60
        @min_p = int, the minimum number of data points needed in a given window for it not to be removed, default is 20

        returns the same object as entered into the function
        """
        data['t_w'] = data['t'].map(lambda t: window * floor(t / window))
        data['n_points'] = data.groupby(['t_w'])['t_w'].transform('count')
        data = data[data.n_points > min_p]
        data.drop(columns = ['t_w', 'n_points'], inplace = True)

        return data

    dc = curate_sparse_roi_data(dc)

    return dc

def sleep_annotation(data, 
                    time_window_length = 10,
                    min_time_immobile = 300,
                    motion_detector_FUN = max_velocity_detector,
                    masking_duration = 6,
                    velocity_correction_coef = 3e-3
                    ):
    """ 
    This function first uses a motion classifier to decide whether an animal is moving during a given time window.
    Then, it defines sleep as contiguous immobility for a minimum duration.
    
        Args:
            data (pandas dataframe): The dataframe containing behavioural variables from one animals.
            time_window_length (int, optional): The period of time the data is binned and sampled to. Default is 10
            min_time_immobile (int, optional): Immobility bouts longer or equal to this value are considered as asleep. Default is 300 (i.e 5 mins)
            motion_detector_FUN (function, optional): The function to curate raw ethoscope data into velocity measurements. Default is max_velocity_detector.
            masking_duration (int, optional): The number of seconds during which any movement is ignored (velocity is set to 0) after a stimulus is delivered (a.k.a. interaction). 
                If using the AGO set to 0. Default is 6.
            velocity_correction_coef (float, optional): A coefficient to correct the velocity data (change for different length tubes). For 'small' tubes (20 per ethoscope) =
                3e-3, for 'long' tubes (10 per ethoscope) = 15e-4. Default is 3e-3.
            
    returns: 
        A pandas dataframe containing columns 'moving' and 'asleep'
    """

    if len(data.index) < 100:
        return None
    
    d_small = motion_detector_FUN(data, time_window_length, masking_duration = masking_duration, velocity_correction_coef = velocity_correction_coef)

    if len(d_small.index) < 100:
        return None

    time_map = pd.Series(range(d_small.t.iloc[0], 
                        d_small.t.iloc[-1] + time_window_length, 
                        time_window_length
                        ), name = 't')

    missing_values = time_map[~time_map.isin(d_small['t'].tolist())]
    d_small = d_small.merge(time_map, how = 'right', on = 't', copy = False).sort_values(by=['t'])
    d_small['is_interpolated'] = np.where(d_small['t'].isin(missing_values), True, False)
    d_small['moving'] = np.where(d_small['is_interpolated'] == True, False, d_small['moving'])

    def sleep_contiguous(moving, fs, min_valid_time = 300):
        """ 
        Checks if contiguous bouts of immobility are greater than the minimum valid time given

        Params:
        @moving = pandas series, series object comtaining the movement data of individual flies
        @fs = int, sampling frequency (Hz) to scale minimum length to time in seconds
        @min_valid_time = min amount immobile time that counts as sleep, default is 300 (i.e 5 mins) 
        
        returns a list object to be added to a pandas dataframe
        """
        min_len = fs * min_valid_time
        r_sleep =  rle(np.logical_not(moving)) 
        valid_runs = r_sleep[2] >= min_len 
        r_sleep_mod = valid_runs & r_sleep[0]
        r_small = []
        for c, i in enumerate(r_sleep_mod):
            r_small += ([i] * r_sleep[2][c])

        return r_small

    d_small['asleep'] = sleep_contiguous(d_small['moving'], 1/time_window_length, min_valid_time = min_time_immobile)
    
    return d_small

def stimulus_response(data, start_response_window = 0, response_window_length = 10, add_false = False, velocity_correction_coef = 3e-3):
    """
    Stimulus_response finds interaction times from raw ethoscope data to detect responses in a given window.
    This function will only return data from around interaction times and not whole movement data from the experiment.

        Args:
            data (pd.DataFrame): The dataframe containing behavioural variable from many or one multiple animals 
            response_window (int, optional): The period of time (seconds) after the stimulus to check for a response (movement). Default is 10 seconds
            add_false (bool / int, optional): If not False then an int which is the percentage of the total of which to add false interactions, recommended is 10.
                This is for use with old datasets with no false interactions so you can observe spontaneous movement with a HMM. Default is False
            velocity_correction_coef (float, optional): A coefficient to correct the velocity data (change for different length tubes). Default is 3e-3.
    
    returns  
        A pandas dataframe object with columns such as 'interaction_t' and 'has_responded'
    """

    if start_response_window == response_window_length or start_response_window > response_window_length:
        raise ValueError("start_response_window must be less than response_window_length") 

    # check for has_interaction column, as is removed during loading of roi if all false
    if any('has_interacted' in ele for ele in data.columns.tolist()) is False:
        print('ROI was unable to load due to there being no interactions in the database')
        return None

    data['deltaT'] = data.t.diff()
    data['dist'] = 10 ** (data.xy_dist_log10x1000 / 1000)
    data['velocity'] = data.dist / velocity_correction_coef
    data.drop(columns = ['deltaT', 'dist'], inplace = True)

    if add_false is not False:
        if add_false <= 0 or add_false >= 101:
            raise ValueError("add_false must be between 1 and 100") 
        int_list = [2] * (int(len(data)*(add_false/100)))
        int_list_2 = [0] * (len(data) - len(int_list))
        int_list_all = int_list + int_list_2 
        shuffle(int_list_all)
        data['has_interacted2'] = int_list_all 
        data['has_interacted'] = np.where(data['has_interacted'] == 1, data['has_interacted'], data['has_interacted2'])
        data = data.drop(columns = ['has_interacted2'])

    #isolate interaction times
    interaction_dt = data['t'][(data['has_interacted'] == 1) | (data['has_interacted'] == 2)].to_frame()
    # interaction_dt.rename(columns = {'t' : 't_int'}, inplace = True)

    #check some interactions took place, return none if empty
    if len(interaction_dt.index) < 1:
        return None

    interaction_dt['start'] = interaction_dt.t
    interaction_dt['end'] = interaction_dt.t + response_window_length
    interaction_dt['int_id'] = np.arange(1, len(interaction_dt) + 1)

    # ints = data.t.values
    starts = interaction_dt.start.values 
    ends = interaction_dt.end.values  

    #### Old method, too memory intensive - the np.where is massive
    # search all time values and retrieve the indexes of values between start and end times
    # creates two lists. first with indices for data dataframe and the second is the indices for interaction_dt
    # create dataframe of two
    # i, j = np.where((ints[:, None] >= starts) & (ints[:, None] <= ends))
    # df = pd.DataFrame(
    #     np.column_stack([data.values[i], interaction_dt.values[j]]),
    #     columns= data.columns.append(interaction_dt.columns)
    # )

    # New method, but can be slow
    df = pd.concat([data[(data['t'] >= i) & (data['t'] < q)] for i, q in zip(starts, ends)])
    df = df.join(interaction_dt, rsuffix = '_int').fillna(method = 'ffill')

    # find relative time to interaction and check for movemokonomiyaki flourent
    df['t_rel'] = df.t - df.t_int
    df = df[(df['t_rel'] > start_response_window) | (df['t_rel'] == 0)]
    df.rename(columns = {'t_int' : 'interaction_t'}, inplace = True)
    df['has_responded'] = np.where((df['t_rel'] > 0) & (df['velocity'] > 1), True, False)
    df['has_walked'] = np.where((df['t_rel'] > 0) & (df['velocity'] > 2.5), True, False)
    df.drop(columns = ['xy_dist_log10x1000', 'start', 'end'], inplace = True)

    response_rows = []
    # is any response take the interaction row and change response to True and t_rel to time till movement
    def find_interactions(response_data):
        if any(response_data['has_responded']):
            response_dict = response_data[response_data['t_rel'] == 0].to_dict('records')[0]
            response_dict['has_responded'] = True
            response_dict['t_rel'] = response_data['t_rel'][response_data['has_responded'] == True].iloc[0]
            response_dict.pop('int_id')
            response_rows.append(response_dict)
        else:
            response_dict = response_data[response_data['t_rel'] == 0].to_dict('records')[0]
            response_dict.pop('int_id')
            response_rows.append(response_dict)

    df.groupby('int_id').apply(find_interactions)

    return pd.DataFrame(response_rows)

def stimulus_prior(data, window = 300, response_window_length = 10, velocity_correction_coef = 3e-3):
    """
    Stimulus_prior is a modification of stimulus_response. It only takes data with a populated has_interacted column.
    The function will take a response window (in seconds) to find the variables recorded by the ethoscope in this window prior to an 
    interaction taking place. Each run is given a unique ID per fly, however it is not unique to other flies. To do so, combine the 
    fly ID with run ID after.

    Args:
        data (pd.DataFrame): A dataframe containing behavioural variable from many or one multiple animals 
        window (int, optional): The period of time (seconds) prior to the stimulus you want data retrieved for. Default is 300.
        response_window_length (int, optional): The period of time (seconds) after the stimulus to check for a response (movement). Default is 10 seconds.
        velocity_correction_coef (float, optional): A coefficient to correct the velocity data (change for different length tubes), default is 3e-3
    
    returns:  
        a pandas dataframe object with columns such as 't_count' and 'has_responded'
    """

    # check for has_interaction column, will be moved in prior download if all interactions are false
    if any('has_interacted' in ele for ele in data.columns.tolist()) is False:
        print('ROI was unable to load due to no interactions in the database')
        return None

    data['deltaT'] = data.t.diff()
    data['dist'] = 10 ** (data.xy_dist_log10x1000 / 1000)
    data['velocity'] = data.dist / velocity_correction_coef
    data.drop(columns = ['deltaT', 'dist'], inplace = True)

    #isolate interaction times
    interaction_dt = data['t'][data['has_interacted'] == 1].to_frame()
    interaction_dt.rename(columns = {'t' : 'int_t'}, inplace = True)

    #check some interactions took place, return none if empty
    if len(interaction_dt.index) < 1:
        return None

    interaction_dt['start'] = interaction_dt.int_t - window
    interaction_dt['end'] = interaction_dt.int_t + response_window

    ints = data.t.values
    starts = interaction_dt.start.values 
    ends = interaction_dt.end.values  

    i, j = np.where((ints[:, None] >= starts) & (ints[:, None] <= ends))
    df = pd.DataFrame(
        np.column_stack([data.values[i], interaction_dt.values[j]]),
        columns = data.columns.append(interaction_dt.columns)
    )

    df['t_rel'] = df.t - df.int_t
    df.rename(columns = {'int_t' : 'interaction_t'}, inplace = True)
    df['has_responded'] = np.where((df['t_rel'] > 0) & (df['velocity'] > 1), True, False)
    df['has_walked'] = np.where((df['t_rel'] > 0) & (df['velocity'] > 2.5), True, False)
    df.drop(columns = ['start', 'end'], inplace = True)

    # filter by window ahead of interaction time and find any postive response, return new df with only interaction time == 0 rows with response
    df['t'] = np.floor(df['t'])
    start_list = np.floor(interaction_dt['start'].to_numpy()).astype(int)
    end_list = np.ceil(interaction_dt['end'].to_numpy()).astype(int)

    response_df = pd.DataFrame()

    def format_window(response):
        if response is True:
            r = 1
        else:
            r = 0

        motif_df = filtered_df[filtered_df['t_rel'] <= 0]
        motif_df['has_responded'] = np.where(motif_df['t_rel'] == 0, response, motif_df['has_responded']) 
        cols = motif_df.columns.tolist()
        motif_df[cols] = motif_df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        d_small = motif_df.groupby('t').agg(**{
                        't' : ('t', 'mean'),
                        'x' : ('x', 'mean'),
                        'y' : ('y', 'mean'),
                        'w' : ('w', 'mean'),
                        'h' : ('h', 'mean'),
                        'phi' : ('phi', 'mean'),
                        'xy_dist_log10x1000' : ('xy_dist_log10x1000', 'mean'),
                        'velocity' : ('velocity', 'mean')
                })
        t_range = list(range(0, len(d_small['t'] + 1)))
        t_range.reverse()
        d_small['t_count'] = t_range
        if len(d_small) != window + 1:
            return None
        else:
            d_small['response'] = [response] * len(d_small)
            id = f'{c}_{r}'
            d_small['run_id'] = [id] * len(d_small)
            return d_small

    for c, (i,q) in enumerate(zip(start_list, end_list)):

        filtered_df = df[df.t.isin(list(range(i, q + 1)))]
        response_data = filtered_df[filtered_df['t_rel'] >= 0]

        if any(response_data['has_responded']):
            formatted_small = format_window(response = True)
            if formatted_small is not None:
                response_df = response_df.append(formatted_small)

        else:
            formatted_small = format_window(response = False)
            if formatted_small is not None:
                response_df = response_df.append(formatted_small)

    return response_df

# The below function needs to be re-written and also i'm not entirely sure what it does so removing it for now.

# def isolate_activity_lengths(data, intervals, window, inactive = True, velocity_correction_coef = 3e-3):
#     """
#     Isolate activity lengths is a loading function for use with interaction datasets. THe function will find consecutive runs of inactivity or activity and segment them into same sized windows
#     at intervals stated by the user. This function

#     Params:
#     @data = pandas dataframe, a dataframe object as provided from the read_single_roi fucntion with a column of time 't' in seconds
#     @intervals = list of ints, a list with the timestamps you want the window to work back from, must be in minutes
#     @window = int, the time frame you want to work back from each interval
#     @inactive = bool, whether to search for runs of activity or inactivity
#     @velocity_correction_coef - float, coeffient to find the velocity over time

#     returns a pandas dataframe with every run according the the requirements and all data values
#     """
#     assert(isinstance(intervals, list))
#     assert(all(isinstance(item, int) for item in intervals))

#     if len(data.index) < 100:
#         return None

#     window = window * 60

#     data['deltaT'] = data.t.diff()
#     data['dist'] = 10 ** (data.xy_dist_log10x1000 / 1000)
#     data['velocity'] = data.dist / velocity_correction_coef
#     data.drop(columns = ['deltaT', 'dist'], inplace = True)
#     data['t'] = np.floor(data['t'])
#     data = data.groupby('t').agg(**{
#                 'x' : ('x', 'mean'),
#                 'y' : ('y', 'mean'),
#                 'w' : ('w', 'mean'),
#                 'h' : ('h', 'mean'),
#                 'phi' : ('phi', 'mean'),
#                 'xy_dist_log10x1000' : ('xy_dist_log10x1000', 'max'),
#                 'velocity' : ('velocity', 'max')
#         })
#     data.reset_index(inplace = True)
#     data['moving'] = np.where(data['velocity'] > 1, 1, 0)

#     def norm_1_0(data, var):
#         v_min = data[var].min()
#         v_max = data[var].max()
#         data[var] = data[var].map(lambda v: (v - v_min) / (v_max - v_min))
#         return data
    
#     data = norm_1_0(data, var = 'x')

#     for i in ['w', 'h']:
#         data[f'{i}_z'] = np.abs(zscore(data[i].to_numpy())) < 3
#         data[i] = np.where(data[f'{i}_z'] == True, data[i], np.nan)
#         data[i] = data[i].fillna(method = 'ffill')
#         data[i] = data[i].fillna(method = 'bfill')
#         data.drop(columns = [f'{i}_z'], inplace = True)
#         data = norm_1_0(data, var = i)

#     def find_inactivity(data, inactive = inactive):
#                 if inactive == True:
#                     elem = 0
#                 else:
#                     elem = 1
#                 inactive_count = []
#                 data_list = []
#                 counter = 1
#                 for c, q in enumerate(data):
#                     if c == 0 and q != elem:
#                         inactive_count.append(np.NaN)
#                         data_list.append(q)

#                     elif c == 0 and q == elem:
#                         inactive_count.append(counter)
#                         data_list.append(q)
#                         counter += 1

#                     else:
#                         if q == elem:
#                             inactive_count.append(counter)
#                             data_list.append(q)
#                             counter += 1

#                         else:
#                             inactive_count.append(np.NaN)
#                             data_list.append(q)
#                             counter = 1

#                 return inactive_count
    
#     data['inactive_count'] =  find_inactivity(data['moving'].to_numpy())
#     data = data[data['inactive_count'] <= max(intervals) * 60]

#     inactivity_df = pd.DataFrame()

#     for interval in intervals:
#         #isolate interaction times
#         interval = interval * 60
#         interaction_dt = data['t'][data['inactive_count'] == interval].to_frame()
#         interaction_dt.rename(columns = {'t' : 'int_t'}, inplace = True)

#         #check some interactions took place, return none if empty
#         if len(interaction_dt.index) < 1:
#             return None

#         interaction_dt['start'] = interaction_dt.int_t - (window-1)
#         interaction_dt['end'] = interaction_dt.int_t
        
#         ints = data.t.values
#         starts = interaction_dt.start.values 
#         ends = interaction_dt.end.values  

#         i, j = np.where((ints[:, None] >= starts) & (ints[:, None] <= ends))
        
#         df = pd.DataFrame(
#             np.column_stack([data.values[i], interaction_dt.values[j]]),
#             columns = data.columns.append(interaction_dt.columns)
#         )
#         df.drop(columns = ['end', 'int_t', 'inactive_count'], inplace = True)

#         gb = df.groupby('start').size()
#         filt_gb = gb[gb == window]
#         filt_df = df[df['start'].isin(filt_gb.index.tolist())]
#         filt_df['t_rel'] = list(range(interval - window, interval)) * len(filt_gb.index)
#         inactivity_df = pd.concat([inactivity_df, filt_df], ignore_index = False)
                

#     return inactivity_df
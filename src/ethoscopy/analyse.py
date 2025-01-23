from typing import Optional, List, Union
import pandas as pd
import numpy as np 
from math import floor
import copy
from random import shuffle
from ethoscopy.misc.general_functions import rle

def max_velocity_detector(data: pd.DataFrame, 
                        time_window_length: int = 10,
                        velocity_correction_coef: float = 3e-3,
                        masking_duration: int = 6,
                        velocity_threshold: float = 1.0,
                        walk_threshold: float = 2.5,
                        ) -> Optional[pd.DataFrame]:
    """ 
    Default movement classification for real-time ethoscope experiments.
    
    Benchmarked against human-generated ground truth to classify different types of movement.

    Args:
        data (pd.DataFrame): Behavioral variables of a single animal (no id)    
        time_window_length (int, optional): Period of time the data is binned and sampled to. Default is 10.
        velocity_correction_coef (float, optional): Coefficient to correct velocity data. Use 3e-3 for 'small' tubes 
            (20 per ethoscope), 15e-4 for 'long' tubes (10 per ethoscope). Default is 3e-3.
        masking_duration (int, optional): Seconds during which movement is ignored after stimulus. Default is 6.
        velocity_threshold (float, optional): Threshold above which movement is detected. Default is 1.0.
        walk_threshold (float, optional): Threshold above which movement is classified as walking. Default is 2.5.

    Returns:
        Optional[pd.DataFrame]: DataFrame with movement classifications or None if insufficient data.
            Includes columns: t, x, y, w, h, phi, max_velocity, mean_velocity, distance, 
            interactions, beam_crosses, moving, micro, walk
    """

    if velocity_threshold <= 0 or walk_threshold <= velocity_threshold:
            raise ValueError("Invalid thresholds: velocity_threshold must be > 0 and < walk_threshold")

    # Validate input data
    if len(data.index) < 100:
        return None

    required_columns = ['t', 'x', 'y', 'w', 'h', 'phi', 'xy_dist_log10x1000']
    # optional columns is a legacy of the R version. All ethoscope data should have it now
    optional_columns = ['has_interacted']

    # Prepare and bin the data
    dt = prep_data_motion_detector(data,
                                required_columns=required_columns,
                                time_window_length=time_window_length,
                                optional_columns=optional_columns)

    # Calculate velocities and distances
    dt['deltaT'] = dt.t.diff()
    # in rethomics it was v = dist / deltaT and then vc = v * (deltaT / vcoef), subrating in the first equation ablates the deltaT leavinf vc = dist / vcoef
    # in this case we are using the log10x1000 distance and then correcting for the velocity correction coefficient
    dt['dist'] = 10 ** (dt.xy_dist_log10x1000 / 1000)
    dt['velocity'] = dt.dist / velocity_correction_coef

    # Detect beam crossings (crossing center of arena)
    dt['beam_cross'] = abs(np.sign(0.5 - dt['x']).diff())
    dt['beam_cross'] = np.where(dt['beam_cross'] == 2.0, True, False)

    # Handle interaction masking
    if 'has_interacted' not in dt.columns:
        masking_duration = 0
        dt['has_interacted'] = 0

    if masking_duration > 0:
        dt['interaction_id'] = dt['has_interacted'].cumsum()
        dt['mask'] = dt.groupby('interaction_id', group_keys = False)['t'].apply(
            lambda x: pd.Series(np.where(x < (x.min() + masking_duration), True, False), index=x.index)
        )
        dt['beam_cross'] = dt['beam_cross'] & ~dt['mask']
        dt = dt.drop(columns = ['interaction_id', 'mask'])

    # Aggregate data by time window using dictionary of operations
    agg_dict = {
        'x': ('x', 'mean'),
        'y': ('y', 'mean'),
        'w': ('w', 'mean'),
        'h': ('h', 'mean'),
        'phi': ('phi', 'mean'),
        'max_velocity': ('velocity', 'max'),
        'mean_velocity': ('velocity', 'mean'),
        'dist': ('dist', 'sum'),
        'has_interacted': ('has_interacted', 'sum'),
        'beam_cross': ('beam_cross', 'sum')
    }

    d_small = dt.groupby('t_round').agg(**agg_dict)

    # Classify movement types using configurable thresholds
    d_small['moving'] = d_small['max_velocity'] > velocity_threshold
    d_small['micro'] = (d_small['max_velocity'] > velocity_threshold) & (d_small['max_velocity'] < walk_threshold)
    d_small['walk'] = d_small['max_velocity'] > walk_threshold
    
    d_small.rename_axis('t', inplace = True)
    d_small.reset_index(level=0, inplace=True)

    return d_small

def prep_data_motion_detector(data: pd.DataFrame,
                            required_columns: List[str],
                            time_window_length: int = 10,
                            optional_columns: Optional[List[str]] = None 
                            ) -> pd.DataFrame:
    """
    Prepare raw ethoscope data for motion detection analysis.
    
    Bins time series data into specified windows and validates data quality.

    Args:
        data (pd.DataFrame): Raw ethoscope tracking data
        required_columns (List[str]): Column names that must be present
        time_window_length (int, optional): Size of time bins in seconds. Default is 10.
        optional_columns (List[str], optional): Additional columns to include. Default is None.
    
    Returns:
        pd.DataFrame: Binned and curated data ready for motion analysis
        
    Raises:
        KeyError: If required columns are missing from input data
    """
    
    # Validate required columns exist
    if not all(col in data.columns.values for col in required_columns):
        raise KeyError(f'Data must contain these required columns: {required_columns}')

    # Add any valid optional columns to the required columns list
    if optional_columns is not None:
        present_optional_cols = set(data.columns).intersection(optional_columns)
        columns_to_keep = list(set(list(present_optional_cols) + required_columns))
    else:
        columns_to_keep = required_columns

    # Create copy with selected columns and bin by time window
    processed_data = copy.deepcopy(data[columns_to_keep])
    processed_data['t_round'] = processed_data['t'].map(
        lambda t: time_window_length * floor(t / time_window_length)
    )
    
    def remove_sparse_data(df: pd.DataFrame,
                          window_size: int = 60,
                          min_points: int = 20
                          ) -> pd.DataFrame:
        """
        Filters out time windows with insufficient data points.
        
        Args:
            df (pd.DataFrame): A dataframe containing time series data
            window_size (int, optional): Size of window in seconds to check for minimum points. Default is 60.
            min_points (int, optional): Minimum required points per window. Default is 20.
        
        Returns:
            pd.DataFrame: A dataframe with sparse windows removed
        """
        df['window_id'] = df['t'].map(lambda t: window_size * floor(t / window_size))
        df['points_in_window'] = df.groupby(['window_id'])['window_id'].transform('count')
        df = df[df.points_in_window > min_points]
        df.drop(columns=['window_id', 'points_in_window'], inplace=True)
        return df

    processed_data = remove_sparse_data(processed_data)
    return processed_data

def sleep_annotation(data: pd.DataFrame, 
                    time_window_length: int = 10,
                    min_sleep_duration: int = 300,
                    motion_detector_function: callable = max_velocity_detector,
                    masking_duration: int = 6,
                    velocity_correction_coef: float = 3e-3
                    ) -> Optional[pd.DataFrame]:
    """
    Analyze movement data to identify sleep periods based on sustained immobility.
    
    Sleep is defined as continuous immobility exceeding a minimum duration threshold.

    Args:
        data (pd.DataFrame): Raw tracking data from a single animal
        time_window_length (int, optional): Duration of time bins in seconds. Default is 10.
        min_sleep_duration (int, optional): Minimum immobility duration for sleep. Default is 300.
        motion_detector_function (callable, optional): Function to classify movement. Default is max_velocity_detector.
        masking_duration (int, optional): Duration to ignore movement after stimulus. Default is 6.
        velocity_correction_coef (float, optional): Coefficient for velocity calculations. Default is 3e-3.
            
    Returns:
        Optional[pd.DataFrame]: DataFrame with movement and sleep classifications or None if insufficient data
    """
    # Check minimum data requirements
    if len(data.index) < 100:
        return None
    
    # Get movement classifications
    binned_data = motion_detector_function(
        data, 
        time_window_length,
        masking_duration=masking_duration,
        velocity_correction_coef=velocity_correction_coef
    )

    if len(binned_data.index) < 100:
        return None

    # Create continuous time series and handle missing values
    time_range = pd.Series(
        range(binned_data.t.iloc[0], binned_data.t.iloc[-1] + time_window_length, time_window_length),
        name='t'
    )
    missing_times = time_range[~time_range.isin(binned_data['t'].tolist())]
    
    # Merge with original data and mark interpolated points
    binned_data = binned_data.merge(time_range, how='right', on='t', copy=False).sort_values(by=['t'])
    binned_data['is_interpolated'] = binned_data['t'].isin(missing_times)
    binned_data['moving'] = np.where(binned_data['is_interpolated'], False, binned_data['moving'])

    def classify_sleep(movement_data: pd.Series,
                      sampling_freq: float,
                      min_duration: int
                      ) -> List[bool]:
        """
        Identifies sleep periods based on sustained immobility.
        
        Args:
            movement_data: Series of boolean movement states
            sampling_freq: Data sampling frequency in Hz
            min_duration: Minimum immobility duration for sleep classification
            
        Returns:
            List of boolean sleep states for each timepoint
        """
        min_samples = sampling_freq * min_duration
        v, _, l = rle(np.logical_not(movement_data))

        # Convert to numpy arrays to ensure compatible types
        valid_sleep = np.array(l >= min_samples)
        sleep_states = np.logical_and(valid_sleep, np.array(v))

        # Expand run lengths back to original time series
        sleep_series = []
        for state, length in zip(sleep_states, l):
            sleep_series.extend([state] * length)
            
        return sleep_series

    binned_data['asleep'] = classify_sleep(
        binned_data['moving'], 
        1/time_window_length,
        min_duration=min_sleep_duration
    )
    
    return binned_data

# Functions for add_false option in stimulus response
def _find_runs(mov: np.ndarray, 
               time: np.ndarray, 
               dt: np.ndarray
               ) -> pd.DataFrame:
    """
    Find continuous runs of movement or non-movement states.

    Helper function to identify consecutive periods of similar movement states.

    Args:
        mov (np.ndarray): Boolean array indicating movement state
        time (np.ndarray): Array of timestamps
        dt (np.ndarray): Array of time differences between consecutive points

    Returns:
        pd.DataFrame: Movement runs with columns ['t', 'moving', 'activity_count', 'deltaT']
    """
    _, _, l = rle(mov)
    count_list = np.concatenate([[c] * cnt for c, cnt in enumerate(l)], dtype = int)
    return pd.DataFrame({'t' : time, 'moving' : mov, 'activity_count' : count_list, 'deltaT' : dt})

def cumsum_delta(dataframe: pd.DataFrame, 
                 immobility_int: int
                 ) -> pd.DataFrame:
    """
    Calculate cumulative sum of time deltas for immobility periods.

    Identifies periods where cumulative immobility exceeds specified threshold.

    Args:
        dataframe (pd.DataFrame): DataFrame containing movement data
        immobility_int (int): Immobility interval threshold in seconds

    Returns:
        pd.DataFrame: DataFrame containing rows where cumulative immobility exceeds the threshold
    """
    response_rows = []
    def internal_sum_delta(data):
        data['cumsum_delta'] = data['deltaT'].cumsum()
        for i in np.arange(immobility_int, data['cumsum_delta'].max(), immobility_int):
            response_dict = data[data['cumsum_delta'] >= i].iloc[0].to_dict()
            response_dict['new_has_interacted'] = 2
            response_rows.append(response_dict)
    dataframe.groupby('activity_count', group_keys=False, sort=False).apply(internal_sum_delta)
    return pd.DataFrame(response_rows)

def stimulus_response(data: pd.DataFrame, 
                     start_response_window: int = 0, 
                     response_window_length: int = 10, 
                     add_false: Union[bool, int] = False, 
                     velocity_correction_coef: float = 3e-3
                     ) -> Optional[pd.DataFrame]:
    """
    Find interaction times and detect responses in given windows.
    
    Analyzes movement responses to stimuli within specified time windows. This function only returns
    data from around interaction times and not whole movement data from the experiment.

    Args:
        data (pd.DataFrame): Behavioral variables from one or multiple animals
        start_response_window (int, optional): Start of response window in seconds. Default is 0.
        response_window_length (int, optional): Duration of response window in seconds. Default is 10.
        add_false (Union[bool, int], optional): If int, then the number of seconds of immobility before 
            a stimulus would be given. 
            For use with old datasets with no false interactions. Default is False.
        velocity_correction_coef (float, optional): Coefficient for velocity calculations. Default is 3e-3.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame with interaction times and responses or None if no interactions.
            Includes columns: interaction_t, has_responded, has_walked, t_rel, response_velocity

    Raises:
        ValueError: If start_response_window is greater than or equal to response_window_length
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

    if add_false is not False:

        if add_false <= 0:
            raise ValueError("add_false must be a positive integer") 
        # add a moving column
        data['moving'] = np.where(data['velocity'] > 1, True, False)
        # find continuous runs of either moving or immobile
        counted_df = _find_runs(data['moving'], data['t'], data['deltaT'])
        # for runs of immobility cumsum the detlta time and add false interactions at every interval of the immobility integer, i.e. every 30 seconds add 2
        new_int_df = cumsum_delta(dataframe=counted_df[counted_df['moving'] == False], immobility_int=add_false)
        data = pd.merge(data, new_int_df[['t', 'new_has_interacted']], how = 'left', on = 't')
        # integrate the new false interactions into the original column, but keeping the true ones
        data['has_interacted'] = np.where((data['new_has_interacted'] == 2) & (data['has_interacted'] == 0), 2, data['has_interacted'])
        data.drop(columns=['new_has_interacted'], inplace=True)

        # THE OLD ADD_FALSE - ADDED AT RANDOM FALSE INTERACTIONS GIVEN A % OF THE TOTAL
        # int_list = [2] * (int(len(data)*(add_false/100)))
        # int_list_2 = [0] * (len(data) - len(int_list))
        # int_list_all = int_list + int_list_2 
        # shuffle(int_list_all)
        # data['has_interacted2'] = int_list_all 
        # data['has_interacted'] = np.where(data['has_interacted'] == 1, data['has_interacted'], data['has_interacted2'])
        # data = data.drop(columns = ['has_interacted2'])

    data.drop(columns = ['deltaT', 'dist'], inplace = True)

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

    ## new method, but can be slow
    df = pd.concat([data[(data['t'] >= i) & (data['t'] < q)] for i, q in zip(starts, ends)])
    df = df.join(interaction_dt, rsuffix = '_int').ffill()

    # find relative time to interaction and check for movemokonomiyaki flourent
    df['t_rel'] = df.t - df.t_int
    df = df[(df['t_rel'] > start_response_window) | (df['t_rel'] == 0)]
    df.rename(columns = {'t_int' : 'interaction_t'}, inplace = True)
    df['has_responded'] = (df['t_rel'] > 0) & (df['velocity'] > 1)
    df['has_walked'] = (df['t_rel'] > 0) & (df['velocity'] > 2.5)
    df.drop(columns = ['xy_dist_log10x1000', 'start', 'end'], inplace = True)

    response_rows = []
    # is any response take the interaction row and change response to True and t_rel to time till movement
    def find_interactions(response_data):
        if any(response_data['has_responded']):
            response_dict = response_data[response_data['t_rel'] == 0].to_dict('records')[0]
            response_dict['has_responded'] = True
            response_dict['has_walked'] = response_data['has_walked'][response_data['has_responded'] == True].iloc[0]
            response_dict['t_rel'] = response_data['t_rel'][response_data['has_responded'] == True].iloc[0]
            response_dict['response_velocity'] = response_data['velocity'][response_data['has_responded'] == True].iloc[0]
            response_dict.pop('int_id')
            response_rows.append(response_dict)
        else:
            response_dict = response_data[response_data['t_rel'] == 0].to_dict('records')[0]
            response_dict.pop('int_id')
            response_rows.append(response_dict)

    df.groupby('int_id').apply(find_interactions)

    return pd.DataFrame(response_rows)

def stimulus_prior(data: pd.DataFrame, 
                  window: int = 300, 
                  response_window_length: int = 10, 
                  velocity_correction_coef: float = 3e-3
                  ) -> Optional[pd.DataFrame]:
    """
    Analyze behavioral data before stimulus interactions.
    
    Modified version of stimulus_response that focuses on pre-stimulus behavior. Takes data with 
    populated has_interacted column and analyzes variables recorded before interaction.

    Args:
        data (pd.DataFrame): Behavioral variables from one or multiple animals
        window (int, optional): Time period before stimulus in seconds. Default is 300.
        response_window_length (int, optional): Duration of response window in seconds. Default is 10.
        velocity_correction_coef (float, optional): Coefficient for velocity calculations. Default is 3e-3.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame with pre-stimulus behavior or None if no interactions.
            Each run is given a unique ID per fly (combine with fly ID for global uniqueness)

    Notes:
        To create globally unique run IDs, combine the returned run ID with the fly ID after function call
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
    interaction_dt['end'] = interaction_dt.int_t + response_window_length

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
    df['has_responded'] = (df['t_rel'] > 0) & (df['velocity'] > 1)
    df['has_walked'] = (df['t_rel'] > 0) & (df['velocity'] > 2.5)
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
                response_df = pd.concat([response_df, formatted_small], ignore_index=True)
        else:
            formatted_small = format_window(response = False)
            if formatted_small is not None:
                response_df = pd.concat([response_df, formatted_small], ignore_index=True)

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
#     data = data.groupby('t').agg(**{. T
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
import pandas as pd
import numpy as np 
import warnings
import pickle
import copy

from math import floor

# from ethoscopy.behavpy import check_conform
from ethoscopy.misc.format_warning import format_warning
from ethoscopy.analyse import max_velocity_detector
from ethoscopy.misc.rle import rle

class behavpy(pd.DataFrame):
    """
    The behavpy class is a store of information for data from the ethoscope system with corresponding methods to augment and manipulate,
    the data as necessary for standard analysis
    Behavpy is subclassed from the pandas dataframe object and can be manipualted using all their tools as well as the custom methods within
    Behavpy sets a metadata dataframe as an attribute which is called upon frequently in the methods, can be accessed through behavpy.meta
    Both metadata and data should share unique ids in their 'id' column that are essential for manipulaiton
    print(df) will only print the data df, to see both use the .display() method
    """
    warnings.formatwarning = format_warning

    # set meta as permenant attribute
    _metadata = ['meta']

    @property
    def _constructor(self):
        return behavpy

    def display(self):
        """
        Alternative to print(), displays both the metadata and data with corresponding headers

        returns a formatted pandas print
        """
        print('\n ==== METADATA ====\n\n{}\n ====== DATA ======\n\n{}'.format(self.meta, self))

    def xmv(self, column, *args):
        """
        Expand metavariable from the behavpy object

        Params:
        @column = string. column heading from the metadata of the behavpy object
        @*args = string, arguments corresponding to groups from the column given

        returns a behavpy object with filtered data and metadata
        """

        if column == 'id':
            index_list = []
            for m in args:
                if m not in self.meta.index.tolist():
                    warnings.warn(f'Metavariable "{m}" is not in the id column')
                    exit()
                else:
                    small_list = self.meta[self.meta.index == m].index.values
                    index_list.extend(small_list)

            # find interection of meta and data id incase metadata contains more id's than in data
            data_id = list(set(self.index.values))
            new_index_list = np.intersect1d(index_list, data_id)

            xmv_df = behavpy(self[self.index.isin(index_list)])
            xmv_df.meta = self.meta[self.meta.index.isin(new_index_list)]

            return xmv_df

        if column not in self.meta.columns:
            warnings.warn(f'Column heading "{column}" is not in the metadata table')
            exit()
        
        index_list = []
        for m in args:
            if m not in self.meta[column].tolist():
                warnings.warn('Metavariable "{}" is not in the column'.format(m))
                exit()
            else:
                small_list = self.meta[self.meta[column] == m].index.values
                index_list.extend(small_list)

        # find interection of meta and data id incase metadata contains more id's than in data
        data_id = list(set(self.index.values))
        new_index_list = np.intersect1d(index_list, data_id)

        xmv_df = behavpy(self[self.index.isin(index_list)])
        xmv_df.meta = self.meta[self.meta.index.isin(new_index_list)]

        return xmv_df

    def t_filter(self, end_time = None, start_time = 0, t_column = 't'):
        """
        Filters the data to only be inbetween the provided start and end points
        argument is given in hours and converted to seconds

        Params:
        @end_time / start_time = integer. the time in hours you want to filter the behavpy object by
        @t_column = string. the column containing the timestamps, is 't' unless changed

        returns a filtered behapvy object
        """
        s_t = start_time * 60 * 60

        if end_time is not None:
            e_t = end_time * 60 * 60
            t_filter_df = self[(self[t_column] >= (s_t)) & (self[t_column] < (e_t))]
        else:
            t_filter_df = self[self[t_column] >= (s_t)]

        return t_filter_df

    # def rejoin(self, new_column):
    #     """
    #     Joins a new column to the metadata

    #     Params:
    #     @new_column = pandas dataframe. The column to be added, must contain an index called 'id' to match original metadata

    #     augments the metadata in place
    #     """

    #     check_conform(new_column)

    #     m = pd.DataFrame(self.meta)
    #     new_m = m.join(new_column, on = 'id')

    #     self.meta = new_m

    def concat(self, *args):
        """
        Wrapper for pd.concat that also concats metadata

        params:
        @*args = behavpy object. Behavpy tables to be concatenated to the original behavpy table

        returns a new instance of a combined behavpy object
        """

        meta_list = [self.meta]
        data_list = [self]

        for df in args:

            if isinstance(df, behavpy) is not True:
                warnings.warn('Object(s) to concat is(are) not a Behavpy object')
                exit()

            meta_list.append(df.meta)
            data_list.append(df)

        meta = pd.concat(meta_list)
        new = pd.concat(data_list) 

        new.meta = meta

        return new

    def pivot(self, column, function):
        """ 
        Wrapper for the groupby pandas method to split by groups in a column and apply a function to said groups

        params:
        @column = string. The name of the column in the data to pivot by
        @function = string or user defined function. The applied function to the grouped data, can be standard 'mean', 'max'.... ect, can also be a user defined function

        returns a pandas dataframe with the transformed grouped data
        """

        if column not in self.columns:
            warnings.warn(f'Column heading, "{column}", is not in the data table')
            exit()
            
        parse_name = f'{column}_{function}' # create new column name
        
        pivot = self.groupby(self.index).agg(**{
            parse_name : (column, function)    
        })

        return pivot

    def sleep_contiguous(self, mov_column = 'moving', min_valid_time = 300):
        """ 
        Checks for contiguous bouts of non-movement, those larger than the threshold are classed as asleep

        params:
        @mov_column = string, default 'moving'. The name of the column containg the sleep boolean values in the data
        @min_valid_time = integer, default 300. The threshold for sleep in seconds, i.e. 300 = 5 mins

        returns a modified behavpy object with an added asleep column
        """

        t_delta = self['t'].iloc[1] - self['t'].iloc[0] 
        fs = 1/t_delta # sampling frequency
        moving = self[mov_column]

        min_len = fs * min_valid_time
        r_sleep =  rle(np.logical_not(moving)) 
        valid_runs = r_sleep[2] >= min_len 
        r_sleep_mod = valid_runs & r_sleep[0]
        r_small = []

        for c, i in enumerate(r_sleep_mod):
            r_small += ([i] * r_sleep[2][c])

        self['asleep'] = r_small

    def sleep_bout_analysis(self, sleep_column = 'asleep', as_hist = False, relative = True, min_bins = 30, asleep = True):
        """ 
        Augments a behavpy objects sleep column to have duration and start of the sleep bouts, must contain a column with boolean values for sleep

        params:
        @sleep_column = string, default 'asleep'. Name of column in the data containing sleep data as a boolean
        @as_hist = bool, default False. If true the data will be augmented further into data appropriate for a histogram 
        Subsequent params only apply if as_hist is True
        @relative = bool, default True. Changes frequency from absolute to proportional with 1 equalling 100%
        @min_bins = integer, default 30. The min bumber of bins for the data to sorted into for the histogram
        @asleep = bool, default True. If True the histogram represents sleep bouts, if false bouts of awake

        returns a behavpy object with duration and time start of both awake and asleep

        if as_hist is True:
        returns a behavpy object with bins, count, and prob as columns
        """

        if sleep_column not in self.columns:
            warnings.warn(f'Column heading "{sleep_column}", is not in the data table')
            exit()

        def wrapped_bout_analysis(data, var_name = sleep_column, as_hist = as_hist, relative = relative, min_bins = min_bins, asleep = asleep):

            index_name = data.index[0]
            
            dt = copy.deepcopy(data[['t',var_name]])
            dt['deltaT'] = dt.t.diff()
            bout_rle = rle(dt[var_name])
            vals = bout_rle[0]
            bout_range = list(range(1,len(vals)+1))

            bout_id = []
            for c, i in enumerate(bout_range):
                bout_id += ([i] * bout_rle[2][c])

            bout_df = pd.DataFrame({'bout_id' : bout_id, 'deltaT' : dt['deltaT']})
            bout_times = bout_df.groupby('bout_id').agg(
            duration = pd.NamedAgg(column='deltaT', aggfunc='sum')
            )
            bout_times[var_name] = vals
            time = list(np.cumsum(pd.Series(0).append(bout_times['duration'].iloc[:-1])) + dt.t.iloc[0])
            bout_times['t'] = time
            bout_times.reset_index(level=0, inplace=True)
            bout_times.drop(columns = ['bout_id'], inplace = True)
            old_index = pd.Index([index_name] * len(bout_times.index), name = 'id')
            bout_times.set_index(old_index, inplace =True)

            if as_hist is True:

                filtered = bout_times[bout_times[var_name] == asleep]

                breaks = list(range(0, min_bins*60, 60))
                bout_cut = pd.DataFrame(pd.cut(filtered.duration, breaks, right = False, labels = breaks[1:]))
                bout_gb = bout_cut.groupby('duration').agg(
                count = pd.NamedAgg(column = 'duration', aggfunc = 'count')
                )
                if relative is True:
                    bout_gb['prob'] = bout_gb['count'] / bout_gb['count'].sum()
                bout_gb.rename_axis('bins', inplace = True)
                bout_gb.reset_index(level=0, inplace=True)
                old_index = pd.Index([index_name] * len(bout_gb.index), name = 'id')
                bout_gb.set_index(old_index, inplace =True)

                return bout_gb

            else:
                return bout_times

        bout_df = behavpy(self.groupby('id', group_keys = False).apply(wrapped_bout_analysis))
        bout_df.meta = self.meta

        return bout_df

    def curate_dead_animals(self, t_column = 't', mov_column = 'moving', time_window = 24, prop_immobile = 0.01, resolution = 24):
        
        """ 
        This function detects when individuals have died based on their first (very) long bout of immobility and removes that and subsequent data

        Params:
        @t_column = string, column heading for the data frames time stamp column (default is 't')
        @mov_column string, logical variable in `data` used to define the moving (alive) state (default is `moving`)
        @time_window = int, window during which to define death 
        @prop_immobile = float, proportion of immobility that counts as "dead" during time_window 
        @resolution = int, how much scanning windows overlap. Expressed as a factor. 

        Returns a behavpy object
        """

        if t_column not in self.columns.tolist():
            warnings.warn('Variable name entered, {}, for t_column is not a column heading!'.format(t_column))
            exit()
        
        if mov_column not in self.columns.tolist():
            warnings.warn('Variable name entered, {}, for mov_column is not a column heading!'.format(mov_column))
            exit()

        def wrapped_curate_dead_animals(data, 
                                        time_var = t_column,
                                        moving_var = mov_column,
                                        time_window = time_window, 
                                        prop_immobile = prop_immobile,
                                        resolution = resolution): 
            time_window = (60 * 60 * time_window)

            d = data[[time_var, moving_var]]
            target_t = np.array(list(range(d.t.min().astype(int), d.t.max().astype(int), floor(time_window / resolution))))
            local_means = np.array([d[d[time_var].between(i, i + time_window)][moving_var].mean() for i in target_t])

            first_death_point = np.where(local_means <= prop_immobile, True, False)

            if any(first_death_point) is False:
                return data

            last_valid_point = target_t[first_death_point]

            curated_data = data[data[time_var].between(data.t.min(), last_valid_point[0])]
            return curated_data

        curated_df = behavpy(self.groupby('id', group_keys = False).apply(wrapped_curate_dead_animals))
        curated_df.meta = self.meta 

        return curated_df

    def bin_time(self, column, bin_secs, t_column = 't', function = 'mean'):
        """
        Bin the time series data into entered groups, pivot by the time series column and apply a function to the selected columns.
        
        Params:
        @column = string, column in the data that you want to the function to be applied to post pivot
        @bin_secs = float, the amount of time you want in each bin in seconds, e.g. 60 would be bins for every minutes
        @t_column = string, column same for the time series data that you want to group and pivot by
        @function = string or user defined function. The applied function to the grouped data, can be standard 'mean', 'max'.... ect, can also be a user defined function
        
        returns a behavpy object with a single data column
        """

        from math import floor

        if column not in self.columns:
            warnings.warn('Column heading "{}", is not in the data table'.format(column))
            exit()

        def wrapped_bin_data(data, column = column, bin_column = t_column, function = function, bin_secs = bin_secs):

            index_name = data.index[0]

            data[bin_column] = data[bin_column].map(lambda t: bin_secs * floor(t / bin_secs))

            output_parse_name = '{}_{}'.format(column, function) # create new column name
        
            bout_gb = data.groupby(bin_column).agg(**{
                output_parse_name : (column, function)    
            })

            bin_parse_name = '{}_bin'.format(bin_column)

            bout_gb.rename_axis(bin_parse_name, inplace = True)
            bout_gb.reset_index(level=0, inplace=True)
            old_index = pd.Index([index_name] * len(bout_gb.index), name = 'id')
            bout_gb.set_index(old_index, inplace =True)

            return bout_gb

        bin_df = behavpy(self.groupby('id', group_keys = False).apply(wrapped_bin_data))
        bin_df.meta = self.meta

        return bin_df

    def summary(self, detailed = False):
        """ 
        Prints a table with summary statistics of metadata and data counts.
            
        Params:
        @detailed = bool , if detailed is True count and range of data points will be broken down per 'id' 
            
        returns None
        """

        def print_table(table):
            longest_cols = [
                (max([len(str(row[i])) for row in table]) + 3)
                for i in range(len(table[0]))
            ]
            row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
            for row in table:
                print(row_format.format(*row))

        if detailed is False:
            individuals = len(self.meta.index)
            metavariable = len(self.meta.columns)
            variables = len(self.columns)
            measurements = len(self.index)
            table = [
                ['individuals', individuals],
                ['metavariable', metavariable],
                ['variables', variables],
                ['measurements', measurements],
            ]
            print('behavpy table with: ')
            print_table(table)

        if detailed is True:

            def time_range(data):
                return (str(min(data)) + '  ->  ' + str(max(data)))


            group = self.groupby('id').agg(
                data_points = pd.NamedAgg(column = 't', aggfunc = 'count'),
                time_range = pd.NamedAgg(column = 't', aggfunc = time_range)
            )

            print(group)

    def add_day_phase(self, reference_hour = None):
        """ 
        Adds a column called 'phase' with either light or dark as catergories according to its time compared to the reference hour
        Adds a column with the day the row in, starting with 1 as day zero and increasing sequentially.

        Params:
        @reference_hour = int, a number in the 24 hour clock from when the lights turn on first in the day
            
        returns the orignal behapvy object with added columns to the data column 
        """

        from math import floor

        self['day'] = self['t'].map(lambda t: floor(t / 86400))
        
        if reference_hour is None:
            self['phase'] = np.where(((self.t % 86400) > 43200), 'dark', 'light')
            self['phase'] = self['phase'].astype('category')

    def motion_detector(self, time_window_length = 10, velocity_correction_coef = 3e-3, masking_duration = 0, optional_columns = None):
        """
        Method version of the motion detector without sleep annotation varaiables.
        Max_velocity_detector is the default movement classification for real-time ethoscope experiments.
        It is benchmarked against human-generated ground truth.
        See function for paramater details.
        
        returns a behavpy object with added columns like 'moving' and 'beam_crosses'
        """
        from motion_detectors import max_velocity_detector
        if optional_columns is not None:
            if optional_columns not in self.columns:
                warnings.warn('Column heading "{}", is not in the data table'.format(optional_columns))
                exit()

        def wrapped_motion_detector(data, 
                                    time_window_length = time_window_length, 
                                    velocity_correction_coef = velocity_correction_coef, 
                                    masking_duration = masking_duration, 
                                    optional_columns = optional_columns):
            
            index_name = data.index[0]
            
            df = max_velocity_detector(data,                                   
                                    time_window_length = time_window_length, 
                                    velocity_correction_coef = velocity_correction_coef, 
                                    masking_duration = masking_duration, 
                                    optional_columns = optional_columns)

            old_index = pd.Index([index_name] * len(df.index), name = 'id')
            df.set_index(old_index, inplace =True)  

            return df                     

        motion_df = behavpy(self.groupby('id', group_keys = False).apply(wrapped_motion_detector))
        motion_df.meta = self.meta

        return  motion_df

    def sleep_annotation(self, time_window_length = 10, min_time_immobile = 300, motion_detector_FUN = max_velocity_detector, masking_duration = 0):
        """
        Method version of the sleep annotation function.
        This function first uses a motion classifier to decide whether an animal is moving during a given time window.
        Then, it defines sleep as contiguous immobility for a minimum duration.
        See function for paramater details

        returns a behavpy object with added columns like 'moving' and 'asleep'
        """

        def wrapped_sleep_annotation(data, 
                                    time_window_length = time_window_length, 
                                    min_time_immobile = min_time_immobile, 
                                    motion_detector_FUN = motion_detector_FUN, 
                                    masking_duration = masking_duration):
            
            from sleep_annotation import sleep_annotation
            
            index_name = data.index[0]
            
            df = sleep_annotation(data,                                   
                                    time_window_length = time_window_length, 
                                    min_time_immobile = min_time_immobile, 
                                    motion_detector_FUN = motion_detector_FUN, 
                                    masking_duration = masking_duration)

            old_index = pd.Index([index_name] * len(df.index), name = 'id')
            df.set_index(old_index, inplace =True)  

            return df                     

        sleep_df = behavpy(self.groupby('id', group_keys = False).apply(wrapped_sleep_annotation))
        sleep_df.meta = self.meta

        return sleep_df

    def wrap_time(self, wrap_time = 24, time_column = 't'):
        """
        Replaces linear values of time in column 't' with a value which is a decimal of the wrap_time input

        Params:
        @wrap_time = int, time in hours you want to wrap the time series by, default is 24 hours
        @time_column  = string, column title for the time series column, default is 't'

        returns a modified version of the given behavpy table
        """

        hours_in_seconds = wrap_time * 60 * 60
        self[time_column] = self[time_column].map(lambda t: t % hours_in_seconds)

    def hmm_train(self, states, observables, random = True, trans_probs = None, emiss_probs = None, start_probs = None, mov_column = 'moving', iterations = 10, hmm_iterations = 100, tol = 50, t_column = 't', bin_time = 60, file_name = '', verbose = False):
        """
        Behavpy wrapper for the hmmlearn package which generates a Hidden Markov Model using the movement data from ethoscope data.
        If users want a restricted framework ,
        E.g. for random:

        Resultant hidden markov models will be saved as a .pkl file if file_name is provided
        Final trained model probability matrices will be printed to terminal at the end of the run time

        Params:
        @states = list of sting(s), names of hidden states for the model to train to
        @observables = list of string(s), names of the observable states for the model to train to.
        The length must be the same number as the different categories in you movement column.
        @random =  bool, if True the input starting transition matrix is randomised, see example for how to structure
        @trans_probs = numpy array, transtion probability matrix with shape 'len(states) x len(states)', 0's restrict the model from training any tranisitons between those states
        @emiss_probs = numpy array, emission probability matrix with shape 'len(observables) x len(observables)', 0's same as above
        @start_probs = numpy array, starting probability matrix with shape 'len(states) x 0', 0's same as above
        @mov_column = string, name for the column containing the movement data to train the model with, default is 'moving'
        @iterations = int, only used if random is True, number of loops using a different randomised starting matrices, default is 10
        @hmm_iterations = int, argument to be passed to hmmlearn, number of iterations of parameter updating without reaching tol before it stops, default is 100
        @tol = int, convergence threshold, EM will stop if the gain in log-likelihood is below this value, default is 50
        @t_column = string, name for the column containing the time series data, default is 't'
        @bin_time = int, the time in seconds the data will be binned to before the training begins, default is 60 (i.e 1 min)
        @file_name = string, name of the .pkl file the resultant trained model will be saved to, if left as '' and random is False the model won't be saved, default is ''
        @verbose = False, bool, argument for hmmlearn, whether per-iteration convergence reports are printed to terminal

        returns a trained hmmlearn HMM Multinomial object
        """

        from tabulate import tabulate
        from hmmlearn import hmm

        warnings.filterwarnings('ignore')

        n_states = len(states)
        n_obs = len(observables)

        t_delta = self[t_column].iloc[1] - self[t_column].iloc[0]

        def bin_to_list(data, t_var, mov_var, t_delta, bin):
            """ 
            Bins the time to the given integer and creates a nested list of the movement column by id
            """
            if t_delta != bin:
                data[t_var] = data[t_var].map(lambda t: bin * floor(t / bin))
                bin_gb = self.groupby(['id', t_var]).agg(**{
                    'moving' : ('moving', 'max')
                })
                bin_gb.reset_index(level = 1, inplace = True)
                gb = np.array(bin_gb.groupby('id')[mov_var].apply(list).tolist(), dtype = 'object')

            else:
                gb = np.array(self.groupby('id')[mov_var].apply(list).tolist(), dtype = 'object')

            return gb

        def hmm_table(start_prob, trans_prob, emission_prob, state_names, observable_names):
            """ 
            Prints a formatted table of the probabilities from a hmmlearn MultinomialHMM object
            """
            df_s = pd.DataFrame(start_prob)
            df_s = df_s.T
            df_s.columns = states
            print("Starting probabilty table: ")
            print(tabulate(df_s, headers = 'keys', tablefmt = "github") + "\n")
            print("Transition probabilty table: ")
            df_t = pd.DataFrame(trans_prob, index = state_names, columns = state_names)
            print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
            print("Emission probabilty table: ")
            df_e = pd.DataFrame(emission_prob, index = state_names, columns = observable_names)
            print(tabulate(df_e, headers = 'keys', tablefmt = "github") + "\n")

        if mov_column == 'beam_crosses':
            self['active'] = np.where(self[mov_column] == 0, 0, 1)
            gb = bin_to_list(self, t_var = t_column, mov_var = mov_column, t_delta = t_delta, bin = bin_time)

        else:
            self[mov_column] = np.where(self[mov_column] == True, 1, 0)
            gb = bin_to_list(self, t_var = t_column, mov_var = mov_column, t_delta = t_delta, bin = bin_time)

        len_seq = []
        for i in gb:
            len_seq.append(len(i))

        seq = np.concatenate(gb, 0)
        seq = seq.reshape(-1, 1)
    
        if random == True:

            if file_name == '':
                warnings.warn('enter a file name and type (.pkl) for the hmm object to be saved under')
                exit()

            h = hmm.MultinomialHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', init_params = 's')

            for i in range(iterations):
                print(f"Iteration {i+1} of {iterations}")

                a = np.random.random(3)
                a /= a.sum()
                a = np.append(a, 0)
                b = np.random.random(3)
                b /= b.sum()
                b = np.append(b, 0)
                c = np.random.random(3)
                c /= c.sum()
                c = np.insert(c, 0, 0)
                d = np.random.random(2)
                d /= d.sum()
                d = np.insert(d, [0, 0 ], [0, 0])
                t_prob = np.array([a, b, c, d])

                a = np.random.random(2)
                a /= a.sum()
                b = np.random.random(2)
                b /= b.sum()
                em_prob = np.array([[1, 0], [1, 0], a, b])

                # set initial probability parameters
                h.transmat_ = t_prob

                if emiss_probs is None:
                    h.emissionprob_ = em_prob
                else:
                    h.emissionprob_ = emiss_probs

                h.n_features = n_obs # number of emission states

                # call the fit function on the dataset input
                h.fit(seq, len_seq)

                # Boolean output of if the number of runs convererged on set of appropriate probabilites for s, t, an e
                print("True Convergence:" + str(h.monitor_.history[-1] - h.monitor_.history[-2] < h.monitor_.tol))
                print("Final log liklihood score:" + str(h.score(seq, len_seq)))

                if i == 0:
                    with open(file_name, "wb") as file: pickle.dump(h, file)

                else:
                    with open(file_name, "rb") as file: 
                        h_old = pickle.load(file)
                    if h.score(seq, len_seq) > h_old.score(seq, len_seq):
                        print('New Matrix:')
                        df_t = pd.DataFrame(h.transmat_, index = states, columns = states)
                        print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
                        with open(file_name, "wb") as file: pickle.dump(h, file)

                if i+1 == iterations:
                    with open(file_name, "rb") as file: 
                        h = pickle.load(file)
                    #print tables of trained emission probabilties, not accessible as objects for the user
                    hmm_table(start_prob = h.startprob_, trans_prob = h.transmat_, emission_prob = h.emissionprob_, state_names = states, observavble_names = observables)

                return h

        else:
            init_params = ''

            if start_probs is None:
                init_params += 's'

            if trans_probs is None:
                init_params += 't'

            if emiss_probs is None:
                init_params += 'e'

            h = hmm.MultinomialHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', init_params = init_params)

            # set initial probability parameters
            if start_probs is not None:
                h.startprob_ = start_probs

            if trans_probs is not None:
                h.transmat_ = trans_probs

            if emiss_probs is not None:
                h.emissionprob_ = emiss_probs
        

            h.monitor_. verbose = verbose # prints to screen live updates of tol score
            h.n_features = n_obs # number of emission states
        
            # call the fit function on the dataset input
            h.fit(seq, len_seq)

            # Boolean output of if the number of runs convererged on set appropriate probabilites for s, t, an e
            print("Convergence: " + str(h.monitor_.converged) + "\n")

            # print tables of trained emission probabilties, not accessible as objects for the user
            hmm_table(start_prob = h.startprob_, trans_prob = h.transmat_, emission_prob = h.emissionprob_, state_names = states, observavble_names = observables)

            # if cache is true a .pkl file will be saved to the working directory with the date and time of the first entry in the metadata table
            if len(file_name) > 0:
                with open(file_name, "wb") as file: pickle.dump(h, file)

            return h

    def baseline(self, column, t_column = 't', inplace = False):
        """
        A function to add days to the time series data per animal so allign interaction times per user discretion

        Params:
        @column = string, name of column containing the number of days to add, must in integers, 0 = no days added
        @t_column = string name of column containing the time series data to be modified, default is 't'
        @inplace = bool, if True the method changes the behavpy object inplace, if False returns a behavpy object
        
        returns a behavpy table with modifed time series columns
        """

        if column not in self.meta.columns:
            warnings.warn('Baseline days column: "{}", is not in the metadata table'.format(column))
            exit()

        dict = self.meta[column].to_dict()

        def d2s(x):
            id = x.name
            seconds = dict.get(id) * 86400
            return x[t_column] + seconds

        if inplace is True:
            self[t_column] = self.apply(d2s, axis = 1)

        else:
            new = copy.deepcopy(self)
            new[t_column] = new.apply(d2s, axis = 1)

            return new

    def heatmap(self, mov_column = 'moving'):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals using plotly
        
        Params:
        @mov_column = string, name for the column containing the movement data to plot, default is 'moving'
        
        returns None
        """
        import plotly.graph_objs as go 

        # change movement values from boolean to intergers and bin to 30 mins finding the mean
        self[mov_column] = np.where(self[mov_column] == True, 1, 0)
        self = self.bin_time(column = mov_column, bin_secs = 1800)
        self['t_bin'] = self['t_bin'] / (60*60)
        # create an array starting with the earliest half hour bin and the last with 0.5 intervals
        start = self['t_bin'].min().astype(int)
        end = self['t_bin'].max().astype(int)
        time_list = np.array([x / 10 for x in range(start*10, end*10+5, 5)])
        time_map = pd.Series(time_list, 
                    name = 't_bin')

        def align_data(data):
            """merge the individual fly groups time with the time map, filling in missing points with NaN values"""

            index_name = data.index[0]

            df = data.merge(time_map, how = 'right', on = 't_bin', copy = False).sort_values(by=['t_bin'])

            # read the old id index lost in the merge
            old_index = pd.Index([index_name] * len(df.index), name = 'id')
            df.set_index(old_index, inplace =True)  

            return df                    

        heatmap_df = self.groupby('id', group_keys = False).apply(align_data)

        gbm = heatmap_df.groupby(heatmap_df.index)['moving_mean'].apply(list)
        id = heatmap_df.groupby(heatmap_df.index)['t_bin'].mean().index.tolist()

        fig = go.Figure(data=go.Heatmap(
                        z = gbm,
                        x = time_list,
                        y = id,
                        colorscale = 'Viridis'))

        fig.update_layout(
            xaxis = dict(
                zeroline = False,
                color = 'black',
                linecolor = 'black',
                gridcolor = 'black',
                title = dict(
                    text = 'ZT Time (Hours)',
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                tick0 = 0,
                dtick = 12,
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 16
                ),
                linewidth = 2)
                )

        fig.show()

    def remove(self, column, *args):
        """ 
        A variation of xmv to remove all rows from a data table whose ID matches those specified from the metadata

        Params:
        @column = string. column heading from the metadata of the behavpy object
        @*args = string, arguments corresponding to groups from the column given

        returns a behavpy object with filtered data and metadata
        """

        if column == 'id':
            remove_index_list = []
            for m in args:
                if m not in self.meta.index.tolist():
                    warnings.warn('Metavariable "{}" is not in the id column'.format(m))
                    exit()
                else:
                    small_list = self.meta[self.meta.index == m].index.values
                    remove_index_list.extend(small_list)
            index_list = [x for x in self.meta.index.tolist() if x not in remove_index_list]

            # find interection of meta and data id incase metadata contains more id's than in data
            data_id = list(set(self.index.values))
            new_index_list = np.intersect1d(index_list, data_id)

            xmv_df = behavpy(self[self.index.isin(index_list)])
            xmv_df.meta = self.meta[self.meta.index.isin(new_index_list)]

            return xmv_df

        if column not in self.meta.columns:
            warnings.warn('Column heading "{}" is not in the metadata table'.format(column))
            exit()
        
        index_list = []
        for m in args:
            if m not in self.meta[column].tolist():
                warnings.warn('Metavariable "{}" is not in the column'.format(m))
                exit()
            else:
                small_list = self.meta[self.meta[column] != m].index.values
                index_list.extend(small_list)

        # find interection of meta and data id incase metadata contains more id's than in data
        data_id = list(set(self.index.values))
        new_index_list = np.intersect1d(index_list, data_id)

        xmv_df = behavpy(self[self.index.isin(index_list)])
        xmv_df.meta = self.meta[self.meta.index.isin(new_index_list)]

        return xmv_df






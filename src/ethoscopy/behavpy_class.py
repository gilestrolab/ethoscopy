import pandas as pd
import numpy as np 
import warnings
import copy
import plotly.graph_objs as go 
import plotly.express as px
from math import floor, ceil
from sys import exit
from scipy.stats import zscore

from ethoscopy.misc.format_warning import format_warning
from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.analyse import max_velocity_detector, sleep_annotation
from ethoscopy.misc.rle import rle
from ethoscopy.misc.bootstrap_CI import bootstrap

class behavpy(pd.DataFrame):
    """
    The behavpy class is a store of information for data from the ethoscope system with corresponding methods to augment and manipulate
    the data as necessary for standard analysis.
    Behavpy is subclassed from the pandas dataframe object and can be manipualted using all their tools as well as the custom methods within
    Behavpy sets a metadata dataframe as an attribute which is called upon frequently in the methods, can be accessed through behavpy.meta
    Both metadata and data should share unique ids in their 'id' column that are essential for manipulaiton
    print(df) will only print the data df, to see both use the .display() method
    """
    warnings.formatwarning = format_warning

    @staticmethod
    def _check_conform(self):
            """ 
            Checks the data augument is a pandas dataframe
            If metadata is provided and skip is False it will check as above and check the ID's in
            metadata match those in the data
            params: 
            @skip = boolean indicating whether to skip a check that unique id's are in both meta and data match 
            """
            
            # formats warming method to not double print and allow string formatting
            warnings.formatwarning = format_warning

            if isinstance(self.meta, pd.DataFrame) is not True:
                warnings.warn('Metadata input is not a pandas dataframe')
                exit()

            if self.index.name != 'id':
                try:
                    self.set_index('id', inplace = True)
                except:
                    warnings.warn("There is no 'id' as a column or index in the data'")
                    exit()

            if self.meta.index.name != 'id':
                try:
                    self.meta.set_index('id', inplace = True)
                except:
                    warnings.warn("There is no 'id' as a column or index in the metadata'")
                    exit()

            metadata_id_list = set(self.meta.index.tolist())
            data_id_list = set(self.index.tolist())
            # checks if all id's of data are in the metadata dataframe
            check_data = all(elem in metadata_id_list for elem in data_id_list)
            if check_data is not True:
                warnings.warn("There are ID's in the data that are not in the metadata, please check. You can skip this process by changing the parameter skip to False")
                exit()

    # set meta as permenant attribute
    _metadata = ['meta']

    @property
    def _constructor(self):
        return behavpy._internal_constructor(self.__class__)

    class _internal_constructor(object):
        def __init__(self, cls):
            self.cls = cls

        def __call__(self, *args, **kwargs):
            kwargs['meta'] = None
            return self.cls(*args, **kwargs)

        def _from_axes(self, *args, **kwargs):
            return self.cls._from_axes(*args, **kwargs)

    def __init__(self, data, meta, check = False, index= None, columns=None, dtype=None, copy=True):
        super(behavpy, self).__init__(data=data,
                                        index=index,
                                        columns=columns,
                                        dtype=dtype,
                                        copy=copy)

        self.meta = meta   

        if check is True:
            self._check_conform(self)
        
    _colours_small = px.colors.qualitative.Safe
    _colours_large = px.colors.qualitative.Dark24

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
        @column = string, column heading from the metadata of the behavpy object
        @*args = string, arguments corresponding to groups from the column given

        returns a behavpy object with filtered data and metadata
        """

        if len(args) == 1:
            if type(args[0]) == list:
                args = args[0]

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

        return behavpy(self[self.index.isin(index_list)], meta = self.meta[self.meta.index.isin(new_index_list)])


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

    def rejoin(self, new_column, check = True):
        """
        Joins a new column to the metadata

        Params:
        @new_column = pandas dataframe. The column to be added, must contain an index called 'id' to match original metadata
        @check = bool. Whether or not to check if the ids in the data match the new column, default is True

        augments the metadata in place
        """

        if check is True:
            self._check_conform(self, new_column)

        m = pd.DataFrame(self.meta)
        new_m = m.join(new_column, on = 'id')

        self.meta = new_m

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

    def sleep_bout_analysis(self, sleep_column = 'asleep', as_hist = False, max_bins = 30, asleep = True):
        """ 
        Augments a behavpy objects sleep column to have duration and start of the sleep bouts, must contain a column with boolean values for sleep

        params:
        @sleep_column = string, default 'asleep'. Name of column in the data containing sleep data as a boolean
        @as_hist = bool, default False. If true the data will be augmented further into data appropriate for a histogram 
        Subsequent params only apply if as_hist is True
        @relative = bool, default True. Changes frequency from absolute to proportional with 1 equalling 100%
        @max_bins = integer, default 30. The number of bins for the data to be sorted into for the histogram, each bin is 1 minute
        @asleep = bool, default True. If True the histogram represents sleep bouts, if false bouts of awake

        returns a behavpy object with duration and time start of both awake and asleep

        if as_hist is True:
        returns a behavpy object with bins, count, and prob as columns
        """

        if sleep_column not in self.columns:
            warnings.warn(f'Column heading "{sleep_column}", is not in the data table')
            exit()

        def wrapped_bout_analysis(data, var_name = sleep_column, as_hist = as_hist, max_bins = max_bins, asleep = asleep):

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

                breaks = list(range(0, max_bins*60, 60))
                bout_cut = pd.DataFrame(pd.cut(filtered.duration, breaks, right = False, labels = breaks[1:]))
                bout_gb = bout_cut.groupby('duration').agg(
                count = pd.NamedAgg(column = 'duration', aggfunc = 'count')
                )
                bout_gb['prob'] = bout_gb['count'] / bout_gb['count'].sum()
                bout_gb.rename_axis('bins', inplace = True)
                bout_gb.reset_index(level=0, inplace=True)
                old_index = pd.Index([index_name] * len(bout_gb.index), name = 'id')
                bout_gb.set_index(old_index, inplace =True)

                return bout_gb

            else:
                return bout_times

        self.reset_index(inplace = True)
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

        self.reset_index(inplace = True)
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
            
            output_parse_name = f'{column}_{function}' # create new column name
        
            bout_gb = data.groupby(bin_column).agg(**{
                output_parse_name : (column, function)    
            })

            bin_parse_name = f'{bin_column}_bin'
            bout_gb.rename_axis(bin_parse_name, inplace = True)
            bout_gb.reset_index(level=0, inplace=True)
            old_index = pd.Index([index_name] * len(bout_gb.index), name = 'id')
            bout_gb.set_index(old_index, inplace =True)

            return bout_gb

        self.reset_index(inplace = True)
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

    def add_day_phase(self, circadian_night = 12):
        """ 
        Adds a column called 'phase' with either light or dark as catergories according to its time compared to the reference hour
        Adds a column with the day the row in, starting with 1 as day zero and increasing sequentially.

        Params:
        @circadian_night = int, the ZT hour when the conditions shift to dark
            
        returns the orignal behapvy object with added columns to the data column 
        """

        from math import floor

        self['day'] = self['t'].map(lambda t: floor(t / 86400))
        
        night_in_secs = circadian_night * 60 * 60

        self['phase'] = np.where(((self.t % 86400) > night_in_secs), 'dark', 'light')
        self['phase'] = self['phase'].astype('category')

    def motion_detector(self, time_window_length = 10, velocity_correction_coef = 3e-3, masking_duration = 0, optional_columns = None):
        """
        Method version of the motion detector without sleep annotation varaiables.
        Max_velocity_detector is the default movement classification for real-time ethoscope experiments.
        It is benchmarked against human-generated ground truth.
        See function for paramater details.
        
        returns a behavpy object with added columns like 'moving' and 'beam_crosses'
        """

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
            
            index_name = data.index[0]
            
            df = sleep_annotation(data,                                   
                                    time_window_length = time_window_length, 
                                    min_time_immobile = min_time_immobile, 
                                    motion_detector_FUN = motion_detector_FUN, 
                                    masking_duration = masking_duration)

            old_index = pd.Index([index_name] * len(df.index), name = 'id')
            df.set_index(old_index, inplace =True)  

            return df    

        self.reset_index(inplace = True)
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

    def baseline(self, column, t_column = 't', inplace = False):
        """
        A function to add days to the time series data per animal so align interaction times per user discretion

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

    def heatmap(self, variable = 'moving'):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals using plotly
        
        Params:
        @variable = string, name for the column containing the variable of interest, the default is moving
        
        returns None
        """

        # change movement values from boolean to intergers and bin to 30 mins finding the mean
        if variable == 'moving':
            self[variable] = np.where(self[variable] == True, 1, 0)

        self = self.bin_time(column = variable, bin_secs = 1800)
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

        gbm = heatmap_df.groupby(heatmap_df.index)[f'{variable}_mean'].apply(list)
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

        if len(args) == 1:
            if type(args[0]) == list:
                args = args[0]

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

            return behavpy(self[self.index.isin(index_list)], self.meta[self.meta.index.isin(new_index_list)])

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

        return behavpy(self[self.index.isin(index_list)], self.meta[self.meta.index.isin(new_index_list)])

    def curate(self, length, t_delta):
        
        data_points = length * t_delta

        def wrapped_curate(data, limit = data_points):
            if len(data) < limit:
                id_list.append(list(set(data['id']))[0])
                return data
            else:
                return data

        id_list = []
        self.reset_index(inplace = True)
        df = self.groupby('id', group_keys = False).apply(wrapped_curate)
        df.set_index('id', inplace = True)
        df = behavpy(df, self.meta)

        return df.remove('id', id_list)

    def plot_overtime(self, variable, wrapped = False, facet_col = None, facet_args = None, labels = None, avg_window = 30, circadian_night = 12, save = False, location = ''):


        if facet_col is not None:
            if facet_col not in self.meta.columns:
                warnings.warn(f'Column "{facet_col}" is not a metadata column')
                exit()
            d_list = []
            if facet_args is None:
                arg_list = list(set(self.meta[facet_col].tolist()))
                for arg in arg_list:
                    temp_df = self.xmv(facet_col, arg)
                    d_list.append(temp_df)
                if labels is None:
                    labels = arg_list
                elif len(d_list) != len(labels):
                    labels = arg_list
            else:
                assert isinstance(facet_args, list)
                arg_list = facet_args
                for arg in arg_list:
                    temp_df = self.xmv(facet_col, arg)
                    d_list.append(temp_df)
                if labels is None:
                    labels = arg_list
                elif len(d_list) != len(labels):
                    labels = arg_list

        else:
            d_list = [self]
            labels = ['']

        if len(d_list) < 11:
            col_list = self._colours_small
        elif len(d_list) < 24:
            col_list = self._colours_large
        else:
            warnings.warn('Too many sub groups to plot with the current colour palette')
            exit()

        def pop_std(array):
            return np.std(array, ddof = 0)

        layout = go.Layout(
            yaxis = dict(
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = f'Probability of {variable}',
                    font = dict(
                        size = 24,
                    )
                ),
                range = [-0.025, 1], 
                tick0 = 0,
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                )
            ),
            xaxis = dict(
                color = 'black',
                linecolor = 'black',
                gridcolor = 'black',
                title = dict(
                    text = 'ZT (Hours)',
                    font = dict(
                        size = 24,
                        color = 'black'
                    )
                ),
                tick0 = 0,
                dtick = 12,
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                )
            ),
            plot_bgcolor = 'white',
            yaxis_showgrid=False,
            xaxis_showgrid = False,
            legend = dict(
                bgcolor = 'rgba(201, 201, 201, 1)',
                bordercolor = 'grey',
                font = dict(
                    size = 12
                ),
                x = 0.92,
                y = 0.99
            )
        )
        fig = go.Figure(layout = layout)

        min_t = []
        max_t = []
        
        max_var = []

        for data, name, col in zip(d_list, labels, col_list):

            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'grey'

            rolling_col = data.groupby(data.index, sort = False)[variable].rolling(avg_window).mean().reset_index(level = 0, drop = True)
            data['rolling'] = rolling_col.to_numpy()

            if wrapped is True:
                data['t'] = data['t'].map(lambda t: t % 86400)
            data['t'] = data['t'].map(lambda t: t / (60*60))

            t_min = int(circadian_night * floor(data.t.min() / circadian_night))
            min_t.append(t_min)
            t_max = int(12 * ceil(data.t.max() / 12)) 
            max_t.append(t_max)

            

            gb = data.groupby('t').agg(**{
                        'mean' : ('rolling', 'mean'), 
                        'SD' : ('rolling', pop_std),
                        'count' : ('rolling', 'count'),
                    })

            max_var.append(max(gb['mean']))

            gb['SE'] = (1.96*gb['SD']) / np.sqrt(gb['count'])
            gb['y_max'] = gb['mean'] + gb['SE']
            gb['y_min'] = gb['mean'] - gb['SE']

            y = gb['mean']
            y_upper = gb['y_max']
            y_lower = gb['y_min']
            x = gb.index.values

            upper_bound = go.Scatter(
            showlegend = False,
            x = x,
            y = y_upper,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0,
                    shape = 'spline'
                    ),
            )
            fig.add_trace(upper_bound)

            trace = go.Scatter(
            x = x,
            y = y,
            mode = 'lines',
            name = name,
            line = dict(
                shape = 'spline',
                color = col
                ),
            fill = 'tonexty'
            )
            fig.add_trace(trace)

            lower_bound = go.Scatter(
            showlegend = False,
            x = x,
            y = y_lower,
            mode='lines',
            marker=dict(color=col),
            line=dict(width = 0,
                    shape = 'spline'
                    ),
            fill = 'tonexty'
            )  
            fig.add_trace(lower_bound)

        # Light-Dark annotaion bars
        bar_shapes = circadian_bars(t_min, t_max, circadian_night = circadian_night)
        fig.update_layout(shapes=list(bar_shapes.values()))

        if max(max_var) > 1.01:
            fig['layout']['yaxis'].update(
                    range = [0, max_var]
                )

        if save is True:
            fig.write_image(location, width=1500, height=650)
            print(f'Saved to {location}')
            fig.show()
        else:
            fig.show()

    def plot_quantify(self, variable, facet_col = None, facet_args = None, labels = None, save = False, location = ''):

        if facet_col is not None:
            if facet_col not in self.meta.columns:
                warnings.warn(f'Column "{facet_col}" is not a metadata column')
                exit()
            d_list = []
            if facet_args is None:
                arg_list = list(set(self.meta[facet_col].tolist()))
                for arg in arg_list:
                    temp_df = self.xmv(facet_col, arg)
                    d_list.append(temp_df)
                if labels is None:
                    labels = arg_list
                elif len(d_list) != len(labels):
                    labels = arg_list
            else:
                assert isinstance(facet_args, list)
                arg_list = facet_args
                for arg in arg_list:
                    temp_df = self.xmv(facet_col, arg)
                    d_list.append(temp_df)
                if labels is None:
                    labels = arg_list
                elif len(d_list) != len(labels):
                    labels = arg_list

        else:
            d_list = [self]
            labels = ['']

        if len(d_list) < 11:
            col_list = self._colours_small
        elif len(d_list) < 24:
            col_list = self._colours_large
        else:
            warnings.warn('Too many sub groups to plot with the current colour palette')
            exit()
        
        layout = go.Layout(
            yaxis = dict(
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = f'Fraction of time spent {variable}',
                    font = dict(
                        size = 24,
                    )
                ),
                range = [0, 1.01], 
                tick0 = 0,
                dtick = 0.2,
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                )
            ),
            xaxis = dict(
                color = 'black',
                linecolor = 'black',
                gridcolor = 'black',
                title = dict(
                    font = dict(
                        size = 24,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                )
            ),
            plot_bgcolor = 'white',
            yaxis_showgrid=False,
            xaxis_showgrid = False,
        )

        fig = go.Figure(layout = layout)

        max_var = []

        for data, name, col in zip(d_list, labels, col_list):

            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'grey'

            data = data.reset_index()
            gdf = data.groupby('id').agg(**{
                                    'mean' : (variable, 'mean'),
                })
            zscore_list = gdf['mean'].to_numpy()[np.abs(zscore(gdf['mean'].to_numpy())) < 3]

            max_var.append(max(zscore_list))

            median_list = [np.mean(zscore_list)]
            q3_list = [bootstrap(zscore_list)[1]]
            q1_list = [bootstrap(zscore_list)[0]]

            trace_box = go.Box(
                showlegend = False,
                median = median_list,
                q3 = q3_list,
                q1 = q1_list,
                x = [name],
                # xaxis = f'x{state+1}',
                marker = dict(
                    color = col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = False,
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9,
            )
            fig.add_trace(trace_box)

            trace_box2 = go.Box(
                showlegend = False,
                y = zscore_list, 
                x = len(zscore_list) * [name],
                line = dict(
                    color = 'rgba(0,0,0,0)'
                ),
                fillcolor = 'rgba(0,0,0,0)',
                marker = dict(
                    color = col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = 'all',
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9
            )
            fig.add_trace(trace_box2)

        if max(max_var) > 1.01:
            fig['layout']['yaxis'].update(
                    range = [0, max_var]
                )

        if save is True:
            fig.write_image(location, width=1500, height=650)
            print(f'Saved to {location}')
            fig.show()
        else:
            fig.show()

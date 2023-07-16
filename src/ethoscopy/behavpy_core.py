import pandas as pd
import numpy as np 
import warnings

from math import floor
from sys import exit
from scipy.stats import zscore
from functools import partial
from scipy.interpolate import interp1d

from ethoscopy.misc.format_warning import format_warning
from ethoscopy.analyse import max_velocity_detector
from ethoscopy.misc.rle import rle
from ethoscopy.misc.bootstrap_CI import bootstrap

class behavpy_core(pd.DataFrame):
    """
    The behavpy class is a store of information for data from the ethoscope system with corresponding methods to augment and manipulate
    the data as necessary for standard analysis.
    Behavpy is subclassed from the pandas dataframe object and can be manipualted using all their tools as well as the custom methods within
    Behavpy sets a metadata dataframe as an attribute which is called upon frequently in the methods, can be accessed through behavpy.meta
    Both metadata and data should share unique ids in their 'id' column that are essential for manipulaiton
    print(df) will only print the data df, to see both use the .display() method

    Initialisation Parameters:
    @data = pandas DataFrame, the experimental recorded data usually loaded by the load_ethoscope function. Must contain an id column with unique IDs per speciemn
    and a time series column called 't' with time per recording in seconds 
    @meta = pandas Dataframe, the metadata i.e. conditions, genotypes ect of the experiment. There should be a unique row per ID in the data. 
    Usually generated from a csv file and link_meta_index function.
    @check = bool, when True this will check the ids in the data are in the metadata. If not an error will be raised. It also removes some columns that are no longer
    needed that are generated in link_meta_index.

    returns a behavpy object with methods to manipulate, analyse, and plot time series behavioural data

    """
    warnings.formatwarning = format_warning

    # set meta as permenant attribute
    _metadata = ['meta']
    _canvas = None

    @property
    def canvas(self):
        return self._canvas

    @property
    def _constructor(self):
        return behavpy_core._internal_constructor(self.__class__)

    class _internal_constructor(object):
        def __init__(self, cls):
            self.cls = cls

        def __call__(self, *args, **kwargs):
            kwargs['meta'] = None
            return self.cls(*args, **kwargs)

        def _from_axes(self, *args, **kwargs):
            return self.cls._from_axes(*args, **kwargs)

    def __init__(self, data, meta, check = False, index= None, columns=None, dtype=None, copy=True):
        super(behavpy_core, self).__init__(data=data,
                                        index=index,
                                        columns=columns,
                                        dtype=dtype,
                                        copy=copy)

        self.meta = meta   

        if check is True:
            self._check_conform(self)

    @staticmethod
    def _check_conform(dataframe):
        """ 
        Checks the data augument is a pandas dataframe
        If metadata is provided and skip is False it will check as above and check the ID's in
        metadata match those in the data
        """
        
        # formats warming method to not double print and allow string formatting
        warnings.formatwarning = format_warning

        if isinstance(dataframe.meta, pd.DataFrame) is not True:
            warnings.warn('Metadata input is not a pandas dataframe')
            exit()

        drop_col_names = ['path', 'file_name', 'file_size', 'machine_id']
        dataframe.meta = dataframe.meta.drop(columns=[col for col in dataframe.meta if col in drop_col_names])

        if dataframe.index.name != 'id':
            try:
                dataframe.set_index('id', inplace = True)
            except:
                warnings.warn("There is no 'id' as a column or index in the data'")
                exit()

        if dataframe.meta.index.name != 'id':
            try:
                dataframe.meta.set_index('id', inplace = True)
            except:
                warnings.warn("There is no 'id' as a column or index in the metadata'")
                exit()

        # checks if all id's of data are in the metadata dataframe
        check_data = all(elem in set(dataframe.meta.index.tolist()) for elem in set(dataframe.index.tolist()))
        if check_data is not True:
            warnings.warn("There are ID's in the data that are not in the metadata, please check")

    def _check_lists(self, f_col, f_arg, f_lab):
        """
        Check if the facet arguments match the labels or populate from the column if not.
        """

        if f_col is not None:
            if f_col not in self.meta.columns:
                warnings.warn(f'Column "{f_col}" is not a metadata column')
                exit()
            if f_arg is None:
                f_arg = list(set(self.meta[f_col].tolist()))
                string_args = []
                for i in f_arg:
                    string_args.append(str(i))
                if f_lab is None:
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    warnings.warn("The facet labels don't match the length of the variables in the column. Using column variables instead")
                    f_lab = string_args
            else:
                string_args = []
                for i in f_arg:
                    if i not in self.meta[f_col].tolist():
                        warnings.warn(f'Argument "{i}" is not in the meta column {f_col}')
                        exit()
                    string_args.append(str(i))
                if f_lab is None:
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    warnings.warn("The facet labels don't match the entered facet arguments in length. Using column variables instead")
                    f_lab = string_args
        else:
            f_arg = [None]
            if f_lab is None:
                f_lab = ['']

        return f_arg, f_lab

    @staticmethod
    def _check_boolean(lst):
        if np.nanmax(lst) == 1 and np.nanmin(lst) == 0:
            y_range = [-0.025, 1.01]
            dtick = 0.2
        else:
            y_range = False
            dtick = False
        return y_range, dtick


    @staticmethod
    def _zscore_bootstrap(array, second_array = None, min_max = False):
        """ calculate the z score of a given array, remove any values +- 3 SD"""
        try:
            if len(array) == 1 or all(array == array[0]):
                median = q3 = q1 = array[0]
                zlist = array
            else:
                zlist = array[np.abs(zscore(array)) < 3]
                if second_array is not None:
                    second_array = second_array[np.abs(zscore(array)) < 3] 
                median = zlist.mean()
                boot_array = bootstrap(zlist)
                q3 = boot_array[1]
                q1 = boot_array[0]

        except ZeroDivisionError:
            median = q3 = q1 = 0
            zlist = array

        if min_max == True:
            q3 = np.max(array)
            q1 = np.min(array)
        
        if second_array is not None:
            return median, q3, q1, zlist, second_array
        else:
            return median, q3, q1, zlist


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

        if type(args[0]) == list or type(args[0]) == np.array:
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
            self = self[self.index.isin(index_list)]
            self.meta = self.meta[self.meta.index.isin(new_index_list)]
            return self

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
        self = self[self.index.isin(index_list)]
        self.meta = self.meta[self.meta.index.isin(new_index_list)]
        return self

    def export(self):
        """
        Merges the current DataFrame with the metadata and converts the metadata columns to type 'category'. 
        
        Returns
        -------
        pandas.DataFrame
            The DataFrame with all metadata columns as type 'category'.
        """
        data = self.join(self.meta)
        data[self.meta.columns] = data[self.meta.columns].astype('category')

        return data


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
            check_data = all(elem in new_column.index.tolist() for elem in set(self.index.tolist()))
            if check_data is not True:
                warnings.warn("There are ID's in the data that are not in the metadata, please check. You can skip this process by changing the parameter skip to False")
                exit()

        m = self.meta.join(new_column, on = 'id')
        self.meta = m

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

            if isinstance(df, type(self)) is not True or isinstance(df, behavpy_core) is not True:
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

    @staticmethod
    def _wrapped_bout_analysis(data, var_name, as_hist, bin_size, max_bins, time_immobile, asleep, t_column = 't'):
        """ Finds runs of bouts of immobility or moving and sorts into a historgram per unqiue specimen in a behavpy dataframe"""

        index_name = data['id'].iloc[0]
        bin_width = bin_size*60
        
        dt = data[[t_column,var_name]].copy(deep = True)
        dt['deltaT'] = dt[t_column].diff()
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
        time = np.array([dt[t_column].iloc[0]])
        time = np.concatenate((time, bout_times['duration'].iloc[:-1]), axis = None)
        time = np.cumsum(time)
        bout_times[t_column] = time
        bout_times.reset_index(level=0, inplace=True)
        bout_times.drop(columns = ['bout_id'], inplace = True)
        old_index = pd.Index([index_name] * len(bout_times.index), name = 'id')
        bout_times.set_index(old_index, inplace =True)

        if as_hist is True:

            filtered = bout_times[bout_times[var_name] == asleep]
            filtered['duration_bin'] = filtered['duration'].map(lambda d: bin_width * floor(d / bin_width))
            bout_gb = filtered.groupby('duration_bin').agg(**{
                        'count' : ('duration_bin', 'count')
            })
            bout_gb['prob'] = bout_gb['count'] / bout_gb['count'].sum()
            bout_gb.rename_axis('bins', inplace = True)
            bout_gb.reset_index(level=0, inplace=True)
            old_index = pd.Index([index_name] * len(bout_gb.index), name = 'id')
            bout_gb.set_index(old_index, inplace =True)
            bout_gb = bout_gb[(bout_gb['bins'] >= time_immobile * 60) & (bout_gb['bins'] <= max_bins*bin_width)]

            return bout_gb

        else:
            return bout_times

    def sleep_bout_analysis(self, sleep_column = 'asleep', as_hist = False, bin_size = 1, max_bins = 30, time_immobile = 5, asleep = True):
        """ 
        Augments a behavpy objects sleep column to have duration and start of the sleep bouts, must contain a column with boolean values for sleep

        params:
        @sleep_column = string, default 'asleep'. Name of column in the data containing sleep data as a boolean
        @as_hist = bool, default False. If true the data will be augmented further into data appropriate for a histogram 
        Subsequent params only apply if as_hist is True
        @max_bins = integer, default 30. The number of bins for the data to be sorted into for the histogram, each bin is 1 minute
        @asleep = bool, default True. If True the histogram represents sleep bouts, if false bouts of awake

        returns a behavpy object with duration and time start of both awake and asleep

        if as_hist is True:
        returns a behavpy object with bins, count, and prob as columns
        """

        if sleep_column not in self.columns:
            warnings.warn(f'Column heading "{sleep_column}", is not in the data table')
            exit()

        tdf = self.reset_index().copy(deep = True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bout_analysis, 
                                                                                                var_name = sleep_column, 
                                                                                                as_hist = as_hist, 
                                                                                                bin_size = bin_size, 
                                                                                                max_bins = max_bins, 
                                                                                                time_immobile = time_immobile, 
                                                                                                asleep = asleep
            )), tdf.meta, check = True)

    @staticmethod
    def _wrapped_curate_dead_animals(data, time_var, moving_var, time_window, prop_immobile, resolution, time_dict = False): 

        time_window = (60 * 60 * time_window)
        d = data[[time_var, moving_var]].copy(deep = True)
        target_t = np.array(list(range(d[time_var].min().astype(int), d[time_var].max().astype(int), floor(time_window / resolution))))
        local_means = np.array([d[d[time_var].between(i, i + time_window)][moving_var].mean() for i in target_t])

        first_death_point = np.where(local_means <= prop_immobile, True, False)

        if any(first_death_point) is False:
            if time_dict is not False:
                time_dict[data['id'].iloc[0]] = [data[time_var].min(), data[time_var].max()]
            return data

        last_valid_point = target_t[first_death_point]
        if time_dict is not False:
            time_dict[data['id'].iloc[0]] = [data[time_var].min(), last_valid_point[0]]
        return data[data[time_var].between(data[time_var].min(), last_valid_point[0])]

    def curate_dead_animals(self, t_column = 't', mov_column = 'moving', time_window = 24, prop_immobile = 0.01, resolution = 24):
        
        """ 
        This function detects when individuals have died based on their first (very) long bout of immobility and removes that and subsequent data

        Params:
        @t_column = string, column heading for the data frames time stamp column (default is 't')
        @mov_column = string, logical variable in `data` used to define the moving (alive) state (default is `moving`)
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

        tdf = self.reset_index().copy(deep=True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_curate_dead_animals,
                                                                                                            time_var = t_column,
                                                                                                            moving_var = mov_column,
                                                                                                            time_window = time_window, 
                                                                                                            prop_immobile = prop_immobile,
                                                                                                            resolution = resolution
        )), tdf.meta, check = True)

    def curate_dead_animals_interactions(self, mov_df, t_column = 't', mov_column = 'moving', time_window = 24, prop_immobile = 0.01, resolution = 24):
        """ 
        A variation of curate dead animals to remove responses after an animal is presumed to have died
        
        Params:
        @mov_df = behavpy, a behavpy dataframe that has the full time series data for each specimen in the response dataframe
        @t_column = string, column heading for the data frames time stamp column (default is 't')
        @mov_column = string, logical variable in `data` used to define the moving (alive) state (default is `moving`)
        @time_window = int, window during which to define death 
        @prop_immobile = float, proportion of immobility that counts as "dead" during time_window 
        @resolution = int, how much scanning windows overlap. Expressed as a factor. 
        
        Returns two modified behavpy object, the 1st the interaction behavpy dataframe that had the method called on it and the
        2nd the movment df with all the data points. Both filted to remove specimens where presumed dead
        """

        def curate_filter(df, dict):
            return df[df[t_column].between(dict[df['id'].iloc[0]][0], dict[df['id'].iloc[0]][1])]

        if t_column not in self.columns.tolist() or t_column not in mov_df.columns.tolist():
            warnings.warn('Variable name entered, {}, for t_column is not a column heading!'.format(t_column))
            exit()
        
        if mov_column not in mov_df.columns.tolist():
            warnings.warn('Variable name entered, {}, for mov_column is not a column heading!'.format(mov_column))
            exit()

        tdf = self.reset_index().copy(deep=True)
        tdf2 = mov_df.reset_index().copy(deep=True)
        time_dict = {}
        curated_df = tdf2.groupby('id', group_keys = False).apply(partial(self._wrapped_curate_dead_animals,
                                                                                                            time_var = t_column,
                                                                                                            moving_var = mov_column,
                                                                                                            time_window = time_window, 
                                                                                                            prop_immobile = prop_immobile,
                                                                                                            resolution = resolution, 
                                                                                                            time_dict = time_dict
        ))
        curated_puff = tdf.groupby('id', group_keys = False, sort = False).apply(partial(curate_filter, dict = time_dict))
        return self.__class__(curated_puff, tdf.meta, check = True), self.__class__(curated_df, tdf2.meta, check = True), 
        

    @staticmethod
    def _wrapped_bin_data(data, column, bin_column, function, bin_secs):

        index_name = data['id'].iloc[0]

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

    @staticmethod
    def _wrapped_interpolate(data, var, step, t_col = 't'):
        """ Take the min and max time, create a time series at a given time step and interpolate missing values from the data """

        id = data['id'].iloc[0]
        sample_seq = np.arange(min(data[t_col]), np.nanmax(data[t_col]), step)
        if len(sample_seq) < 3:
            return None
        f  = interp1d(data[t_col].to_numpy(), data[var].to_numpy())
        return  pd.DataFrame(data = {'id' : id, t_col : sample_seq, var : f(sample_seq)})

    def interpolate(self, variable, step_size, t_column = 't'):
        """ A wrapped for wrapped_interpolate so work on multiple specimens
        params:
        @varibale = string, the column name of the variable of interest
        @step_size = int, the amount of time in seconds the new time series should progress by. I.e. 60 = [0, 60, 120, 180...]
        @t_column = string, string, column name for the time series data in your dataframe
        
        returns a behavpy object with a single data column with interpolated data per specimen"""

        data = self.copy(deep = True)
        data = data.bin_time(column = variable, t_column = t_column, bin_secs = step_size)
        data = data.rename(columns = {f'{t_column}_bin' : t_column, f'{variable}_mean' : variable})
        data = data.reset_index()
        return  self.__class__(data.groupby('id', group_keys = False).apply(partial(self._wrapped_interpolate, var = variable, step = step_size)), data.meta, check = True)

    def bin_time(self, column, bin_secs, t_column = 't', function = 'mean'):
        """
        Bin the time series data into entered groups, pivot by the time series column and apply a function to the selected columns.
        
        Params:
        @column = string, column in the data that you want to the function to be applied to post pivot
        @bin_secs = float, the amount of time you want in each bin in seconds, e.g. 60 would be bins for every minutes
        @t_column = string, column name for the time series data that you want to group and pivot by
        @function = string or user defined function. The applied function to the grouped data, can be standard 'mean', 'max'.... ect, can also be a user defined function
        
        returns a behavpy object with a single data column
        """

        if column not in self.columns:
            warnings.warn('Column heading "{}", is not in the data table'.format(column))
            exit()

        tdf = self.reset_index().copy(deep=True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bin_data,
                                                                                                column = column, 
                                                                                                bin_column = t_column,
                                                                                                function = function, 
                                                                                                bin_secs = bin_secs
        )), tdf.meta, check = True)

    def summary(self, detailed = False, t_column = 't'):
        """ 
        Prints a table with summary statistics of metadata and data counts.
            
        Params:
        @detailed = bool , if detailed is True count and range of data points will be broken down per 'id' 
            
        returns None
        """

        def print_table(table):
            longest_cols = [
                (np.nanmax([len(str(row[i])) for row in table]) + 3)
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
                return (str(min(data)) + '  ->  ' + str(np.nanmax(data)))


            group = self.groupby('id').agg(
                data_points = pd.NamedAgg(column = t_column, aggfunc = 'count'),
                time_range = pd.NamedAgg(column = t_column, aggfunc = time_range)
            )

            print(group)

    def add_day_phase(self, time_column = 't', day_length = 24, lights_off = 12, inplace = True):
        """ 
        Adds a column called 'phase' with either light or dark as catergories according to its time compared to the reference hour
        Adds a column with the day the row in, starting with 1 as day zero and increasing sequentially.

        Params:
        @circadian_night = int, the ZT hour when the conditions shift to dark
            
        returns a new df is inplace is False, else nothing
        """
        day_in_secs = 60*60*day_length
        night_in_secs = lights_off * 60 * 60

        if inplace == True:
            self['day'] = self[time_column].map(lambda t: floor(t / day_in_secs))
            self['phase'] = np.where(((self[time_column] % day_in_secs) > night_in_secs), 'dark', 'light')
            self['phase'] = self['phase'].astype('category')

        elif inplace == False:
            new_df = self.copy(deep = True)
            new_df['day'] = new_df[time_column].map(lambda t: floor(t / day_in_secs)) 
            new_df['phase'] = np.where(((new_df[time_column] % day_in_secs) > night_in_secs), 'dark', 'light')
            new_df['phase'] = new_df['phase'].astype('category')

            return new_df

    @staticmethod
    def _wrapped_motion_detector(data, time_window_length, velocity_correction_coef, masking_duration, optional_columns):
        
        index_name = data['id'].iloc[0]
        
        df = max_velocity_detector(data,                                   
                                time_window_length = time_window_length, 
                                velocity_correction_coef = velocity_correction_coef, 
                                masking_duration = masking_duration, 
                                optional_columns = optional_columns)

        old_index = pd.Index([index_name] * len(df.index), name = 'id')
        df.set_index(old_index, inplace =True)  

        return df    

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

        tdf = self.reset_index().copy(deep=True)
        m = tdf.meta
        return  self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_motion_detector,
                                                                                                        time_window_length = time_window_length,
                                                                                                        velocity_correction_coef = velocity_correction_coef,
                                                                                                        masking_duration = masking_duration,
                                                                                                        optional_columns = optional_columns
        )), m, check = True)

    @staticmethod
    def _wrapped_sleep_contiguous(d_small, mov_column, t_column, time_window_length, min_time_immobile):

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

        time_map = pd.Series(range(d_small[t_column].iloc[0], 
                            d_small[t_column].iloc[-1] + time_window_length, 
                            time_window_length
                            ), name = t_column)

        missing_values = time_map[~time_map.isin(d_small[t_column].tolist())]
        d_small = d_small.merge(time_map, how = 'right', on = t_column, copy = False).sort_values(by=[t_column])
        d_small['is_interpolated'] = np.where(d_small[t_column].isin(missing_values), True, False)
        d_small[mov_column] = np.where(d_small['is_interpolated'] == True, False, d_small[mov_column])

        d_small['asleep'] = sleep_contiguous(d_small[mov_column], 1/time_window_length, min_valid_time = min_time_immobile)
        d_small['id'] = [d_small['id'].iloc[0]] * len(d_small)
        d_small.set_index('id', inplace = True)

        return d_small  

    def sleep_contiguous(self, mov_column = 'moving', t_column = 't', time_window_length = 10, min_time_immobile = 300):
        """
        Method version of the sleep annotation function.
        This function first uses a motion classifier to decide whether an animal is moving during a given time window.
        Then, it defines sleep as contiguous immobility for a minimum duration.
        See function for paramater details

        returns a behavpy object with added columns like 'moving' and 'asleep'
        """
        if mov_column not in self.columns.tolist():
            warnings.warn(f'The movement column {mov_column} is not in the dataset')
            exit()
        if t_column not in self.columns.tolist():
            warnings.warn(f'The time column {t_column} is not in the dataset')
            exit()  

        tdf = self.reset_index().copy(deep = True)
        m = tdf.meta
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_sleep_contiguous,
                                                                                                        mov_column = mov_column,
                                                                                                        t_column = t_column,
                                                                                                        time_window_length = time_window_length,
                                                                                                        min_time_immobile = min_time_immobile
        )), m, check = True)

    def wrap_time(self, wrap_time = 24, time_column = 't', inplace = False):
        """
        Replaces linear values of time in column 't' with a value which is a decimal of the wrap_time input

        Params:
        @wrap_time = int, time in hours you want to wrap the time series by, default is 24 hours
        @time_column  = string, column title for the time series column, default is 't'

        returns nothig, all actions are inplace generating a modified version of the given behavpy table
        """
        hours_in_seconds = wrap_time * 60 * 60
        if inplace == False:
            new = self.copy(deep = True)
            new[time_column] = new[time_column] % hours_in_seconds
            return new
        else:
            self[time_column] = self[time_column] % hours_in_seconds

    def baseline(self, column, t_column = 't', day_length = 24, inplace = False):
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

        day_dict = self.meta[column].to_dict()

        if inplace == True:
            self['tmp_col'] = self.index.to_series().map(day_dict)
            self[t_column] = self[t_column] + (self['tmp_col'] * (60*60*24))
            self.drop(columns = ['tmp_col'], inplace = True)
        else:
            new = self.copy(deep = True)
            new['tmp_col'] = new.index.to_series().map(day_dict)
            new[t_column] = new[t_column] + (new['tmp_col'] * (60*60*24))
            return new.drop(columns = ['tmp_col'])

    def remove(self, column, *args):
        """ 
        A variation of xmv to remove all rows from a data table whose ID matches those specified from the metadata

        Params:
        @column = string. column heading from the metadata of the behavpy object
        @*args = string, arguments corresponding to groups from the column given

        returns a behavpy object with filtered data and metadata
        """

        if type(args[0]) == list:
            args = args[0]
    
        if column == 'id':
            for m in args:
                if m not in self.meta.index.tolist():
                    warnings.warn('Metavariable "{}" is not in the id column'.format(m))
                    exit()
            index_list = [x for x in self.meta.index.tolist() if x not in args]
            # find interection of meta and data id incase metadata contains more id's than in data
            data_id = list(set(self.index.values))
            new_index_list = np.intersect1d(index_list, data_id)
            self = self[self.index.isin(index_list)]
            self.meta = self.meta[self.meta.index.isin(new_index_list)]
            return self
        else:
            if column not in self.meta.columns:
                warnings.warn('Column heading "{}" is not in the metadata table'.format(column))
                exit()
            column_varaibles = list(set(self.meta[column].tolist()))

            index_list = []
            for m in args:
                if m not in self.meta[column].tolist():
                    warnings.warn('Metavariable "{}" is not in the column'.format(m))
                    exit()
                else:
                    column_varaibles.remove(m)
            for v in column_varaibles:
                small_list = self.meta[self.meta[column] == v].index.values
                index_list.extend(small_list)

            # find interection of meta and data id incase metadata contains more id's than in data
            data_id = list(set(self.index.values))
            new_index_list = np.intersect1d(index_list, data_id)
            self = self[self.index.isin(index_list)]
            self.meta = self.meta[self.meta.index.isin(new_index_list)]
            return self

    def curate(self, points):
        """
        A method to remove specimens without enough data points. The user must work out the number of points that's equivalent to their wanted time coverage.

        Params:
        @points, int, the number of minimum data points a specimen must have to not be removed.

        returns a behavpy object with specimens of low data points removed from the metadata and data
        """

        def wrapped_curate(data, limit = points):
            if len(data) < limit:
                id_list.append(list(set(data['id']))[0])
                return data
            else:
                return data

        id_list = []
        df = self.reset_index()
        df.groupby('id', group_keys = False).apply(wrapped_curate)
        
        return self.remove('id', id_list)


    @staticmethod
    def _find_runs(mov, time, id):
        _, _, l = rle(mov)
        # count_list = np.concatenate([np.append(np.arange(1, cnt + 1, 1)[: -1], np.nan) for cnt in l], dtype = float)
        count_list = np.concatenate([np.arange(1, cnt + 1, 1) for cnt in l], dtype = float)
        previous_count_list = count_list[:-1]
        previous_count_list = np.insert(previous_count_list, 0, np.nan)
        previous_mov = mov[:-1].astype(float)
        previous_mov = np.insert(previous_mov, 0, np.nan)
        return {'id': id, 't' : time, 'moving' : mov, 'previous_moving' : previous_mov, 'activity_count' : count_list, 'previous_activity_count' : previous_count_list}

    def feeding(self, food_position, dist_from_food = 0.05, micro_mov = 'micro', x_position = 'x', time_col = 't'):
        """ A method that approximates the time spent feeding for flies in the ethoscope given their micromovements near to the food
        Params:
        @food_postion = string, must be either "outside" or "inside". This signifies the postion of the food in relation to the center of the arena
        @dist_from_food = float, the distance measured between 0-1, as the x coordinate in the ethoscope, that you classify as being near the food, default 0.05
        @micro_mov = string, the name of the column that contains the data for whether micromovements occurs, True/False
        @x_position = string, the name of the column that contains the x postion
        @time_col = string, the name of the column that contains the time. This is so the first hour can be ignored for when the ethoscope is 
        calcuting it's tracking. If set to False it will ignore this filter

        returns an augmented behavpy object with an addtional column 'feeding' with boolean variables
        """
        if food_position != 'outside' and food_position != 'inside':
            raise ValueError("Argument for food_position must be 'outside' or 'inside'")
            
        ds = self.copy(deep = True)
        
        # normalise x values for ROI on the right 11-20
        ds_r = ds.xmv('region_id', list(range(11,21)))
        ds_l = ds.xmv('region_id', list(range(1,11)))
        ds_r[x_position] = 1 - ds_r[x_position]
        ds = ds_l.concat(ds_r)
        
        def find_feed(d):
            
            # if there's less than 2 data points just run the check
            if len(d) < 2:
                if food_position == 'outside':
                    d['feeding'] = np.where((d[x_position] < d[x_position].min()+dist_from_food) & (d[micro_mov] == True), True, False)
                elif food_position == 'inside':
                    d['feeding'] = np.where((d[x_position] > d[x_position].max()-dist_from_food) & (d[micro_mov] == True), True, False)
                return d
            
            # ignore the first hour in case tracking is wonky and get x min and max
            t_diff = d[time_col].iloc[1] - d[time_col].iloc[0]
            if t_col is not False:
                t_ignore = int(3600 / t_diff)
                tdf = d.iloc[t_ignore:]
            else:
                tdf = d
            x_min = tdf[x_position].min()
            x_max = tdf[x_position].max()
            
            # if the fly is near to the food and mirco moving, then they are assumed to be feeding
            if food_position == 'outside':
                d['feeding'] = np.where((d[x_position] < x_min+dist_from_food) & (d[micro_mov] == True), True, False)
            elif food_position == 'inside':
                d['feeding'] = np.where((d[x_position] > x_max-dist_from_foodo) & (d[micro_mov] == True), True, False)
            return d
            
        ds.reset_index(inplace = True)   
        ds_meta = ds.meta
        return self.__class__(ds.groupby('id', group_keys = False).apply(find_feed).set_index('id'), ds_meta)


    def remove_sleep_deprived(self, start_time, end_time, remove = False, sleep_column = 'asleep', t_column = 't'):
        """ Removes specimens that during a period of sleep deprivation are asleep a certain percentage of the period
        Params:
        @start_time = int, the time in seconds that the period of sleep deprivation begins
        @end_time = int, the time in seconds that the period of sleep deprivation ends
        @remove = int or bool, an int >= 0 or < 1 that is the percentage of sleep allowed during the period without being removed.
        The default is False, which will return a new groupby pandas df with the asleep percentages per specimen
        @sleep_column = string, the name of the column that contains the data of whether the specimen is asleep or not
        @t_column = string, the name of the column that contains the time

        returns if remove is not False an augmented behavpy df with specimens removed that are not sleep deprived enough. See remove for the alternative.
        """

        if start_time > end_time:
            raise KeyError('The start time can not be greater than the end time')

        if start_time < 0:
            raise KeyError('The start time can not be less than zero')

        if remove is not False:
            if remove >= 0 and remove < 1:
                pass
            else:
                raise ValueError('Remove must be a float that is greater/equal to zero and less than 1')

        # Filter by the period and remove other columns to reduce memory
        fdf = self.t_filter(start_time = start_time, end_time = end_time, t_column = t_column)[[t_column, sleep_column]]

        gb = fdf.groupby(fdf.index).agg(**{
            'time asleep' : (sleep_column, 'sum'),
            'min t' : (t_column, 'min'),
            'max t' : (t_column, 'max'),

        })
        tdiff = fdf[t_column].diff().mode()[0]
        gb['time asleep(s)'] = gb['time asleep'] * tdiff
        gb['time(s)'] = gb['max t'] - gb['min t']
        gb['Percent Asleep'] = gb['time asleep(s)'] / gb['time(s)']
        gb.drop(columns = ['time asleep', 'max t', 'min t'], inplace = True)

        if remove == False:
            return gb
        else:
            remove_ids = gb[gb['Percent Asleep'] > remove].index.tolist()
            return df.remove('id', remove_ids)


    def heatmap_dataset(self, variable = 'moving', t_column = 't'):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals
        
        Params:
        @variable = string, name for the column containing the variable of interest, the default is moving
        
        returns gbm, time_list, id
        """

        heatmap_df = self.copy(deep = True)
        # change movement values from boolean to intergers and bin to 30 mins finding the mean
        if variable == 'moving':
            heatmap_df[variable] = np.where(heatmap_df[variable] == True, 1, 0)

        heatmap_df = heatmap_df.bin_time(column = variable, bin_secs = 1800, t_column = t_column)
        heatmap_df['t_bin'] = heatmap_df['t_bin'] / (60*60)
        # create an array starting with the earliest half hour bin and the last with 0.5 intervals
        start = heatmap_df['t_bin'].min().astype(int)
        end = heatmap_df['t_bin'].max().astype(int)
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

        heatmap_df = heatmap_df.groupby('id', group_keys = False).apply(align_data)

        gbm = heatmap_df.groupby(heatmap_df.index)[f'{variable}_mean'].apply(list)
        id = heatmap_df.groupby(heatmap_df.index)['t_bin'].mean().index.tolist()

        return gbm, time_list, id 
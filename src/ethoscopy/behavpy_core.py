import pandas as pd
import numpy as np 
import pickle

from math import floor, ceil, sqrt
from scipy.stats import zscore
from functools import partial, update_wrapper

from tabulate import tabulate
from hmmlearn import hmm

from scipy.signal import find_peaks
from astropy.timeseries import LombScargle

from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.analyse import max_velocity_detector
from ethoscopy.misc.rle import rle
from ethoscopy.misc.bootstrap_CI import bootstrap
from ethoscopy.misc.periodogram_functions import chi_squared, lomb_scargle, fourier, welch, wavelet

class behavpy_core(pd.DataFrame):
    """
    The behavpy class is a store of information for data from the ethoscope system with corresponding methods to augment and manipulate
    the data as necessary for standard analysis.
    Behavpy is subclassed from the pandas dataframe object and can be manipualted using all their tools as well as the custom methods within
    Behavpy sets a metadata dataframe as an attribute which is called upon frequently in the methods, can be accessed through behavpy.meta
    Both metadata and data should share unique ids in their 'id' column that are essential for manipulaiton
    print(df) will only print the data df, to see both use the .display() method

    Initialisation Parameters:
        data (pandas DataFrame): The experimental recorded data usually loaded by the load_ethoscope function. Must contain an id column with unique IDs per speciemn
        and a time series column called 't' with time per recording in seconds 
        meta (pandas Dataframe): The metadata i.e. conditions, genotypes ect of the experiment. There should be a unique row per ID in the data. 
        Usually generated from a csv file and link_meta_index function.
        colour (str, optional): The name of a palette you want to use for plotly. See https://plotly.com/python/discrete-color/ for the types. Default is 'Safe'
        long_colour (str, optional): The name of a palette for use when you have more than 11 groups to plot, i.e. Dark24, Light24, Alphabet. Default is 'Dark24'
        check (bool, optional): when True this will check the ids in the data are in the metadata. If not an error will be raised. It also removes some columns that are no longer
        needed that are generated in link_meta_index.

    returns a behavpy object with methods to manipulate, analyse, and plot time series behavioural data

    """
    # set meta as permenant attribute
    _metadata = ['meta']
    _canvas = None
    _hmm_colours = None
    _hmm_labels = None


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

    def __init__(self, data, meta, palette = None, long_palette = None, check = False, index= None, columns=None, dtype=None, copy=True):
        super(behavpy_core, self).__init__(data=data,
                                        index=index,
                                        columns=columns,
                                        dtype=dtype,
                                        copy=copy)

        self.meta = meta   
        if check is True:
            self._check_conform(self)
        self.attrs = {'sh_pal' : palette, 'lg_pal' : long_palette}

    @staticmethod
    def _check_conform(dataframe):
        """ 
        Checks the data augument is a pandas dataframe
        If metadata is provided and skip is False it will check as above and check the ID's in
        metadata match those in the data
        """
        
        if isinstance(dataframe.meta, pd.DataFrame) is not True:
            raise TypeError('Metadata input is not a pandas dataframe')

        drop_col_names = ['path', 'file_name', 'file_size', 'machine_id']
        dataframe.meta = dataframe.meta.drop(columns=[col for col in dataframe.meta if col in drop_col_names])

        if dataframe.index.name != 'id':
            try:
                dataframe.set_index('id', inplace = True)
            except:
                raise KeyError("There is no 'id' as a column or index in the data'")

        if dataframe.meta.index.name != 'id':
            try:
                dataframe.meta.set_index('id', inplace = True)
            except:
                raise KeyError("There is no 'id' as a column or index in the metadata'")

        # checks if all id's of data are in the metadata dataframe
        check_data = all(elem in set(dataframe.meta.index.tolist()) for elem in set(dataframe.index.tolist()))
        if check_data is not True:
            raise RuntimeError("There are ID's in the data that are not in the metadata, please check")

    def _check_lists(self, f_col, f_arg, f_lab):
        """
        Check if the facet arguments match the labels or populate from the column if not.
        """

        if f_col is not None:
            if f_col not in self.meta.columns:
                raise KeyError(f'Column "{f_col}" is not a metadata column')
            if f_arg is None:
                f_arg = list(set(self.meta[f_col].tolist()))
                string_args = []
                for i in f_arg:
                    string_args.append(str(i))
                if f_lab is None:
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    print("The facet labels don't match the length of the variables in the column. Using column variables instead")
                    f_lab = string_args
            else:
                string_args = []
                for i in f_arg:
                    if i not in self.meta[f_col].tolist():
                        print(self.meta[f_col].tolist())
                        raise KeyError(f'Argument "{i}" is not in the meta column {f_col}')
                    string_args.append(str(i))
                if f_lab is None:
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    raise ValueError("The facet labels don't match the entered facet arguments in length. Using column variables instead")
                    f_lab = string_args
        else:
            f_arg = [None]
            if f_lab is None:
                f_lab = ['']

        return f_arg, f_lab

    @staticmethod
    def _check_boolean(lst):
        """
        Checks to see if a column of data (as a list) max and min is 1 and 0, so as to make a appropriately scaled y-axis
        """
        if np.nanmax(lst) == 1 and np.nanmin(lst) == 0:
            y_range = [-0.025, 1.01]
            dtick = 0.2
        else:
            y_range = False
            dtick = False
        return y_range, dtick

    @staticmethod
    def _zscore_bootstrap(array, z_score = True, second_array = None, min_max = False):
        """ Calculate the z score of a given array, remove any values +- 3 SD and then perform bootstrapping on the remaining
        returns the mean and then several lists with the confidence intervals and z-scored values
        """
        try:
            if len(array) == 1 or all(array == array[0]):
                mean = median = q3 = q1 = array[0]
                zlist = array
            else:
                if z_score is True:
                    zlist = array[np.abs(zscore(array)) < 3]
                    if second_array is not None:
                        second_array = second_array[np.abs(zscore(array)) < 3] 
                else:
                    zlist = array
                mean = np.mean(zlist)
                median = np.median(zlist)
                boot_array = bootstrap(zlist)
                q3 = boot_array[1]
                q1 = boot_array[0]

        except ZeroDivisionError:
            mean = median = q3 = q1 = 0
            zlist = array

        if min_max == True:
            q3 = np.max(array)
            q1 = np.min(array)
        
        if second_array is not None:
            return mean, median, q3, q1, zlist, second_array
        else:
            return mean, median, q3, q1, zlist

    def display(self):
        """
        Alternative to print(), displays both the metadata and data with corresponding headers

        returns a formatted pandas print
        """
        print('\n ==== METADATA ====\n\n{}\n ====== DATA ======\n\n{}'.format(self.meta, self))

    def xmv(self, column, *args):
        """
        Filter your data and metadata by a column and groups from the metadata.

        Args:
            column (str): Column heading from the metadata of the behavpy object
            *args (pandas cell input): Arguments corresponding to groups from the column given, can be given as a list or several args, but not a mix.

        returns a behavpy object with filtered data and metadata
        """

        if type(args[0]) == list or type(args[0]) == np.array:
            args = args[0]

        if column == 'id':
            index_list = []
            for m in args:
                if m not in self.meta.index.tolist():
                    raise KeyError(f'Metavariable "{m}" is not in the id column')
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
            raise KeyError(f'Column heading "{column}" is not in the metadata table')
        
        index_list = []
        for m in args:
            if m not in self.meta[column].tolist():
                raise KeyError('Metavariable "{}" is not in the column'.format(m))
            else:
                small_list = self.meta[self.meta[column] == m].index.values
                index_list.extend(small_list)

        # find interection of meta and data id incase metadata contains more id's than in data
        data_id = list(set(self.index.values))
        new_index_list = np.intersect1d(index_list, data_id)
        self = self[self.index.isin(index_list)]
        self.meta = self.meta[self.meta.index.isin(new_index_list)]
        return self

    def remove(self, column, *args):
        """ 
        A variation of xmv to remove all rows from a data table whose ID matches those specified from the metadata

            Args:
                column (str): The column heading from the metadata of the behavpy object that you wish to filter by
                *args (string/int/float): Arguments corresponding to groups from the column given

        returns:
            A behavpy object with filtered data and metadata
        """

        if type(args[0]) == list:
            args = args[0]
    
        if column == 'id':
            for m in args:
                if m not in self.meta.index.tolist():
                    raise KeyError('Metavariable "{}" is not in the id column'.format(m))

            index_list = [x for x in self.meta.index.tolist() if x not in args]
            # find interection of meta and data id incase metadata contains more id's than in data
            data_id = list(set(self.index.values))
            new_index_list = np.intersect1d(index_list, data_id)
            self = self[self.index.isin(index_list)]
            self.meta = self.meta[self.meta.index.isin(new_index_list)]
            return self
        else:
            if column not in self.meta.columns:
                raise KeyError('Column heading "{}" is not in the metadata table'.format(column))
            column_varaibles = list(set(self.meta[column].tolist()))

            index_list = []
            for m in args:
                if m not in self.meta[column].tolist():
                    raise KeyError('Metavariable "{}" is not in the column'.format(m))
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


    def summary(self, detailed = False, t_column = 't'):
        """ 
        Prints a table with summary statistics of metadata and data counts.
            
        Args:
            detailed (bool, optional): If detailed is True count and range of data points will be broken down per 'id'. Default is False.
            
        Returns:
            no return
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

    def add_day_phase(self, t_column = 't', day_length = 24, lights_off = 12, inplace = True):
        """ 
        Adds a column called 'phase' with either light or dark as catergories according to its time compared to the reference hour
        Adds a column with the day the row is in, starting with 1 as the first day and increasing sequentially.

        Args:
            t_column (str): The name of column containing the timing data (in seconds). Default is 't'.
            day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
            inplace (bool, optional): 
            
        returns a new df is inplace is False, else nothing
        """
        day_in_secs = 60*60*day_length
        night_in_secs = lights_off * 60 * 60

        if inplace == True:
            self['day'] = self[t_column].map(lambda t: floor(t / day_in_secs))
            self['phase'] = np.where(((self[t_column] % day_in_secs) > night_in_secs), 'dark', 'light')
            self['phase'] = self['phase'].astype('category')

        elif inplace == False:
            new_df = self.copy(deep = True)
            new_df['day'] = new_df[t_column].map(lambda t: floor(t / day_in_secs)) 
            new_df['phase'] = np.where(((new_df[t_column] % day_in_secs) > night_in_secs), 'dark', 'light')
            new_df['phase'] = new_df['phase'].astype('category')

            return new_df

    def t_filter(self, start_time = 0, end_time = np.inf, t_column = 't'):
        """
        Filters the data to only be inbetween the provided start and end points
        argument is given in hours and converted to seconds.

        Args:
            start_time (int, optional): In hours the start time you wish to filter by.
            end_time (int, optional): In hours the end time you wish to filter by/
            t_column (str, optional): The column containing the timestamps in seconds. Default is 't'

        returns a filtered behapvy object
        """
        if start_time > end_time:
            raise ValueError('Error: end_time is larger than start_time')

        return self[(self[t_column] >= (start_time * 60 * 60)) & (self[t_column] < (end_time * 60 * 60))]

    def rejoin(self, new_column, check = True):
        """
        Joins a new column to the metadata. 

        Args:
            new_column (pd DataFrame). The column to be added, must contain an index called 'id' to match original metadata
            check (Bool, optional): Whether or not to check if the ids in the data match the new column, default is True

        No return, the behavpy object is augmented inplace
        """

        if check is True:
            check_data = all(elem in new_column.index.tolist() for elem in set(self.index.tolist()))
            if check_data is not True:
                raise KeyError("There are ID's in the data that are not in the metadata, please check. You can skip this process by changing the parameter skip to False")

        m = self.meta.join(new_column, on = 'id')
        self.meta = m

    def concat(self, *args):
        """
        Wrapper for pd.concat that also concats metadata of multiple behavpy objects

        Args:
            args (behvapy): Behavpy tables to be concatenated to the original behavpy table, each behavpy object should be entered as its own argument and not a list.

        returns:
            A new instance of a combined behavpy object
        """

        meta_list = [self.meta]
        data_list = [self]

        for df in args:

            if isinstance(df, self.__class__) is not True or isinstance(df, behavpy) is not True:
                raise TypeError('Object(s) to concat is(are) not a Behavpy object')

            meta_list.append(df.meta)
            data_list.append(df)

        meta = pd.concat(meta_list)
        new = pd.concat(data_list)

        new.meta = meta

        return new

    def stitch_consecutive(self, machine_name_col = 'machine_name', region_id_col = 'region_id', date_col = 'date'):
        """ 
        A method to stitch the data together for ROIs of the same machine in the dataframe. Use this for when you needed to stop and restart an experiment with the same specimens in it.
        The method selectes all the unique machine names and ROI numbers and merges those that match, taking the start date of the experiment to calculate how much to modify the time.
        THe method will also only retain the metadata for the earliest experiment and change the id in the date to match the first running, 
        this is so later methods aren't confused. So make sure all information for all is in the the first set.
        Make sure the dataframe only contains experiments of the same machines you want to stitch together!

        Args:
            machine_name_col (str): The name of the column which contains the name of each machine, .e.g 'ETHOSCOPE_030'. Default is 'machine_name'.
            region_id_col (str): The name of the column which contains the name of the ROI per machine. Default is 'region_id'.
            date_col (str): The name of the column which contains the name of the date column for the experiment. Default is 'date'.

        returns:
            A pandas dataframe with the metadata and data transformed to be a single experiment per machine/roi
        """
        mach_list = set(self.meta[machine_name_col].tolist())
        roi_list = set(self.meta[region_id_col].tolist())

        def augment_time(d):
            d.meta['day_diff'] = (pd.to_datetime(d.meta['date']) - pd.to_datetime(d.meta['date'].min())).dt.days
            indx_map = d.meta[['day_diff']].to_dict()['day_diff']
            d['day'] = indx_map
            d['t'] = (d['day'] * 86400) + d['t']
            d['id'] = [list(indx_map.keys())[list(indx_map.values()).index(0)]] * len(d)
            d = d.set_index('id')
            d.meta = d.meta[d.meta['day_diff'] == 0]
            d.meta.drop(['day_diff'], axis = 1, inplace = True)
            d.drop(['day'], axis = 1, inplace = True)
            return d

        df_list = []

        for i in mach_list:
            mdf = self.xmv(machine_name_col, i)
            for q in roi_list:
                try:
                    rdf = mdf.xmv(region_id_col, q)
                    ndf = augment_time(rdf)
                    df_list.append(ndf)
                except KeyError:
                    continue

        return df_list[0].concat(*df_list[1:])

    def analyse_column(self, column, function):
        """ 
        Wrapper for the groupby pandas method to split by groups in a column and apply a function to said groups

        Args:
            column (str): The name of the column in the data to pivot by
            function (str or user defined function): The applied function to the grouped data, can be standard 'mean', 'max'.... ect, can also be a user defined function

        returns:
            A pandas dataframe with the transformed grouped data with an index
        """

        if column not in self.columns:
            raise KeyError(f'Column heading, "{column}", is not in the data table')
            
        try:
            parse_name = f'{column}_{function.__name__}' # create new column name if the function is a users own function
        except AttributeError:
            parse_name = f'{column}_{function}' # create new column name with string defined function

        pivot = self.groupby(self.index).agg(**{
            parse_name : (column, function)    
        })

        return pivot

    def wrap_time(self, wrap_time = 24, time_column = 't'):
        """
        Replaces linear values of time in column 't' with a value which is a decimal of the wrap_time input

            Args:
                wrap_time (int, optional): Time in hours you want to wrap the time series by, default is 24 hours
                time_column (string, optional): The column name for the time series column, default is 't'

        returns none, all actions are inplace generating a modified version of the given behavpy table
        """
        hours_in_seconds = wrap_time * 60 * 60

        new = self.copy(deep = True)
        new[time_column] = new[time_column] % hours_in_seconds
        return new

    def baseline(self, column, t_column = 't', day_length = 24):
        """
        A function to add days to the time series data per animal to align interaction times per user discretion

            Args:
                column (string): The name of column containing the number of days to add, must be an integer, 0 = no days added
                t_column = (string, optional): The name of column containing the time series data to be modified, default is 't'
                day_length = (int, optional): The time in hours that a days are in the experiment, default is 24
        
        returns: 
            A behavpy table with modifed time series columns
        """

        if column not in self.meta.columns:
            raise KeyError('Baseline days column: "{}", is not in the metadata table'.format(column))

        day_dict = self.meta[column].to_dict()

        new = self.copy(deep = True)
        new['tmp_col'] = new.index.to_series().map(day_dict)
        new[t_column] = new[t_column] + (new['tmp_col'] * (60*60*24))
        return new.drop(columns = ['tmp_col'])

    def curate(self, points):
        """
        A method to remove specimens without enough data points. The user must work out the number of points that's equivalent to their wanted time coverage.

            Args:
                points (int) The number of minimum data points a specimen must have to not be removed.

        returns: 
            A behavpy object with specimens of low data points removed from the metadata and data
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


    def remove_sleep_deprived(self, start_time, end_time, remove = False, sleep_column = 'asleep', t_column = 't'):
        """ Removes specimens that during a period of sleep deprivation are asleep a certain percentage of the period

        Args:
            start_time (int): The time in hours that the period of sleep deprivation begins
            end_time (int): The time in hours that the period of sleep deprivation ends
            remove (int or bool, optional): An int >= 0 or < 1 that is the percentage of sleep allowed during the period without being removed.
                The default is False, which will return a new groupby pandas df with the asleep percentages per specimen
            sleep_column (string, optional) The name of the column that contains the data of whether the specimen is asleep or not
            t_column (string, optional): The name of the column that contains the time

        returns: 
            If remove is not False an augmented behavpy df with specimens removed that are not sleep deprived enough. 
            See remove for the alternative.
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
            return self.__class__(gb, self.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check=True)
        else:
            remove_ids = gb[gb['Percent Asleep'] > remove].index.tolist()
            return self.remove('id', remove_ids)

    @staticmethod
    def _time_alive(df, facet_col, repeat = False, t_column = 't'):
        """ Method to call to the function that finds the amount of time a specimen has survived.
        If repeat is True then the function will look for a column in the metadata called 'repeat' and use it to sub filter the dataframe. 
        """

        def _wrapped_time_alive(df, name):
            """ The wrapped method called by _time_alive. This function finds the max and min time per specimen and creates an aranged list
            per hour. These are then stacked and divided by the max to find the percentage alive at a given hour.
            The returned data frame is formatted for use with dataframe plotters such as Seaborne and Plotly express.
            """
            gb = df.groupby(df.index).agg(**{
                'tmin' : ('t', 'min'),
                'tmax' : ('t', 'max')
                })
            gb['time_alive'] = round(((gb['tmax'] - gb['tmin']) / 86400) * 24)
            gb.drop(columns = ['tmax', 'tmin'], inplace = True)   
            
            m = int(gb['time_alive'].max())
            cols = []
            for k, v in gb.to_dict()['time_alive'].items():
                y = np.repeat(1, v)
                cols.append(np.pad(y, (0, m - len(y))))

            col = np.vstack(cols).sum(axis = 0)
            col = (col / np.max(col)) * 100
            return pd.DataFrame(data = {'hour' : range(0, len(col)), 'survived' : col, 'label' : [name] * len(col)})

        # set name as the facet_arg 
        if facet_col is None:
            name = ''
        else:
            name = df[facet_col].tolist()[0]

        if repeat is False:
            return _wrapped_time_alive(df, name)
        else:
            tmp = pd.DataFrame()
            for rep in set(df.meta[repeat].tolist()):
                tdf = df.xmv(repeat, rep)
                tmp = pd.concat([tmp, _wrapped_time_alive(tdf, name)])
            return tmp

    # GROUPBY SECTION

    @staticmethod
    def _wrapped_bout_analysis(data, var_name, as_hist, bin_size, max_bins, time_immobile, asleep, t_column = 't'):
        """ Finds runs of bouts of immobility or moving and sorts into a historgram per unqiue specimen if as_hist is True. """

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

    def sleep_bout_analysis(self, sleep_column = 'asleep', as_hist = False, bin_size = 1, max_bins = 60, time_immobile = 5, asleep = True, t_column = 't'):
        """ 
        Augments a behavpy objects sleep column to have duration and start of the sleep bouts, must contain a column with boolean values for sleep.
        If as_hist is True then the a dataframe containing the data needed to make a histogram of the data is created. Use plot_sleep_bouts to automatically
        plot this histogram.

        Args:
            sleep_column (str, optional): Name of column in the data containing sleep data as boolean values. Default is 'asleep'
            as_hist (bool, optional). If true the data will be augmented further into data appropriate for a histogram 
            Subsequent params only apply if as_hist is True
            bin_size (int, optional): The size of the histogram bins that will be plotted. Default is 1
            max_bins (int, optional): The maximum number of bins you want the data sorted into. Default is 60
            @asleep = bool, default True. If True the histogram represents sleep bouts, if false bouts of awake

        returns:
            returns a behavpy object with duration and time start of both awake and asleep

            if as_hist is True:
            returns a behavpy object with bins, count, and prob as columns
        """

        if sleep_column not in self.columns:
            raise KeyError(f'Column heading "{sleep_column}", is not in the data table')

        tdf = self.reset_index().copy(deep = True)
        return self.__class__(self)(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bout_analysis, 
                                                                                                var_name = sleep_column, 
                                                                                                as_hist = as_hist, 
                                                                                                bin_size = bin_size, 
                                                                                                max_bins = max_bins, 
                                                                                                time_immobile = time_immobile, 
                                                                                                asleep = asleep,
                                                                                                t_column = t_column
            )), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)


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

        Args:
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            mov_column (str, optional): Name of column in the data containing movemment data as boolean values. Default is 'moving'
            time_window (int, optioanl): The size of the moving window (in hours) during which to define death. Default is 24.
            prop_immobile (float, optional): Proportion of immobility that counts as "dead" during time_window. Default is 0.01 (1%)
            resolution (int, optional): How much scanning windows overlap. Expressed as a factor. Default is 24.

        Returns:
            Returns a behavpy object with filtered rows
        """

        if t_column not in self.columns.tolist():
            raise KeyError('Variable name entered, {}, for t_column is not a column heading!'.format(t_column))
        
        if mov_column not in self.columns.tolist():
            raise KeyError('Variable name entered, {}, for mov_column is not a column heading!'.format(mov_column))

        tdf = self.reset_index().copy(deep=True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_curate_dead_animals,
                                                                                                            time_var = t_column,
                                                                                                            moving_var = mov_column,
                                                                                                            time_window = time_window, 
                                                                                                            prop_immobile = prop_immobile,
                                                                                                            resolution = resolution
        )), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    def curate_dead_animals_interactions(self, mov_df, t_column = 't', mov_column = 'moving', time_window = 24, prop_immobile = 0.01, resolution = 24):
        """ 
        A variation of curate dead animals to remove responses after an animal is presumed to have died.
        
        Args:
            mov_df (behavpy): A behavpy dataframe that has the full time series data for each specimen in the response dataframe
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            mov_column (str, optional): Name of column in the data containing movemment data as boolean values. Default is 'moving'
            time_window (int, optioanl): The size of the moving window (in hours) during which to define death. Default is 24.
            prop_immobile (float, optional): Proportion of immobility that counts as "dead" during time_window. Default is 0.01 (1%)
            resolution (int, optional): How much scanning windows overlap. Expressed as a factor. Default is 24.
        
        Returns:
            returns two modified behavpy object, the 1st the interaction behavpy dataframe that had the method called on it and the
            2nd the movment df with all the data points. Both filted to remove specimens where presumed dead.
        """

        def curate_filter(df, dict):
            return df[df[t_column].between(dict[df['id'].iloc[0]][0], dict[df['id'].iloc[0]][1])]

        if t_column not in self.columns.tolist() or t_column not in mov_df.columns.tolist():
            raise KeyError('Variable name entered, {}, for t_column is not a column heading!'.format(t_column))
        
        if mov_column not in mov_df.columns.tolist():
            raise KeyError('Variable name entered, {}, for mov_column is not a column heading!'.format(mov_column))

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
        return self.__class__(curated_puff, tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'],check = True), self.__class__(curated_df, tdf2.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'],check = True), 

    @staticmethod
    def _wrapped_interpolate_lin(data, var, step, t_col = 't'):
        """ Take the min and max time, create a time series at a given time step and interpolate missing values from the data """

        id = data['id'].iloc[0]
        sample_seq = np.arange(min(data[t_col]), np.nanmax(data[t_col]) + step, step)

        if len(sample_seq) < 3:
            return None

        f = np.interp(sample_seq, data[t_col].to_numpy(), data[var].to_numpy())

        return  pd.DataFrame(data = {'id' : [id] * len(sample_seq), t_col : sample_seq, var : f})


    def interpolate_linear(self, variable, step_size, t_column = 't'):
        """ A method to interpolate data from a given dataset according to a new time step size.
            The data must be ints or floats and have a linear distribution.

        Args:
            varibale (str): The column name of the variable you wish to interpolate.
            step_size (int): The amount of time in seconds the new time series should progress by. I.e. 60 = [0, 60, 120, 180...]
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
        
        Returns;
            returns a behavpy object with a single data column with interpolated data per specimen 
        """

        data = self.copy(deep = True)
        data = data.bin_time(variable = variable, t_column = t_column, bin_secs = step_size)
        data = data.rename(columns = {f'{t_column}_bin' : t_column, f'{variable}_mean' : variable})
        data = data.reset_index()
        return  self.__class__(data.groupby('id', group_keys = False).apply(partial(self._wrapped_interpolate_lin, 
                                                                                    var = variable, 
                                                                                    step = step_size)), 
        data.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    @staticmethod
    def _wrapped_bin_data(data, column, bin_column, function, bin_secs):
        """ a method that will bin all data poits to a larger time bin and then summarise a column """
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

    def bin_time(self, variable, bin_secs, function = 'mean', t_column = 't'):
        """
        A method bin the time series data into a user desired sized bin and further applying a function to a single column of choice across the new bins.
        
        Args:
            variable (str): The column in the data that you want to the function to be applied to post pivot
            bin_secs (int): The amount of time (in seconds) you want in each bin in seconds, e.g. 60 would be bins for every minutes
            function (str or user defined function, optional): The applied function to the grouped data, can be standard 'mean', 'max'.... ect, can also be a user defined function
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'

        
        Returns:
            returns a behavpy object with a single data column
        """

        if variable not in self.columns:
            raise KeyError('Column heading "{}", is not in the data table'.format(column))

        tdf = self.reset_index().copy(deep=True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bin_data,
                                                                                                column = variable, 
                                                                                                bin_column = t_column,
                                                                                                function = function, 
                                                                                                bin_secs = bin_secs
        )), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    def remove_first_last_bout(self):
        """
        A method to remove the first and last bout, only use for columns like 'moving' and 'asleep', that have continuous runs of True and False variables.
        For use with plotting and analysis where you are not sure if the starting and ending bouts weren't cut in two when filtering or stopping experiment.

        Args:
            None

        Returns:
            returns a modified behavpy object with fewer rows
        """
        
        def _wrapped_remove_first_last_bout(data):
            v = data['moving'].tolist() 
            try:
                change_list = np.where(np.roll(v,1)!=v)[0]
                ind1 = np.where(np.roll(v,1)!=v)[0][0]
                ind2 = np.where(np.roll(v,1)!=v)[0][-1]
            except:
                return data
            return data.iloc[ind1:ind2]

        tdf = self.reset_index().copy(deep=True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_remove_first_last_bout,
        )), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

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
        See function in that analysis folder for paramater details.
        
        returns:
            a behavpy object with added columns like 'moving' and 'beam_crosses'
        """

        if optional_columns is not None:
            if optional_columns not in self.columns:
                raise KeyError(f'Column heading {optional_columns}, is not in the data table')

        tdf = self.reset_index().copy(deep=True)
        return  self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_motion_detector,
                                                                                                        time_window_length = time_window_length,
                                                                                                        velocity_correction_coef = velocity_correction_coef,
                                                                                                        masking_duration = masking_duration,
                                                                                                        optional_columns = optional_columns
        )), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    @staticmethod
    def _wrapped_sleep_contiguous(d_small, mov_column, t_column, time_window_length, min_time_immobile):

        def sleep_contiguous(moving, fs, min_valid_time = 300):
            """ 
            Checks if contiguous bouts of immobility are greater than the minimum valid time given

            Args:
                moving (pandas series): series object comtaining the movement data of individual flies
                fs (int): sampling frequency (Hz) to scale minimum length to time in seconds
                min_valid_time (int, optional): min amount immobile time that counts as sleep, default is 300 (i.e 5 mins) 
            
            returns: 
                A list object to be added to a pandas dataframe
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
            raise KeyError(f'The movement column {mov_column} is not in the dataset')
        if t_column not in self.columns.tolist():
            raise KeyError(f'The time column {t_column} is not in the dataset')

        tdf = self.reset_index().copy(deep = True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_sleep_contiguous,
                                                                                                        mov_column = mov_column,
                                                                                                        t_column = t_column,
                                                                                                        time_window_length = time_window_length,
                                                                                                        min_time_immobile = min_time_immobile
        )), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    def feeding(self, food_position, dist_from_food = 0.05, micro_mov = 'micro', x_position = 'x', t_column = 't', check = False):
        """ A method that approximates the time spent feeding for flies in the ethoscope given their micromovements near to the food

        Args:
            food_postion (str): Must be either "outside" or "inside". This signifies the postion of the food in relation to the center of the arena.
            dist_from_food (float, optional): The distance measured between 0-1, as the x coordinate in the ethoscope, that you classify as being near the food. Default 0.05
            micro_mov (str, optional): The name of the column that contains the data for whether micromovements occurs as boolean values. Default is 'mirco'.
            x_position = string, the name of the column that contains the x postion
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            check (bool, optional): A check in place to remove the first hours worth of data, as often the tracking is still being honed in at the start. If set to False it will ignore this filter. Default is True.

        Returns:
            returns an augmented behavpy object with an addtional column 'feeding' with boolean variables where True equals predicted feeding.
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
            if check == True:
                t_diff = d[t_column].iloc[1] - d[t_column].iloc[0]
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
                d['feeding'] = np.where((d[x_position] > x_max-dist_from_food) & (d[micro_mov] == True), True, False)
            return d
            
        ds.reset_index(inplace = True)   
        ds_meta = ds.meta
        return self.__class__(ds.groupby('id', group_keys = False).apply(find_feed), ds_meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    # HMM SECTION

    @staticmethod
    def _hmm_decode(d, h, b, var, fun, t, return_type = 'array'):

        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        if var == 'moving' or var == 'asleep':
            d[var] = np.where(d[var] == True, 1, 0)

        # bin the data to 60 second intervals with a selected column and function on that column
        bin_df = d.bin_time(var, b, t_column = t, function = fun)
        gb = bin_df.groupby(bin_df.index, sort=False)[f'{var}_{fun}'].apply(list)
        time_list = bin_df.groupby(bin_df.index, sort=False)['t_bin'].apply(list)

        # logprob_list = []
        states_list = []
        df = pd.DataFrame()

        for i, t, id in zip(gb, time_list, time_list.index):
            seq_o = np.array(i)
            seq = seq_o.reshape(-1, 1)
            logprob, states = h.decode(seq)

            #logprob_list.append(logprob)
            if return_type == 'array':
                states_list.append(states)
            if return_type == 'table':
                label = [id] * len(t)
                previous_state = np.array(states[:-1], dtype = float)
                previous_state = np.insert(previous_state, 0, np.nan)
                all = zip(label, t, states, previous_state, seq_o)
                all = pd.DataFrame(data = all)
                df = pd.concat([df, all], ignore_index = False)
        
        if return_type == 'array':
            return states_list, time_list #, logprob_list
        if return_type == 'table':
            df.columns = ['id', 'bin', 'state', 'previous_state', var]
            return df


    # Internal methods for checking data/arguments before plotting
    def _check_hmm_shape(self, hm, lab, col):
        """
        Check the colours and labels passed to a plotting method are of equal length. If None then it will be populated with the defaults.
        """
        if isinstance(hm, list):
            hm = hm[0]

        if hm.transmat_.shape[0] == 4 and lab == None and col == None:
            _labels = self._hmm_labels
            _colours = self._hmm_colours
        elif hm.transmat_.shape[0] == 4 and lab == None and col != None:
            _labels = self._hmm_labels
            _colours = col
        elif hm.transmat_.shape[0] == 4 and lab != None and col == None:
            _labels = lab
            _colours = self._hmm_colours
        elif hm.transmat_.shape[0] != 4:
            if col is None or lab is None:
                raise RuntimeError('Your trained HMM is not 4 states, please provide the lables and colours for this hmm. See doc string for more info')
                # give generic names and populate with colours from the given palette 
                _labels = [f'state_{i}' for i in range(0, hm.transmat_.shape[0])]
                _colours = self.get_colours(hm.transmat_)
            elif len(col) != len(lab):
                raise RuntimeError('You have more or less states than colours, please rectify so the lists are equal in length')
            else:
                _labels = lab
                _colours = col
        else:
            _labels = lab
            _colours = col

        if len(_labels) != len(_colours):
            raise RuntimeError('Internal check fail: You have more or less states than colours, please rectify so they are equal in length')
        
        return _labels, _colours

    def _check_lists_hmm(self, f_col, f_arg, f_lab, h, b):
        """
        Check if the facet arguments match the labels or populate from the column if not.
        Check if there is more than one HMM object for HMM comparison. Populate hmm and bin lists accordingly.
        """
        if isinstance(h, list):
            assert isinstance(b, list)
            if len(h) != len(f_arg) or len(b) != len(f_arg):
                raise RuntimeError('There are not enough hmm models or bin intergers for the different groups or vice versa')
            else:
                h_list = h
                b_list = b

        if f_col is not None:
            if f_arg is None:
                f_arg = list(set(self.meta[f_col].tolist()))
                if f_lab is None:
                    string_args = []
                    for i in f_arg:
                        if i not in self.meta[f_col].tolist():
                            raise KeyError(f'Argument "{i}" is not in the meta column {f_col}')
                        string_args.append(str(i))
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    print("The facet labels don't match the length of the variables in the column. Using column variables instead")
                    f_lab = f_arg
            else:
                if f_lab is None:
                    string_args = []
                    for i in f_arg:
                        string_args.append(str(i))
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    print("The facet labels don't match the entered facet arguments in length. Using column variables instead")
                    f_lab = f_arg
        else:
            f_arg = [None]
            if f_lab is None:
                f_lab = ['']

        if isinstance(h, list) is False:
            h_list = [h]
            b_list = [b]
            if len(h_list) != len(f_arg):
                h_list = [h_list[0]] * len(f_arg)
            if len(b_list) != len(f_arg):
                b_list = [b_list[0]] * len(f_arg)

        return f_arg, f_lab, h_list, b_list


    @staticmethod
    def _hmm_table(start_prob, trans_prob, emission_prob, state_names, observable_names):
        """ 
        Prints a formatted table of the probabilities from a hmmlearn MultinomialHMM object
        """
        df_s = pd.DataFrame(start_prob)
        df_s = df_s.T
        df_s.columns = state_names
        print("Starting probabilty table: ")
        print(tabulate(df_s, headers = 'keys', tablefmt = "github") + "\n")
        print("Transition probabilty table: ")
        df_t = pd.DataFrame(trans_prob, index = state_names, columns = state_names)
        print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
        print("Emission probabilty table: ")
        df_e = pd.DataFrame(emission_prob, index = state_names, columns = observable_names)
        print(tabulate(df_e, headers = 'keys', tablefmt = "github") + "\n")

    def hmm_display(self, hmm, states, observables):
        """
        Prints to screen the transion probabilities for the hidden state and observables for a given hmmlearn hmm object
        """
        self._hmm_table(start_prob = hmm.startprob_, trans_prob = hmm.transmat_, emission_prob = hmm.emissionprob_, state_names = states, observable_names = observables)

    def hmm_train(self, states, observables, var_column, file_name, trans_probs = None, emiss_probs = None, start_probs = None, iterations = 10, hmm_iterations = 100, tol = 50, t_column = 't', bin_time = 60, test_size = 10, verbose = False):
        """
        Behavpy wrapper for the hmmlearn package which generates a Hidden Markov Model using the movement data from ethoscope data.
        If users want a restricted framework ,
        E.g. for random:

        There must be no NaNs in the training data

        Resultant hidden markov models will be saved as a .pkl file if file_name is provided
        Final trained model probability matrices will be printed to terminal at the end of the run time

            Args:
                states (list[string]): names of hidden states for the model to train to
                observables (list[string]): names of the observable states for the model to train to.
                    The length must be the same number as the different categories in you movement column.
                var_column (string): name for the column containing the variable of choice to train the model
                file_name (string): Name and path of the .pkl file the resultant trained model will be saved to.
                    If 'moving' or 'beam_crosses' it is assumed to be boolean and will be converted to 0, 1.
                    Variable categories must be represented by ints starting at 0. I.e. 3 observable states are 0, 1, 2.
                trans_probs (np.array, optional): Transtion probability matrix with shape 'len(states) x len(states)', 0's restrict the model from training any tranisitons between those states
                emiss_probs (np.array, optional): emission probability matrix with shape 'len(observables) x len(observables)', 0's same as above
                start_probs (np.array, optional): starting probability matrix with shape 'len(states) x 0', 0's same as above
                iterations (int, optional): The number of loops using a different randomised starting matrices, default is 10
                hmm_iterations (int, optional): An argument to be passed to hmmlearn, number of iterations of parameter updating without reaching tol before it stops, default is 100
                tol (int, optioanl): The convergence threshold passed tio hmmlearn, EM will stop if the gain in log-likelihood is below this value, default is 50
                t_column (string, optional): The name for the column containing the time series data, default is 't'
                bin_time (int, optional): The time in seconds the data will be binned to before the training begins, default is 60 (i.e 1 min)
                test_size (int, optional): The percentage as an int of 100 that you want to be the test portion in a test-train split, default is 10
                verbose (bool, optional): The argument for hmmlearn, whether per-iteration convergence reports are printed to terminal, default is False

        returns: 
            A trained hmmlearn HMM Multinomial object, which is also saved at the given path
        """
        
        if file_name.endswith('.pkl') is False:
            raise TypeError('enter a file name and type (.pkl) for the hmm object to be saved under')

        n_states = len(states)
        n_obs = len(observables)

        hmm_df = self.copy(deep = True)

        def bin_to_list(data, t_column, mov_var, bin):
            """ 
            Bins the time to the given integer and creates a nested list of the movement column by id
            """
            stat = 'max'
            data = data.reset_index()
            t_delta = data[t_column].iloc[1] - data[t_column].iloc[0]
            if t_delta != bin:
                data[t_column] = data[t_column].map(lambda t: bin * floor(t / bin))
                bin_gb = data.groupby(['id', t_var]).agg(**{
                    mov_var : (var_column, stat)
                })
                bin_gb.reset_index(level = 1, inplace = True)
                gb = bin_gb.groupby('id')[mov_var].apply(np.array)
            else:
                gb = data.groupby('id')[mov_var].apply(np.array)
            return gb

        if var_column == 'beam_crosses':
            hmm_df['active'] = np.where(hmm_df[var_column] == 0, 0, 1)
            gb = bin_to_list(hmm_df, t_column = t_column, mov_var = var_column, bin = bin_time)

        elif var_column == 'moving':
            hmm_df[var_column] = np.where(hmm_df[var_column] == True, 1, 0)
            gb = bin_to_list(hmm_df, t_column = t_column, mov_var = var_column, bin = bin_time)

        else:
            gb = bin_to_list(hmm_df, t_column = t_column, mov_var = var_column, bin = bin_time)

        # split runs into test and train lists
        test_train_split = round(len(gb) * (test_size/100))
        rand_runs = np.random.permutation(gb)
        train = rand_runs[test_train_split:]
        test = rand_runs[:test_train_split]

        len_seq_train = [len(ar) for ar in train]
        len_seq_test = [len(ar) for ar in test]

        # Augmenting shape to be accepted by hmmlearn
        seq_train = np.concatenate(train, 0)
        seq_train = seq_train.reshape(-1, 1)
        seq_test = np.concatenate(test, 0)
        seq_test = seq_test.reshape(-1, 1)

        for i in range(iterations):
            print(f"Iteration {i+1} of {iterations}")
            
            init_params = ''
            # h = hmm.MultinomialHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', verbose = verbose)
            h = hmm.CategoricalHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', verbose = verbose)

            if start_probs is None:
                init_params += 's'
            else:
                s_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in start_probs], dtype = np.float64)
                s_prob = np.array([[y / sum(x) for y in x] for x in s_prob], dtype = np.float64)
                h.startprob_ = s_prob

            if trans_probs is None:
                init_params += 't'
            else:
                # replace 'rand' with a new random number being 0-1
                t_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in trans_probs], dtype = np.float64)
                t_prob = np.array([[y / sum(x) for y in x] for x in t_prob], dtype = np.float64)
                h.transmat_ = t_prob

            if emiss_probs is None:
                init_params += 'e'
            else:
                # replace 'rand' with a new random number being 0-1
                em_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in emiss_probs], dtype = np.float64)
                em_prob = np.array([[y / sum(x) for y in x] for x in em_prob], dtype = np.float64)
                h.emissionprob_ = em_prob

            h.init_params = init_params
            h.n_features = n_obs # number of emission states

            # call the fit function on the dataset input
            h.fit(seq_train, len_seq_train)

            # Boolean output of if the number of runs convererged on set of appropriate probabilites for s, t, an e
            print("True Convergence:" + str(h.monitor_.history[-1] - h.monitor_.history[-2] < h.monitor_.tol))
            print("Final log liklihood score:" + str(h.score(seq_train, len_seq_train)))

            # On first iteration save and print the first trained matrix
            if i == 0:
                try:
                    h_old = pickle.load(open(file_name, "rb"))
                    if h.score(seq_test, len_seq_test) > h_old.score(seq_test, len_seq_test):
                        print('New Matrix:')
                        df_t = pd.DataFrame(h.transmat_, index = states, columns = states)
                        print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
                        
                        with open(file_name, "wb") as file: pickle.dump(h, file)
                except OSError as e:
                    with open(file_name, "wb") as file: pickle.dump(h, file)

            # After see score on the test portion and only save if better
            else:
                h_old = pickle.load(open(file_name, "rb"))
                if h.score(seq_test, len_seq_test) > h_old.score(seq_test, len_seq_test):
                    print('New Matrix:')
                    df_t = pd.DataFrame(h.transmat_, index = states, columns = states)
                    print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
                    with open(file_name, "wb") as file: pickle.dump(h, file)

            # Final iteration, load the best model and print its values
            if i+1 == iterations:
                h = pickle.load(open(file_name, "rb"))
                #print tables of trained emission probabilties, not accessible as objects for the user
                self._hmm_table(start_prob = h.startprob_, trans_prob = h.transmat_, emission_prob = h.emissionprob_, state_names = states, observable_names = observables)
                return h

    def get_hmm_raw(self, hmm, variable = 'moving', bin = 60, func = 'max'):
        """
        Decode all the time series per specimin, returning an augmented behavpy dataframe that has just one row per specimin and two columns, 
        one containing the decoded timeseries as a list and one with the given observable variable as a list

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): A trained categorical HMM object as produced by hmm_train.
                variable (str): The column name of the variable you wish to decode with the trained HMM. Default is 'moving'.
                bin (int): The amount of time (in seconds) you want to bin the time series to. Default is 60
                func (str): The function that is applied to the time series when binning it. Default is 'max'

        returns:
            A behavpy_HMM dataframe with columns bin (time), state, previous_state, moving
        """

        tdf = self.copy(deep=True)
        return self.__class__(self._hmm_decode(tdf, hmm, bin, variable, func, return_type= 'table'), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    # PERIODOGRAM SECTION

    def _validate(self):
        """ Validator to check further periodogram methods if the data is produced from the periodogram method """
        if  any([i not in self.columns.tolist() for i in ['period', 'power']]):
            raise AttributeError('This method is for the computed periodogram data only, please run the periodogram method on your data first')
        

    def _check_periodogram_input(self, v, per, per_range, t_col, wavelet_type = False):
        """ Method to check the input to periodogram methods"""

        periodogram_list = ['chi_squared', 'lomb_scargle', 'fourier', 'welch']

        if v not in self.columns.tolist():
            raise AttributeError(f"Variable column {v} is not a column title in your given dataset")

        if t_col not in self.columns.tolist():
            raise AttributeError(f"Time column {t_col} is not a column title in your given dataset")

        if wavelet_type is not False:
            fun = eval(per)
            return fun

        if per in periodogram_list:
            fun = eval(per)
        else:
            raise AttributeError(f"Unknown periodogram type, please use one of {*periodogram_list,}")

        if isinstance(per_range, list) is False and isinstance(per_range, np.array) is False:
            raise TypeError(f"per_range should be a list or nummpy array, please change")

        if isinstance(per_range, list) or isinstance(per_range, np.array):

            if len(per_range) != 2:
                raise TypeError("The period range can only be a tuple/array of length 2, please amend")

            if per_range[0] < 0 or per_range[1] < 0:
                raise ValueError(f"One or both of the values of the period_range given are negative, please amend")

        return fun

    def periodogram(self, mov_variable, periodogram, period_range = [10, 32], sampling_rate = 15, alpha = 0.01, t_col = 't'):
        """ A method to apply a periodogram function to given behavioural data. Call this method first to create an analysed dataset that can access 
        the other methods of this class 
        params:
        """

        fun = self._check_periodogram_input(mov_variable, periodogram, period_range, t_col)

        sampling_rate = 1 / (sampling_rate * 60)

        data = self.copy(deep = True)
        sampled_data = data.interpolate_linear(variable = mov_variable, step_size = 1 / sampling_rate)
        sampled_data = sampled_data.reset_index()
        return  self.__class__(sampled_data.groupby('id', group_keys = False)[[t_col, mov_variable]].apply(partial(fun, var = mov_variable, t_col = t_col, period_range = period_range, freq = sampling_rate, alpha = alpha)), data.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    @staticmethod
    def wavelet_types():
        wave_types = ['morl', 'cmor', 'mexh', 'shan', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8']
        return wave_types

    def _format_wavelet(self, mov_variable, sampling_rate = 15, wavelet_type = 'morl', t_col = 't'):
        """ A background method for the preperation of data for a wavelet plot.
            Head to https://pywavelets.readthedocs.io/en/latest/ for information about the pacakage and the other wavelet types
            This method will produce a single wavelet transformation plot, averaging the the data from across all specimens. It is therefore recommended you filter your dataset accordingly before applying 
            this method, i.e. by different experimental groups or a single specimen.
        """
        # check input and return the wavelet function with given wave type
        fun = self._check_periodogram_input(v = mov_variable, per = 'wavelet', per_range = None, t_col = t_col, wavelet_type = wavelet_type)
        sampling_rate = 1 / (sampling_rate * 60)

        # re-sample the data at the given rate and interpolate 
        data = self.copy(deep = True)
        sampled_data = data.interpolate_linear(variable = mov_variable, step_size = 1 / sampling_rate)
        # average across the dataset
        avg_data = sampled_data.groupby(t_col).agg(**{
                        mov_variable : (mov_variable, 'mean')
        })
        avg_data = avg_data.reset_index()

        return fun, avg_data

    @staticmethod
    def _wrapped_find_peaks(data, num, height = None):

        if height is True:
            peak_ind, _ = find_peaks(x = data['power'].to_numpy(), height = data['sig_threshold'].to_numpy())
        else:
            peak_ind, _ = find_peaks(x = data['power'].to_numpy())

        peaks = data['period'].to_numpy()[peak_ind]

        peak_power = data['power'].to_numpy()[peak_ind]
        order = peak_power.argsort()[::-1]
        ranks = order.argsort() + 1

        rank_dict = {k : int(v) for k,v in zip(peaks, ranks)}
        data['peak'] = data['period'].map(rank_dict).fillna(False)
        data['peak'] =  np.where(data['peak'] > num, False, data['peak'])

        return data
    
    def find_peaks(self, num_peaks):
        """ Find the peaks in a computed periodogram, a wrapper for the scipy find_peaks function"""
        self._validate()
        data = self.copy(deep=True)
        data = data.reset_index()
        if 'sig_threshold' in data.columns.tolist():
            return  self.__class__(data.groupby('id', group_keys = False).apply(partial(self._wrapped_find_peaks, num = num_peaks, height = True)), data.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)
        else:
            return  self.__class__(data.groupby('id', group_keys = False).apply(partial(self._wrapped_find_peaks, num = num_peaks)), data.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)
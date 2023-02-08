import pandas as pd
import numpy as np 
import warnings
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative

from math import floor, ceil, sqrt
from sys import exit
from scipy.stats import zscore
from functools import partial
from scipy.interpolate import interp1d

from ethoscopy.misc.format_warning import format_warning
from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.analyse import max_velocity_detector
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

    Initialisation Parameters:
    @data :
    @meta :
    @check

    """
    warnings.formatwarning = format_warning

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

    @staticmethod
    def _pop_std(array):
        return np.std(array, ddof = 0)

    @staticmethod
    def _check_conform(dataframe):
        """ 
        Checks the data augument is a pandas dataframe
        If metadata is provided and skip is False it will check as above and check the ID's in
        metadata match those in the data
        params: 
        @skip = boolean indicating whether to skip a check that unique id's are in both meta and data match 
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
            warnings.warn("There are ID's in the data that are not in the metadata, please check. You can skip this process by changing the parameter skip to False")
            exit()

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
        if all(isinstance(i, bool) for i in lst):
            y_range = [-0.025, 1.01]
            dtick = 0.2
        else:
            y_range = False
            dtick = False
        return y_range, dtick

    @staticmethod
    def _plot_ylayout(fig, yrange, ylabel, title, t0 = False, dtick = False, secondary = False, xdomain = False, tickvals = False, ticktext = False, ytype = "-", grid = False):
        """ create a plotly y-axis layout """
        if secondary is not False:
            fig['layout']['yaxis2'] = {}
            axis = 'yaxis2'
        else:
            axis = 'yaxis'
            fig['layout'].update(title = title,
                            plot_bgcolor = 'white',
                            legend = dict(
                            bgcolor = 'rgba(201, 201, 201, 1)',
                            bordercolor = 'grey',
                            font = dict(
                                size = 12
                                ),
                                x = 0.85,
                                y = 0.99
                            )
                        )
        fig['layout'][axis].update(
                        linecolor = 'black',
                        type = ytype,
                        title = dict(
                            text = ylabel,
                            font = dict(
                                size = 18,
                                color = 'black'
                            )
                        ),
                        rangemode = 'tozero',
                        zeroline = False,
                                ticks = 'outside',
                        tickwidth = 2,
                        tickfont = dict(
                            size = 18,
                            color = 'black'
                        ),
                        linewidth = 4
                    )
        if yrange is not False:
            fig['layout'][axis]['range'] = yrange
        if t0 is not False:
            fig['layout'][axis]['tick0'] = t0
        if dtick is not False:
            fig['layout'][axis]['dtick'] = dtick
        if secondary is not False:
            fig['layout'][axis]['side'] = 'right'
            fig['layout'][axis]['overlaying'] = 'y'
            fig['layout'][axis]['anchor'] = xdomain
        if tickvals is not False:
            fig['layout'][axis].update(tickvals = tickvals)
        if ticktext is not False:
            fig['layout'][axis].update(ticktext = ticktext)
        if grid is False:
            fig['layout'][axis]['showgrid'] = False
        else:
            fig['layout'][axis]['showgrid'] = True
            fig['layout'][axis]['gridcolor'] = 'black'

    @staticmethod
    def _plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = False, domains = False, axis = False, tickvals = False, ticktext = False, type = "-"):
        """ create a plotly x-axis layout """
        if domains is not False:
            fig['layout'][axis] = {}
        else:
            axis = 'xaxis'
        fig['layout'][axis].update(
                        showgrid = False,
                        linecolor = 'black',
                        type = type,
                        title = dict(
                            font = dict(
                                size = 18,
                                color = 'black'
                            )
                        ),
                        zeroline = False,
                                ticks = 'outside',
                        tickwidth = 2,
                        tickfont = dict(
                            size = 18,
                            color = 'black'
                        ),
                        linewidth = 4
                    )

        if xrange is not False:
            fig['layout'][axis].update(range = xrange)
        if t0 is not False:
            fig['layout'][axis].update(tick0 = t0)
        if dtick is not False:
            fig['layout'][axis].update(dtick = dtick)
        if xlabel is not False:
            fig['layout'][axis]['title'].update(text = xlabel)
        if domains is not False:
            fig['layout'][axis].update(domain = domains)
        if tickvals is not False:
            fig['layout'][axis].update(tickvals = tickvals)
        if ticktext is not False:
            fig['layout'][axis].update(ticktext = ticktext)

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

    @staticmethod
    def _plot_meanbox(median, q3, q1, x, colour, showlegend, name, xaxis):
        trace_box = go.Box(
            showlegend = showlegend,
            median = median,
            q3 = q3,
            q1 = q1,
            x = x,
            xaxis = xaxis,
            marker = dict(
                color = colour,
            ),
            boxpoints = False,
            jitter = 0.75, 
            pointpos = 0, 
            width = 0.9,
            name = name,
            legendgroup = name
        )
        return trace_box
    
    @staticmethod
    def _plot_boxpoints(y, x, colour, showlegend, name, xaxis, marker_size = None):
        trace_box = go.Box(
            showlegend = showlegend,
            y = y, 
            x = x,
            xaxis = xaxis,
            line = dict(
                color = 'rgba(0,0,0,0)'
            ),
            fillcolor = 'rgba(0,0,0,0)',
            marker = dict(
                color = colour,
                opacity = 0.5,
                size = 4
            ),
            boxpoints = 'all',
            jitter = 0.75, 
            pointpos = 0, 
            width = 0.9,
            name = name,
            legendgroup = name,
        )
        # if marker_size is not None:
        #     trace_box['marker_size'] = marker_size
        return trace_box

    @staticmethod  
    def _plot_line(df, x_col, name, marker_col):
        """ creates traces to plot a mean line with 95% confidence intervals for a plotly figure """

        max_var = max(df['mean'])

        upper_bound = go.Scatter(
        showlegend = False,
        legendgroup = name,
        x = df[x_col],
        y = df['y_max'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0,
                shape = 'spline'
                ),
        )
        trace = go.Scatter(
            legendgroup = name,
            x = df[x_col],
            y = df['mean'],
            mode = 'lines',
            name = name,
            line = dict(
                shape = 'spline',
                color = marker_col
                ),
            fill = 'tonexty'
        )

        lower_bound = go.Scatter(
            showlegend = False,
            legendgroup = name,
            x = df[x_col],
            y = df['y_min'],
            mode='lines',
            marker=dict(
                color = marker_col
                ),
            line=dict(width = 0,
                    shape = 'spline'
                    ),
            fill = 'tonexty'
        )  
        return upper_bound, trace, lower_bound, max_var

    @staticmethod
    def _get_colours(plot_list):
        """ returns a colour palette from plotly for plotly"""
        if len(plot_list) <= 11:
            return qualitative.Safe
        elif len(plot_list) < 24:
            return qualitative.Dark24
        else:
            warnings.warn('Too many sub groups to plot with the current colour palette')
            exit()
            
    # set meta as permenant attribute
    _metadata = ['meta']

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

            if isinstance(df, type(self)) is not True or isinstance(df, behavpy) is not True:
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
    def _wrapped_bout_analysis(data, var_name, as_hist, bin_size, max_bins, time_immobile, asleep):
        """ Finds runs of bouts of immobility or moving and sorts into a historgram per unqiue specimen in a behavpy dataframe"""

        index_name = data['id'].iloc[0]
        bin_width = bin_size*60
        
        dt = data[['t',var_name]].copy(deep = True)
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
        time = np.array([dt.t.iloc[0]])
        time = np.concatenate((time, bout_times['duration'].iloc[:-1]), axis = None)
        time = np.cumsum(time)
        bout_times['t'] = time
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
        return behavpy(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bout_analysis, 
                                                                                                var_name = sleep_column, 
                                                                                                as_hist = as_hist, 
                                                                                                bin_size = bin_size, 
                                                                                                max_bins = max_bins, 
                                                                                                time_immobile = time_immobile, 
                                                                                                asleep = asleep
            )), tdf.meta, check = True)

    def plot_sleep_bouts(self, sleep_column = 'asleep', facet_col = None, facet_arg = None, facet_labels = None, bin_size = 1, max_bins = 30, time_immobile = 5, asleep = True, title = '', grids = False):
        """ Plot with faceting the sleep bouts analysis function"""
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']
        
        col_list = self._get_colours(d_list)

        fig = go.Figure()
        max_y = []
        for data, name, col in zip(d_list, facet_labels, col_list):

            data = data.reset_index()
            bouts = data.groupby('id', group_keys = False).apply(partial(self._wrapped_bout_analysis, 
            var_name = sleep_column, 
            as_hist = True, 
            bin_size = bin_size, 
            max_bins = max_bins, 
            time_immobile = time_immobile, 
            asleep = asleep))

            plot_gb = bouts.groupby('bins').agg(**{
                    'mean' : ('prob', 'mean'),
                    'SD' : ('prob', self._pop_std),
                    'count' : ('prob', 'count')
            })
            plot_gb['SE'] = (1.96*plot_gb['SD']) / np.sqrt(plot_gb['count'])

            x = plot_gb.index.to_numpy()
            x = x / 60
            y = plot_gb['mean'].to_numpy()
            max_y.append(round(np.max(y) + 0.1, 1))

            trace = go.Bar(
                showlegend = True,
                name = name,
                x = x, 
                y = y,
                opacity = 0.5,
                marker = dict(
                    color = col,
                    line = dict(
                        color = col
                    )
                ),
                error_y = dict(
                    array = plot_gb['SE'].tolist(),
                    symmetric = True,
                    )
                )
            fig.add_trace(trace)
            
        fig.update_layout(barmode = 'overlay', bargap=0)
        self._plot_ylayout(fig, yrange = [0, max(max_y)], t0 = 0, dtick = max(max_y) / 5, ylabel = 'Proportion of total Bouts', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = [time_immobile, np.max(x)+0.5], t0 = time_immobile, dtick = bin_size, xlabel = 'Bouts (minutes)')

        return fig

    @staticmethod
    def _wrapped_curate_dead_animals(data, time_var, moving_var, time_window, prop_immobile, resolution): 
        
        time_window = (60 * 60 * time_window)
        d = data[[time_var, moving_var]].copy(deep = True)
        target_t = np.array(list(range(d.t.min().astype(int), d.t.max().astype(int), floor(time_window / resolution))))
        local_means = np.array([d[d[time_var].between(i, i + time_window)][moving_var].mean() for i in target_t])

        first_death_point = np.where(local_means <= prop_immobile, True, False)

        if any(first_death_point) is False:
            return data

        last_valid_point = target_t[first_death_point]

        curated_data = data[data[time_var].between(data.t.min(), last_valid_point[0])]
        return curated_data

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
        return behavpy(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_curate_dead_animals,
                                                                                                            time_var = t_column,
                                                                                                            moving_var = mov_column,
                                                                                                            time_window = time_window, 
                                                                                                            prop_immobile = prop_immobile,
                                                                                                            resolution = resolution
        )), tdf.meta, check = True)

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
        sample_seq = np.arange(min(data[t_col]), max(data[t_col]), step)
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
        data = data.rename(columns = {f'{t_column}_bin' : 't', f'{variable}_mean' : variable})
        data = data.reset_index()
        return  behavpy(data.groupby('id', group_keys = False).apply(partial(self._wrapped_interpolate, var = variable, step = step_size)), data.meta, check = True)

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
        return behavpy(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bin_data,
                                                                                                column = column, 
                                                                                                bin_column = t_column,
                                                                                                function = function, 
                                                                                                bin_secs = bin_secs
        )), tdf.meta, check = True)

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
        return  behavpy(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_motion_detector,
                                                                                                        time_window_length = time_window_length,
                                                                                                        velocity_correction_coef = velocity_correction_coef,
                                                                                                        masking_duration = masking_duration,
                                                                                                        optional_columns = optional_columns
        )), tdf.meta, check = True)

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

        time_map = pd.Series(range(d_small.t.iloc[0], 
                            d_small.t.iloc[-1] + time_window_length, 
                            time_window_length
                            ), name = 't')

        missing_values = time_map[~time_map.isin(d_small['t'].tolist())]
        d_small = d_small.merge(time_map, how = 'right', on = 't', copy = False).sort_values(by=['t'])
        d_small['is_interpolated'] = np.where(d_small['t'].isin(missing_values), True, False)
        d_small[mov_column] = np.where(d_small['is_interpolated'] == True, False, d_small[mov_column])

        d_small['asleep'] = sleep_contiguous(d_small[mov_column], 1/time_window_length, min_valid_time = min_time_immobile)

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
        return behavpy(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_sleep_contiguous,
                                                                                                        mov_column = mov_column,
                                                                                                        t_column = t_column,
                                                                                                        time_window_length = time_window_length,
                                                                                                        min_time_immobile = min_time_immobile
        )), tdf.meta, check = True)

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

        dict = self.meta[column].to_dict()

        def d2s(x):
            id = x.name
            seconds = dict.get(id) * (60*60*day_length)
            return x[t_column] + seconds

        if inplace is True:
            self[t_column] = self.apply(d2s, axis = 1)

        else:
            new = self.copy(deep = True)
            new[t_column] = new.apply(d2s, axis = 1)

            return new

    def heatmap(self, variable = 'moving', title = ''):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals using plotly
        
        Params:
        @variable = string, name for the column containing the variable of interest, the default is moving
        
        returns None
        """
        heatmap_df = self.copy(deep = True)
        # change movement values from boolean to intergers and bin to 30 mins finding the mean
        if variable == 'moving':
            heatmap_df[variable] = np.where(heatmap_df[variable] == True, 1, 0)

        heatmap_df = heatmap_df.bin_time(column = variable, bin_secs = 1800)
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

        fig = go.Figure(data=go.Heatmap(
                        z = gbm,
                        x = time_list,
                        y = id,
                        colorscale = 'Viridis'))

        fig.update_layout(
            title = title,
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

        return fig

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

    def plot_overtime(self, variable, wrapped = False, facet_col = None, facet_arg = None, facet_labels = None, avg_window = 30, day_length = 24, lights_off = 12, title = '', grids = False):

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)

        max_var = []
        y_range, dtick = self._check_boolean(list(self[variable].dropna()))
        if y_range is False:
            max_var.append(1)
        
        fig = go.Figure() 

        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = day_length/4, xlabel = 'ZT (Hours)')

        min_t = []
        max_t = []

        for data, name, col in zip(d_list, facet_labels, col_list):

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'grey'

            rolling_col = data.groupby(data.index, sort = False)[variable].rolling(avg_window).mean().reset_index(level = 0, drop = True)
            data['rolling'] = rolling_col.to_numpy()
            data = data.dropna(subset = ['rolling'])

            if wrapped is True:
                data['t'] = data['t'].map(lambda t: t % (60*60*day_length))
            data['t'] = data['t'].map(lambda t: t / (60*60))

            t_min = int(lights_off * floor(data.t.min() / lights_off))
            min_t.append(t_min)
            t_max = int(12 * ceil(data.t.max() / 12)) 
            max_t.append(t_max)

            gb_df = data.groupby('t').agg(**{
                        'mean' : ('rolling', 'mean'), 
                        'SD' : ('rolling', self._pop_std),
                        'count' : ('rolling', 'count')
                    })
            gb_df = gb_df.reset_index()
            gb_df['SE'] = (1.96*gb_df['SD']) / np.sqrt(gb_df['count'])
            gb_df['y_max'] = gb_df['mean'] + gb_df['SE']
            gb_df['y_min'] = gb_df['mean'] - gb_df['SE']

            upper, trace, lower, maxV = self._plot_line(df = gb_df, x_col = 't', name = name, marker_col = col)
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

            max_var.append(maxV)

        # Light-Dark annotaion bars
        bar_shapes = circadian_bars(min(min_t), max(max_t), max_y = max(max_var), day_length = day_length, lights_off = lights_off)
        fig.update_layout(shapes=list(bar_shapes.values()))
    
        fig['layout']['xaxis']['range'] = [min(min_t), max(max_t)]

        return fig

    def plot_quantify(self, variable, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False):

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)

        fig = go.Figure() 
        y_range, dtick = self._check_boolean(list(self[variable].dropna()))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = '')

        stats_dict = {}

        for data, name, col in zip(d_list, facet_labels, col_list):
            
            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'grey'

            data = data.dropna(subset = [variable])
            gdf = data.pivot(column = variable, function = fun)
            median, q3, q1, zlist = self._zscore_bootstrap(gdf[f'{variable}_{fun}'].to_numpy())
            stats_dict[name] = zlist

            fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
            x = [name], colour =  col, showlegend = False, name = name, xaxis = 'x'))

            fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [name], colour = col, 
            showlegend = False, name = name, xaxis = 'x'))

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        if fun != 'mean':
            fig['layout']['yaxis']['autorange'] = True

        return fig, stats_df

    def plot_day_night(self, variable, facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, lights_off = 12, title = '', grids = False):

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        fig = go.Figure()
        y_range, dtick = self._check_boolean(list(self[variable].dropna()))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)

        stats_dict = {}

        for data, name in zip(d_list, facet_labels):

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            data.add_day_phase(day_length = day_length, lights_off = lights_off)

            for c, phase in enumerate(['light', 'dark']):
                
                d = data[data['phase'] == phase]
                t_gb = d.pivot(column = variable, function = 'mean')
                median, q3, q1, zlist = self._zscore_bootstrap(t_gb[f'{variable}_mean'].to_numpy())
                stats_dict[f'{name}_{phase}'] = zlist

                if phase == 'light':
                    col = 'goldenrod'
                else:
                    col = 'black'

                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [name], colour =  col, showlegend = False, name = name, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [name], colour = col, 
                showlegend = False, name = name, xaxis = f'x{c+1}'))

                domains = np.arange(0, 2, 1/2)
                axis = f'xaxis{c+1}'
                self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = phase, domains = domains[c:c+2], axis = axis)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df
    
    def plot_compare_variables(self, variables, facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False):
        """the first variable in the list is the left hand axis, the last is the right hand axis"""

        assert(isinstance(variables, list))

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)
        
        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(facet_arg)

        fig = make_subplots(specs=[[{ "secondary_y" : True}]])

        stats_dict = {}

        for c, (data, name) in enumerate(zip(d_list, facet_labels)):   

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            bool_list = len(variables) * [False]
            bool_list[-1] = True

            for c2, (var, secondary) in enumerate(zip(variables, bool_list)):

                t_gb = data.pivot(column = var, function = 'mean')
                median, q3, q1, zlist = self._zscore_bootstrap(t_gb[f'{var}_mean'].to_numpy())
                stats_dict[f'{name}_{var}'] = zlist

                if len(facet_arg) == 1:
                    col_index = c2
                else:
                    col_index = c

                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [var], colour =  col_list[col_index], showlegend = False, name = var, xaxis = f'x{c+1}'), secondary_y = secondary)

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [var], colour = col_list[col_index], 
                showlegend = False, name = var, xaxis = f'x{c+1}'), secondary_y = secondary)

            domains = np.arange(0, 1+(1/len(facet_arg)), 1/len(facet_arg))
            axis = f'xaxis{c+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = name, domains = domains[c:c+2], axis = axis)

        axis_counter = 1
        for i in range(len(facet_arg) * (len(variables) * 2)):
            if i%((len(variables) * 2)) == 0 and i != 0:
                axis_counter += 1
            fig['data'][i]['xaxis'] = f'x{axis_counter}'

        y_range, dtick = self._check_boolean(list(self[variables[0]].dropna()))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[0], title = title, secondary = False, grid = grids)

        y_range, dtick = self._check_boolean(list(self[variables[-1]].dropna()))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[-1], title = title, secondary = True, xdomain = f'x{axis_counter}', grid = grids)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_anticipation_score(self, mov_variable = 'moving', facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, lights_off = 12, title = '', grids = False):

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)
        fig = go.Figure()

        self._plot_ylayout(fig, yrange = [0, 100], t0 = 0, dtick = 20, ylabel = 'Anticipatory Phase Score', title = title, grid = grids)

        def ap_score(total, small):
            try:
                return (small / total) * 100
            except ZeroDivisionError:
                return 0
        
        def analysis(data_list, phase):
            median_list = []
            q3_list = []
            q1_list = []
            con_list = []
            label_list = []

            if phase == 'Lights Off':
                start = [lights_off - 6, lights_off - 3]
                end = lights_off
            elif phase == 'Lights On':
                start = [day_length - 6, day_length - 3]
                end = day_length

            for d, l in zip(data_list, facet_labels):
                d = d.dropna(subset = [mov_variable])
                d.wrap_time(inplace = True)
                d = d.t_filter(start_time = start[0], end_time = end)
                total = d.pivot(column = mov_variable, function = 'sum')
                d = d.t_filter(start_time = start[1], end_time = end)
                small = d.groupby(d.index).agg(**{
                        'moving_small' : (mov_variable, 'sum')
                        })
                d = total.join(small)
                d = d.dropna()
                d['score'] = d[[f'{mov_variable}_sum', 'moving_small']].apply(lambda x: ap_score(*x), axis = 1)   
                zscore_list = d['score'].to_numpy()[np.abs(zscore(d['score'].to_numpy())) < 3]
                median_list.append(np.mean(zscore_list))
                q1, q3 = bootstrap(zscore_list)
                q3_list.append(q3)
                q1_list.append(q1)
                con_list.append(zscore_list)
                label_list.append(len(zscore_list) * [l])

            return median_list, q3_list, q1_list, con_list, label_list
        
        stats_dict = {}

        for c, phase in enumerate(['Lights Off', 'Lights On']):

            median_list, q3_list, q1_list, con_list, label_list = analysis(d_list, phase = phase)

            for c2, label in enumerate(facet_labels):

                if len(facet_arg) == 1:
                    col_index = c
                else:
                    col_index = c2

                stats_dict[f'{label}_{phase}'] = con_list[c2]

                fig.add_trace(self._plot_meanbox(median = [median_list[c2]], q3 = [q3_list[c2]], q1 = [q1_list[c2]], 
                x = [label], colour =  col_list[col_index], showlegend = False, name = label, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = con_list[c2], x = label_list[c2], colour = col_list[col_index], 
                showlegend = False, name = label, xaxis = f'x{c+1}'))

            domains = np.arange(0, 2, 1/2)
            axis = f'xaxis{c+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = phase, domains = domains[c:c+2], axis = axis)
        
        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    @staticmethod
    def _get_subplots(length):
        """Get the nearest higher square number"""
        square = sqrt(length) 
        closest = [floor(square)**2, ceil(square)**2]
        return int(sqrt(closest[1]))

    @staticmethod
    def _actogram_plot(fig, data, mov, day, row, col):
        try:
            max_days = int(data['day'].max())
            for i in range(max_days):
                x_list_2 = data['t_bin'][data['day'] == i+1].to_numpy() + day
                x_list = np.append(data['t_bin'][data['day'] == i].to_numpy(), x_list_2)
                y_list = np.append(data[f'{mov}_mean'][data['day'] == i].tolist(), data[f'{mov}_mean'][data['day'] == i+1].tolist())
                y_mod = np.array([i+1] * len(y_list)) - (y_list)
                fig.append_trace(go.Box(
                        showlegend = False,
                        median = (([i+1]*len(x_list) + y_mod) / 2),
                        q1 = y_mod,
                        q3 = [i+1]*len(x_list),
                        x = x_list,
                        marker = dict(
                            color = 'black',
                        ),
                        fillcolor = 'black',
                        boxpoints = False
                ), row = row, col = col)
        except ValueError:
            x_list = list(range(0,24,2))
            fig.append_trace(go.Box(
                    showlegend = False,
                    x = x_list,
                    marker = dict(
                        color = 'black',
                    ),
                    fillcolor = 'black',
                    boxpoints = False
            ), row = row, col = col)

    def plot_actogram(self, mov_variable = 'moving', bin_window = 5, t_column = 't', facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, title = ''):
        
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col != None:
            root = self._get_subplots(len(facet_arg))
            title_list = facet_labels
        else:
            facet_arg = [None]
            root =  self._get_subplots(1)
            title_list = ['']

        # make a square subplot domain
        fig = make_subplots(rows=root, cols=root, shared_xaxes = False, subplot_titles = title_list)
        col_list = list(range(1, root+1)) * root
        row_list = list([i] * root for i in range(1, root+1))
        row_list = [item for sublist in row_list for item in sublist]

        data = self.copy(deep=True)
        data = data.bin_time(mov_variable, bin_window*60, t_column = t_column)
        data.add_day_phase(time_column = 't_bin')

        for arg, col, row in zip(facet_arg, col_list, row_list): 

            if facet_col is not None:
                d = data.xmv(facet_col, arg)

                if len(d) == 0:
                    print(f'Group {arg} has no values and cannot be plotted')
                    continue

                d = d.groupby('t_bin').agg(**{
                    'moving_mean' : ('moving_mean', 'mean'),
                    'day' : ('day', 'max')
                })
                d.reset_index(inplace = True)
                d['t_bin'] = d['t_bin'].map(lambda t: (t % (day_length*60*60)) / (60*60))
            else:
                d = data.wrap_time(24, time_column = 't_bin')
                d['t_bin'] = d['t_bin'] / (60*60)

            self._actogram_plot(fig = fig, data = d, mov = mov_variable, day = day_length, row = row, col = col)

        fig.update_xaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            range = [0,48],
            tick0 = 0,
            dtick = 6,
            ticks = 'outside',
            tickfont = dict(
                size = 12
            ),
            showgrid = False
        )

        fig.update_yaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            range = [0,int(data['day'].max())],
            tick0 = 0,
            dtick = 1,
            ticks = 'outside',
            showgrid = True,
            autorange =  'reversed'
        )
        
        if facet_col == None:
            fig.update_annotations(font_size=8)
        fig['layout']['title'] = title
        fig['layout']['plot_bgcolor'] = 'white'

        return fig
    
    def plot_actogram_tile(self, mov_variable = 'moving', bin_window = 5, t_column = 't', labels = None, day_length = 24, title = ''):
        
        if labels is not None:
            if labels not in self.meta.columns.tolist():
                raise AttributeError(f'{labels} is not a column in the metadata')
            title_list = self.meta[labels].tolist() 
        else:
            title_list = self.meta.index.tolist()

        facet_arg = self.meta.index.tolist()
        root =  self._get_subplots(len(facet_arg))
        
        data = self.copy(deep=True)

        # make a square subplot domain
        fig = make_subplots(rows=root, cols=root, shared_xaxes = False, subplot_titles = title_list)
        col_list = list(range(1, root+1)) * root
        row_list = list([i] * root for i in range(1, root+1))
        row_list = [item for sublist in row_list for item in sublist]

        data = data.bin_time(mov_variable, bin_window*60, t_column = t_column)
        data.add_day_phase(time_column = 't_bin')

        for arg, col, row in zip(facet_arg, col_list, row_list): 

            d = data.xmv('id', arg)
            d = d.wrap_time(24, time_column = 't_bin')
            d['t_bin'] = d['t_bin'] / (60*60)

            self._actogram_plot(fig = fig, data = d, mov = mov_variable, day = day_length, row = row, col = col)

        fig.update_xaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            range = [0,48],
            tick0 = 0,
            dtick = 6,
            ticks = 'outside',
            tickfont = dict(
                size = 12
            ),
            showgrid = False
        )

        fig.update_yaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            range = [0,int(data['day'].max())],
            tick0 = 0,
            dtick = 1,
            ticks = 'outside',
            showgrid = True,
            autorange =  'reversed'
        )

        fig.update_annotations(font_size=8)
        fig['layout']['title'] = title
        fig['layout']['plot_bgcolor'] = 'white'

        return fig

    @staticmethod
    def _find_runs(mov, time, id):
        _, _, l = rle(mov)
        count_list = np.concatenate([np.append(np.arange(1, cnt + 1, 1)[: -1], np.nan) for cnt in l], dtype = float)
        previous_count_list = count_list[:-1]
        previous_count_list = np.insert(previous_count_list, 0, np.nan)
        return {'id': id, 't' : time, 'moving' : mov, 'activity_count' : count_list, 'previous_activity_count' : previous_count_list}

    def plot_response_overtime(self, response_df, activity = 'inactive', mov_variable = 'moving', bin = 60, title = '', grids = False): #facet_col = None, facet_arg = None, facet_labels = None,
        """ plot function to measure the response rate of flies to a puff of odour from a mAGO or AGO experiment over the consecutive minutes active or inactive
                response_df = behapy dataframe intially analysed by the puff_mago loading function
                activity = string, the choice to display reponse rate for continuous bounts of inactivity, activity, or both
        """
        
        activity_choice = ['inactive', 'active', 'both']
        if activity not in activity_choice:
            raise KeyError(f'activity argument must be one of {*activity_choice,}')

        # facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # d_list = []
        # if facet_col is not None:
        #     for arg in facet_arg:
        #         d_list.append(self.xmv(facet_col, arg))
        # else:
        #     d_list = [self.copy(deep = True)]
        #     facet_labels = ['']

        # col_list = self._get_colours(d_list)

        fig = go.Figure() 
        y_range, dtick = self._check_boolean(list(self[mov_variable].dropna()))

        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Response Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = 1, xlabel = f'Consecutive Minutes {activity}')

        def activity_count(df, puff_df):

            bin_df = df.bin_time(mov_variable, bin, function = 'max')
            mov_gb = bin_df.groupby(bin_df.index)[f'{mov_variable}_max'].apply(np.array)
            time_gb = bin_df.groupby(bin_df.index)['t_bin'].apply(np.array)
            zip_gb = zip(mov_gb, time_gb, mov_gb.index)

            all_runs = []

            for m, t, id in zip_gb:
                spec_run = self._find_runs(m, t, id)
                all_runs.append(spec_run)

            counted_df = pd.concat([pd.DataFrame(specimen) for specimen in all_runs])

            puff_df['t'] = puff_df['interaction_t'].map(lambda t: t % 86400)
            puff_df['t'] = puff_df['interaction_t'].map(lambda t:  bin * floor(t / bin))
            puff_df.reset_index(inplace = True)

            merged = pd.merge(counted_df, puff_df, how = 'inner', on = ['id', 't'])
            merged['t_check'] = merged.interaction_t + merged.t_rel
            merged['t_check'] = merged['t_check'].map(lambda t:  bin * floor(t / bin))
            merged['previous_activity_count'] = np.where(merged['t_check'] > merged['t'], merged['activity_count'], merged['previous_activity_count'])
            merged.dropna(subset = ['previous_activity_count'], inplace=True)

            interaction_dict = {}
            for i in [0, 1]:
                first_filter = merged[merged['moving'] == i]
                for q in list(set(first_filter.has_interacted)):
                    second_filter = first_filter[first_filter['has_interacted'] == q]
                    big_gb = second_filter.groupby('previous_activity_count').agg(**{
                                    'mean' : ('has_responded', 'mean'),
                                    'count' : ('has_responded', 'count'),
                                    'ci' : ('has_responded', bootstrap)
                        })
                    big_gb[['y_max', 'y_min']] = pd.DataFrame(big_gb['ci'].tolist(), index =  big_gb.index)
                    big_gb.drop('ci', axis = 1, inplace = True)
                    big_gb.reset_index(inplace=True)
                    big_gb['previous_activity_count'] = big_gb['previous_activity_count'].astype(int)

                    interaction_dict[f'{i}_{int(q)}'] = big_gb

            return interaction_dict

        max_x = []

        # for data, name in zip(d_list, facet_labels):

        #     if len(data) == 0:
        #         print(f'Group {name} has no values and cannot be plotted')
        #         continue
        
        response_dict = activity_count(self, response_df)
    
        if activity == 'inactive':
            col_list = ['blue', 'black']
            plot_list = ['0_1', '0_2']
            label_list = ['Inactive', 'Inactive Spon. Mov.']
        elif activity == 'active':
            col_list = ['red', 'grey']
            plot_list = ['1_1', '1_2']
            label_list = ['Active', 'Active Spon. Mov.']
        else:
            col_list = ['blue', 'black', 'red', 'grey']
            plot_list = ['0_1', '0_2', '1_1', '1_2']
            label_list = ['Inactive', 'Inactive Spon. Mov.', 'Active', 'Active Spon. Mov.']

        for plot, col, label in zip(plot_list, col_list, label_list):
            small_data = response_dict[plot]
            if len(small_data) == 0:
                print(f'Group {label} has no values and cannot be plotted')
                continue
            max_x.append(max(small_data['previous_activity_count']))
            upper, trace, lower, _ = self._plot_line(df = small_data, x_col = 'previous_activity_count', name = label, marker_col = col)
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)
                
        fig['layout']['xaxis']['range'] = [1, max(max_x)]

        return fig

    def plot_response_quantify(self, response_col = 'has_responded', facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False): 
        """ A augmented version of plot_quanitfy that looks for true and false (spontaneous movement) interactions """

        if response_col not in self.columns.tolist():
            raise KeyError(f'The column you gave {response_col}, is not in the data. Check you have analyed the dataset with puff_mago')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)

        fig = go.Figure() 
        y_range, dtick = self._check_boolean(list(self[response_col].dropna()))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Resonse Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = '')

        stats_dict = {}

        for data, name, col in zip(d_list, facet_labels, col_list):
            
            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            for q in list(set(data.has_interacted)):

                filtered = data[data['has_interacted'] == q]
                filtered = filtered.dropna(subset = [response_col])

                gdf = data.pivot(column = response_col, function = 'mean')
                median, q3, q1, zlist = self._zscore_bootstrap(gdf[f'{response_col}_mean'].to_numpy())
                stats_dict[f'{name}_{q}'] = zlist

                if q == 1:
                    col = col
                    name = name
                else:
                    col = 'grey'
                    name = f'{name} Spon. Mov'

                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [name], colour =  col, showlegend = False, name = name, xaxis = 'x'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [name], colour = col, 
                showlegend = False, name = name, xaxis = 'x'))

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df
import pandas as pd
import numpy as np 

import plotly.graph_objs as go 
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative

from math import floor, ceil, sqrt
from scipy.stats import zscore
from functools import partial, update_wrapper
from scipy.interpolate import interp1d
from colour import Color
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

    def __init__(self, data, meta, colour = 'Safe', long_colour = 'Dark24', check = False, index= None, columns=None, dtype=None, copy=True):
        super(behavpy, self).__init__(data=data,
                                        index=index,
                                        columns=columns,
                                        dtype=dtype,
                                        copy=copy)

        self.meta = meta   
        if check is True:
            self._check_conform(self)
        self.attrs = {'short_col' : colour, 'long_col' : long_colour}

    # set meta as permenant attribute
    _metadata = ['meta']

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
            warnings.warn("There are ID's in the data that are not in the metadata, please check")

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
                    raise ValueError("The facet labels don't match the length of the variables in the column. Using column variables instead")
                    f_lab = string_args
            else:
                string_args = []
                for i in f_arg:
                    if i not in self.meta[f_col].tolist():
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
                                x = 1.01,
                                y = 0.5
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
                        linewidth = 2.5
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
                        linewidth = 2.5
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
        """ Calculate the z score of a given array, remove any values +- 3 SD and then perform bootstrapping on the remaining
        returns the mean and then several lists with the confidence intervals and z-scored values
        """
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
        """ For quantify plots, creates a box with a mean line and then extensions showing the confidence intervals 
        """
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
        """ Accompanies _plot_meanbox. Use this method to plot the data points as dots over the meanbox
        """
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

        max_var = np.nanmax(df['mean'])

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
    def _check_rgb(lst):
        """ checks if the colour list is RGB plotly colours, if it is it changes it to its hex code """
        try:
            return [Color(rgb = tuple(np.array(eval(col[3:])) / 255)) for col in lst]
        except:
            return lst

    def _get_colours(self, plot_list):
        """ returns a colour palette from plotly for plotly """
        if len(plot_list) <= len(getattr(qualitative, self.attrs['short_col'])):
            return getattr(qualitative, self.attrs['short_col'])
        elif len(plot_list) <= len(getattr(qualitative, self.attrs['long_col'])):
            return getattr(qualitative, self.attrs['long_col'])
        elif len(plot_list) <= 48:
            return qualitative.Dark24 + qualitative.Light24
        else:
            raise IndexError('Too many sub groups to plot with the current colour palette (max is 48)')

    def _adjust_colours(self, colour_list):
        """ Takes a list of colours written names or hex codes.
        Returns two lists of hex colour codes. The first is a lighter version of the second which is the original.
        """
        def adjust_color_lighten(r,g,b, factor):
            return [round(255 - (255-r)*(1-factor)), round(255 - (255-g)*(1-factor)), round(255 - (255-b)*(1-factor))]

        colour_list = self._check_rgb(colour_list)

        start_colours = []
        end_colours = []
        for col in colour_list:
            c = Color(col)
            c_hex = c.hex
            end_colours.append(c_hex)
            r, g, b = c.rgb
            r, g, b = adjust_color_lighten(r*255, g*255, b*255, 0.75)
            start_hex = "#%02x%02x%02x" % (r,g,b)
            start_colours.append(start_hex)

        return start_colours, end_colours

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
        col_dict = self.attrs
        for df in args:

            if isinstance(df, type(self)) is not True or isinstance(df, behavpy) is not True:
                raise TypeError('Object(s) to concat is(are) not a Behavpy object')

            meta_list.append(df.meta)
            data_list.append(df)

        meta = pd.concat(meta_list)
        new = pd.concat(data_list)

        new.meta = meta
        new.attrs = col_dict

        return new

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
        return type(self)(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bout_analysis, 
                                                                                                var_name = sleep_column, 
                                                                                                as_hist = as_hist, 
                                                                                                bin_size = bin_size, 
                                                                                                max_bins = max_bins, 
                                                                                                time_immobile = time_immobile, 
                                                                                                asleep = asleep,
                                                                                                t_column = t_column
            )), tdf.meta, colour = tdf.attrs['short_col'], long_colour = tdf.attrs['long_col'], check = True)

    def plot_sleep_bouts(self, sleep_column = 'asleep', facet_col = None, facet_arg = None, facet_labels = None, bin_size = 1, max_bins = 60, time_immobile = 5, asleep = True, title = '', t_column = 't', grids = False):
        """ 
        Generates a plotly histogram that shows the distribution of bouts (can be either sleep bouts or wake bouts).
        This method uses the sleep_bout_analysis method to generate the data for the plot.

        Args:
            sleep_column (str, optional): The name of the column that contains the sleep data which should be in a boolean format (True for alseep and vice versa). Default is 'asleep'.
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
            bin_size (int, optional): The size of the histogram bins that will be plotted. Default is 1
            max_bins (int, optional): The maximum number of bins you want the data sorted into. Default is 60
            time_immobile (int, optional): The minimum time in minutes that you want to classify your sleep or wake bout. Default is 5 (as per 5 minute sleep rule for Drosophila)
            asleep (bool, optional): If true then only sleep bouts will be plotted, if false then wake bouts. Default is True.
            title (str, optional): The title of the plot. Default is an empty string.
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False

        returns:
            returns a plotly figure object
        """
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arf]
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
            asleep = asleep,
            t_column = t_column))

            plot_gb = bouts.groupby('bins').agg(**{
                    'mean' : ('prob', 'mean'),
                    'SD' : ('prob', 'std'),
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
        self._plot_ylayout(fig, yrange = [0, np.nanmax(max_y)], t0 = 0, dtick = np.nanmax(max_y) / 5, ylabel = 'Proportion of total Bouts', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = [time_immobile, np.max(x)+0.5], t0 = time_immobile, dtick = bin_size, xlabel = 'Bouts (minutes)')

        return fig

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
        return type(self)(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_curate_dead_animals,
                                                                                                            time_var = t_column,
                                                                                                            moving_var = mov_column,
                                                                                                            time_window = time_window, 
                                                                                                            prop_immobile = prop_immobile,
                                                                                                            resolution = resolution
        )), tdf.meta, colour = tdf.attrs['short_col'], long_colour = tdf.attrs['long_col'], check = True)

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
        return type(self)(curated_puff, tdf.meta, colour = tdf.attrs['short_col'], long_colour = tdf.attrs['long_col'], check = True), type(self)(curated_df, tdf2.meta, colour = tdf.attrs['short_col'], long_colour = tdf.attrs['long_col'], check = True), 
        

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
        """ A method to interpolate data from a given dataset according to a new time step size.
            The data must by etiher a int, float, or bool.

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
        return  type(self)(data.groupby('id', group_keys = False).apply(partial(self._wrapped_interpolate, var = variable, step = step_size)), data.meta, colour = self.attrs['short_col'], long_colour = self.attrs['long_col'], check = True)

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
        return type(self)(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bin_data,
                                                                                                column = variable, 
                                                                                                bin_column = t_column,
                                                                                                function = function, 
                                                                                                bin_secs = bin_secs
        )), tdf.meta, colour = self.attrs['short_col'], long_colour = self.attrs['long_col'], check = True)

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

    def add_day_phase(self, time_column = 't', day_length = 24, lights_off = 12, inplace = True):
        """ 
        Adds a column called 'phase' with either light or dark as catergories according to its time compared to the reference hour
        Adds a column with the day the row is in, starting with 1 as the first day and increasing sequentially.

        Args:
            time_column (str): The name of column containing the timing data (in seconds). Default is 't'.
            day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
            inplace (bool, optional): 
            
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
                raise KeyError(f'Column heading {optional_columns}, is not in the data table')

        tdf = self.reset_index().copy(deep=True)
        return  type(self)(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_motion_detector,
                                                                                                        time_window_length = time_window_length,
                                                                                                        velocity_correction_coef = velocity_correction_coef,
                                                                                                        masking_duration = masking_duration,
                                                                                                        optional_columns = optional_columns
        )), tdf.meta, colour = self.attrs['short_col'], long_colour = self.attrs['long_col'], check = True)

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
            raise KeyError(f'The movement column {mov_column} is not in the dataset')
        if t_column not in self.columns.tolist():
            raise KeyError(f'The time column {t_column} is not in the dataset')

        tdf = self.reset_index().copy(deep = True)
        return type(self)(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_sleep_contiguous,
                                                                                                        mov_column = mov_column,
                                                                                                        t_column = t_column,
                                                                                                        time_window_length = time_window_length,
                                                                                                        min_time_immobile = min_time_immobile
        )), tdf.meta, colour = self.attrs['short_col'], long_colour = self.attrs['long_col'], check = True)

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
            raise KeyError('Baseline days column: "{}", is not in the metadata table'.format(column))

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

    def heatmap(self, variable = 'moving', t_column = 't', title = ''):
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

        heatmap_df = heatmap_df.bin_time(variable = variable, bin_secs = 1800, t_column = t_column)
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
    def _generate_overtime_plot(data, name, col, var, avg_win, wrap, day_len, light_off, t_col):

        if len(data) == 0:
            print(f'Group {name} has no values and cannot be plotted')
            return None, None, None, None, None, None

        if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
            col = 'grey'

        if avg_win  != False:
            rolling_col = data.groupby(data.index, sort = False)[var].rolling(avg_win, min_periods = 1).mean().reset_index(level = 0, drop = True)
            data['rolling'] = rolling_col.to_numpy()
            # removing dropna to speed it up
            # data = data.dropna(subset = ['rolling'])
        else:
            data = data.rename(columns={var: 'rolling'})

        if day_len != False:
            if wrap is True:
                data[t_col] = data[t_col] % (60*60*day_len)
            data[t_col] = data[t_col] / (60*60)

            t_min = int(light_off * floor(data[t_col].min() / light_off))
            t_max = int(12 * ceil(data[t_col].max() / 12)) 
        else:
            t_min, t_max = None, None

        # Not using bootstrapping here as it takes too much time
        gb_df = data.groupby(t_col).agg(**{
                    'mean' : ('rolling', 'mean'), 
                    'SD' : ('rolling', 'std'),
                    'count' : ('rolling', 'count')
                })
        gb_df = gb_df.reset_index()
        gb_df['SE'] = (1.96*gb_df['SD']) / np.sqrt(gb_df['count'])
        gb_df['y_max'] = gb_df['mean'] + gb_df['SE']
        gb_df['y_min'] = gb_df['mean'] - gb_df['SE']

        upper, trace, lower, maxV = data._plot_line(df = gb_df, x_col = t_col, name = name, marker_col = col)

        return upper, trace, lower, maxV, t_min, t_max

    def plot_overtime(self, variable, wrapped = False, facet_col = None, facet_arg = None, facet_labels = None, avg_window = 180, day_length = 24, lights_off = 12, title = '', grids = False, t_column = 't'):
        """
        A plot to view a variable of choice over an experiment of experimental day. The variable must be within the data. White and black boxes are generated to signify when lights are on and off and can be augmented.
        Args:
            variable (str): The name of the column you wish to plot from your data. 
            wrapped (bool, optional): If true the data is augmented to represent one day, combining data of the same time on consequtive days.
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
            avg_window (int, optional): The number that is applied to the rolling smoothing function. The default is 180 which works best with time difference of 10 seconds between rows.
            day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
            title (str, optional): The title of the plot. Default is an empty string.
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'

        
        returns:
            returns a plotly figure object
        """
        assert isinstance(wrapped, bool)
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)

        max_var = []
        y_range, dtick = self._check_boolean(list(self[variable]))
        if y_range is not False:
            max_var.append(1)
        
        fig = go.Figure() 

        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = day_length/4, xlabel = 'ZT (Hours)')

        min_t = []
        max_t = []

        for data, name, col in zip(d_list, facet_labels, col_list):
            upper, trace, lower, maxV, t_min, t_max = self._generate_overtime_plot(data = data, name = name, col = col, var = variable, 
                                                                                    avg_win = avg_window, wrap = wrapped, day_len = day_length, 
                                                                                    light_off= lights_off, t_col = t_column)
            if upper is None:
                continue

            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

            max_var.append(maxV)
            min_t.append(t_min)
            max_t.append(t_max)

        # Light-Dark annotaion bars
        bar_shapes, min_bar = circadian_bars(np.nanmin(min_t), np.nanmax(max_t), max_y = np.nanmax(max_var), day_length = day_length, lights_off = lights_off)
        fig.update_layout(shapes=list(bar_shapes.values()))
    
        fig['layout']['xaxis']['range'] = [np.nanmin(min_t), np.nanmax(max_t)]
        if min_bar < 0:
            fig['layout']['yaxis']['range'] = [min_bar, np.nanmax(max_var)+0.01]

        return fig

    def plot_quantify(self, variable, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False):
        """
        A plot that finds the average (default mean) for a given variable per specimen. The plots will show each specimens average and a box representing the mean and 95% confidence intervals.
        Addtionally, a pandas dataframe is generated that contains the averages per specimen per group for users to perform statistics with.

        Args:
            variable (str): The name of the column you wish to plot from your data. 
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
            fun (str, optional): The average function that is applied to the data. Must be one of 'mean', 'median', 'count'.
            title (str, optional): The title of the plot. Default is an empty string.
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False

        Returns:
            returns a plotly figure object and a pandas DataFrame
        """
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)

        fig = go.Figure() 
        y_range, dtick = self._check_boolean(list(self[variable]))
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
            gdf = data.analyse_column(column = variable, function = fun)
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

    def plot_day_night(self, variable, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', day_length = 24, lights_off = 12, title = '', grids = False):
        """
        A plot that shows the average of a varaible split between the day (lights on) and night (lights off).
        Addtionally, a pandas dataframe is generated that contains the averages per specimen per group for users to perform statistics with.
        Args:
            variable (str): The name of the column you wish to plot from your data. 
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
            fun (str, optional): The average function that is applied to the data. Must be one of 'mean', 'median', 'count'.
            day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
            title (str, optional): The title of the plot. Default is an empty string.
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False

        returns:
            returns a plotly figure object and a pandas DataFrame
        """
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        fig = go.Figure()
        y_range, dtick = self._check_boolean(list(self[variable]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)

        stats_dict = {}

        for data, name in zip(d_list, facet_labels):

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            data.add_day_phase(day_length = day_length, lights_off = lights_off)

            for c, phase in enumerate(['light', 'dark']):
                
                d = data[data['phase'] == phase]
                t_gb = d.analyse_column(column = variable, function = fun)
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
        
        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
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

                t_gb = data.analyse_column(column = var, function = 'mean')
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

        y_range, dtick = self._check_boolean(list(self[variables[0]]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[0], title = title, secondary = False, grid = grids)

        y_range, dtick = self._check_boolean(list(self[variables[-1]]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[-1], title = title, secondary = True, xdomain = f'x{axis_counter}', grid = grids)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_anticipation_score(self, mov_variable = 'moving', facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, lights_off = 12, title = '', grids = False):

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
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
                total = d.analyse_column(column = mov_variable, function = 'sum')
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
        data.add_day_phase(time_column = f'{t_column}_bin')

        for arg, col, row in zip(facet_arg, col_list, row_list): 

            if facet_col is not None:
                d = data.xmv(facet_col, arg)

                if len(d) == 0:
                    print(f'Group {arg} has no values and cannot be plotted')
                    continue

                d = d.groupby(f'{t_column}_bin').agg(**{
                    'moving_mean' : ('moving_mean', 'mean'),
                    'day' : ('day', 'max')
                })
                d.reset_index(inplace = True)
                d[f'{t_column}_bin'] = (d[f'{t_column}_bin'] % (day_length*60*60)) / (60*60)
            else:
                d = data.wrap_time(24, time_column = f'{t_column}_bin')
                d[f'{t_column}_bin'] = d[f'{t_column}_bin'] / (60*60)

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
        # count_list = np.concatenate([np.append(np.arange(1, cnt + 1, 1)[: -1], np.nan) for cnt in l], dtype = float)
        count_list = np.concatenate([np.arange(1, cnt + 1, 1) for cnt in l], dtype = float)
        previous_count_list = count_list[:-1]
        previous_count_list = np.insert(previous_count_list, 0, np.nan)
        previous_mov = mov[:-1].astype(float)
        previous_mov = np.insert(previous_mov, 0, np.nan)
        return {'id': id, 't' : time, 'moving' : mov, 'previous_moving' : previous_mov, 'activity_count' : count_list, 'previous_activity_count' : previous_count_list}

    def plot_response_over_bouts(self, response_df, activity = 'inactive', mov_variable = 'moving', facet_col = None, facet_arg = None, facet_labels = None, title = '', t_column = 't', grids = False):
        """ A plotting function for AGO or mAGO datasets that have been loaded with the analysing function stimulus_response.
        Plot function to measure the response rate of flies to a stimulus from a mAGO or AGO experiment over the consecutive minutes active or inactive

        Params:
        @response_df = behavpy, behapy dataframe intially analysed by the stimulus_response loading function
        @activity = string, the choice to display reponse rate for continuous bounts of inactivity, activity, or both. Choice one of ['inactive', 'active', 'both']
        @mov_variable = string, the name of the column that contains the response per each interaction, should be boolean values
        @bin = int, the value in seconds time should be binned to and then count consecutive bouts
        @title = string, a title for the plotted figure
        @grids = bool, true/false whether the resulting figure should have grids

        returns a plotly figure object
        """
        
        activity_choice = ['inactive', 'active', 'both']
        if activity not in activity_choice:
            raise KeyError(f'activity argument must be one of {*activity_choice,}')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        if activity == 'inactive':
            col_list = [['blue'], ['black']]
            plot_list = ['0_1', '0_2']
            label_list = ['Inactive', 'Inactive Spon. Mov.']
        elif activity == 'active':
            col_list = [['red'], ['grey']]
            plot_list = ['1_1', '1_2']
            label_list = ['Active', 'Active Spon. Mov.']
        else:
            col_list = [['blue'], ['black'], ['red'], ['grey']]
            plot_list = ['0_1', '0_2', '1_1', '1_2']
            label_list = ['Inactive', 'Inactive Spon. Mov.', 'Active', 'Active Spon. Mov.']

        if facet_col is not None:
            
            if activity == 'both':
                start_colours, end_colours = self._adjust_colours([col[0] for col in col_list])
                col_list = []
                colours_dict = {'start' : start_colours, 'end' : end_colours}
                for c in range(len(plot_list)):
                    start_color = colours_dict.get('start')[c]
                    end_color = colours_dict.get('end')[c]
                    N = len(facet_arg)
                    col_list.append([x.hex for x in list(Color(start_color).range_to(Color(end_color), N))])
            
            else:
                col_list = [tuple(np.array(eval(col[3:])) / 255) for col in self._get_colours(facet_arg)]
                end_colours, start_colours = self._adjust_colours(col_list, rgb = True)
                col_list = [start_colours, end_colours]

        fig = go.Figure() 
        y_range, dtick = self._check_boolean(list(self[mov_variable]))

        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Response Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = 1, xlabel = f'Consecutive Minutes {activity}')

        def activity_count(df, puff_df):
            puff_df = puff_df.copy(deep=True)
            df[mov_variable] = np.where(df[mov_variable] == True, 1, 0)
            bin_df = df.bin_time(mov_variable, 60, function = 'max', t_column = t_column)
            mov_gb = bin_df.groupby(bin_df.index)[f'{mov_variable}_max'].apply(np.array)
            time_gb = bin_df.groupby(bin_df.index)[f'{t_column}_bin'].apply(np.array)
            zip_gb = zip(mov_gb, time_gb, mov_gb.index)

            all_runs = []

            for m, t, id in zip_gb:
                spec_run = self._find_runs(m, t, id)
                all_runs.append(spec_run)

            counted_df = pd.concat([pd.DataFrame(specimen) for specimen in all_runs])

            puff_df[t_column] = puff_df['interaction_t'] % 86400
            puff_df[t_column] = puff_df['interaction_t'].map(lambda t:  60 * floor(t / 60))
            puff_df.reset_index(inplace = True)

            merged = pd.merge(counted_df, puff_df, how = 'inner', on = ['id', t_column])
            merged['t_check'] = merged.interaction_t + merged.t_rel
            merged['t_check'] = merged['t_check'].map(lambda t:  60 * floor(t / 60))            
            merged['previous_activity_count'] = np.where(merged['t_check'] > merged[t_column], merged['activity_count'], merged['previous_activity_count'])
            merged.dropna(subset = ['previous_activity_count'], inplace=True)

            interaction_dict = {}
            for i in [0, 1]:
                first_filter = merged[merged['previous_moving'] == i]
                if len(first_filter) == 0:
                    for q in [1, 2]:
                        interaction_dict[f'{i}_{int(q)}'] = None
                        continue
                # for q in list(set(first_filter.has_interacted)):
                for q in [1, 2]:
                    second_filter = first_filter[first_filter['has_interacted'] == q]
                    if len(second_filter) == 0:
                        interaction_dict[f'{i}_{int(q)}'] = None
                        continue
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

        for c1, (data, name) in enumerate(zip(d_list, facet_labels)):

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue
        
            response_dict = activity_count(data, response_df)

            for c2, (plot, label) in enumerate(zip(plot_list, label_list)):

                col = col_list[c2][c1]
                small_data = response_dict[plot]

                label = f'{name} {label}'
                if small_data is None:
                    print(f'Group {label} has no values and cannot be plotted')
                    continue

                max_x.append(np.nanmax(small_data['previous_activity_count']))
                upper, trace, lower, _ = self._plot_line(df = small_data, x_col = 'previous_activity_count', name = label, marker_col = col)
                fig.add_trace(upper)
                fig.add_trace(trace) 
                fig.add_trace(lower)
                
        fig['layout']['xaxis']['range'] = [1, np.nanmax(max_x)]

        return fig

    def plot_response_quantify(self, response_col = 'has_responded', facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False): 
        """ A plotting function for AGO or mAGO datasets that have been loaded with the analysing function stimulus_response.
        A augmented version of plot_quanitfy that looks for true and false (spontaneous movement) interactions.
        
        Params:
        @response_col = string, the name of the column in the data with the response per interaction, column data should be in boolean form
        @facet_col = string, the name of the column in the metadata you wish to filter the data by
        @facet_arg = list, if not None then a list of items from the column given in facet_col that you wish to be plotted
        @facet_labels = list, if not None then a list of label names for facet_arg. If not provided then facet_arg items are used
        @title = string, a title for the plotted figure
        @grids = bool, true/false whether the resulting figure should have grids

        returns a plotly figure object and a pandas Dataframe with the plotted data
        """

        if response_col not in self.columns.tolist():
            raise KeyError(f'The column you gave {response_col}, is not in the data. Check you have analyed the dataset with stimlus_response')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)

        fig = go.Figure() 
        y_range, dtick = self._check_boolean(list(self[response_col]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Resonse Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = '')

        stats_dict = {}

        if len(set(self.has_interacted)) == 1:
            loop_itr = list(set(self.has_interacted))
        else:
            loop_itr = [2, 1]

        for data, name, col in zip(d_list, facet_labels, col_list):
            
            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            for q in loop_itr:

                if q == 1:
                    qcol = col
                    lab = name
                elif q == 2:
                    qcol = 'grey'
                    lab = f'{name} Spon. Mov'

                filtered = data[data['has_interacted'] == q]

                if len(filtered) == 0:
                    print(f'Group {lab} has no values and cannot be plotted')
                    continue

                filtered = filtered.dropna(subset = [response_col])
                gdf = filtered.analyse_column(column = response_col, function = 'mean')
                median, q3, q1, zlist = self._zscore_bootstrap(gdf[f'{response_col}_mean'].to_numpy())


                stats_dict[lab] = zlist

                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [lab], colour =  qcol, showlegend = False, name = lab, xaxis = 'x'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = qcol, 
                showlegend = False, name = lab, xaxis = 'x'))

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

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
        return type(self)(ds.groupby('id', group_keys = False).apply(find_feed).set_index('id'), ds_meta)


    def remove_sleep_deprived(self, start_time, end_time, remove = False, sleep_column = 'asleep', t_column = 't'):
        """ Removes specimens that during a period of sleep deprivation are asleep a certain percentage of the period
        Params:
        @start_time = int, the time in hours that the period of sleep deprivation begins
        @end_time = int, the time in hours that the period of sleep deprivation ends
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
            return gb.remove('id', remove_ids)

    @staticmethod
    def _time_alive(df, name, repeat = False):
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

        if repeat is False:
            return wrapped_time_alive(df, name)
        else:
            tmp = pd.DataFrame()
            for rep in set(df.meta[repeat].tolist()):
                tdf = df.xmv(repeat, rep)
                tmp = pd.concat([tmp, _wrapped_time_alive(tdf, name)])
            return tmp

    def survival_plot(self, facet_col = None, facet_arg = None, facet_labels = None, repeat = False, title = ''):
        """
        Currently only returns a data frame that can be used with seaborn to plot whilst we go through the changes

        Args:
            facet_col (str, optional): The name of the column to use for faceting. Can be main column or from metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. Default is None.
            facet_labels (list, optional): The labels to use for faceting. Default is None.
            repeat (bool/str, optional): If False the function won't look for a repeat column. If wanted the user should change the argument to the column in the metadata that contains repeat information. Default is False
            title (str, optional): The title of the plot. Default is an empty string.
        
        Returns:
            A Pandas DataFrame with columns hour, survived, and label. It is formatted to fit a Seaborn plot

        """

        if repeat is True:
            if repeat not in self.meta.columns:
                raise KeyError(f'Column "{facet_tile}" is not a metadata column, please check and add if you want repeat data')


        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        sur = pd.DataFrame()
        for data, name in zip(d_list, facet_labels):
            sur = pd.concat([sur, self._time_alive(data, name, repeat = repeat)])

        return sur

    def make_tile(self, facet_tile, plot_fun, rows = None, cols = None):
        """ A wrapper to take any behavpy plot and create a tile plot
        Params:
        @facet_tile = string, the name of column in the metadata you can to split the tile plot by
        @plot_fun = partial function, the plotting method you want per tile with its arguments in the format of partial function. See tutorial.
        @rows = int, the number of rows you would like. Note, if left as the default none the number of rows will be the lengh of faceted variables
        @cols = int, the number of cols you would like. Note, if left as the default none the number of columns will be 1
        **Make sure the rows and cols fit the total number of plots your facet should create.**
        
        returns a plotly subplot figure
        """

        if facet_tile not in self.meta.columns:
            raise KeyError(f'Column "{facet_tile}" is not a metadata column')

        # find the unique column variables and use to split df into tiled parts
        tile_list = list(set(self.meta[facet_tile].tolist()))

        tile_df = [self.xmv(facet_tile, tile) for tile in tile_list]

        if rows is None:
            nrows = len(tile_list)
        if cols is None:
            ncols = 1

        # get a list for col number and rows 
        col_list = list(range(1, ncols+1)) * nrows
        row_list = list([i] * ncols for i in range(1, nrows+1))
        row_list = [item for sublist in row_list for item in sublist]

        # genertate a subplot figure with a single column
        fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes = True, subplot_titles = tile_list)

        layouts = []

        #iterate through the tile df's
        for d, c, r in zip(tile_df, col_list, row_list):
                # take the arguemnts for the plotting function and set them on the function with the small df
                fig_output = getattr(d, plot_fun.func.__name__)(**plot_fun.keywords)
                # if a quantify plot it comes out as a tuple (fig, stats), drop the stats
                if isinstance(fig_output, tuple):
                    fig_output = fig_output[0]
                # add the traces to the plot and put the layout settings into a list
                for f in range(len(fig_output['data'])):
                    fig.add_trace(fig_output['data'][f], col = c, row = r)
                layouts.append(fig_output['layout'])

        # set the background white and put the legend to the side
        fig.update_layout({'legend': {'bgcolor': 'rgba(201, 201, 201, 1)', 'bordercolor': 'grey', 'font': {'size': 12}, 'x': 1.01, 'y': 0.5}, 'plot_bgcolor': 'white'})
        # set the layout on all the different axises
        end_index = len(layouts) - 1
        for c, lay in enumerate(layouts):
            yaxis_title = lay['yaxis'].pop('title')
            xaxis_title = lay['xaxis'].pop('title')
            lay['yaxis']['tickfont'].pop('size')
            lay['xaxis']['tickfont'].pop('size')
            
            fig['layout'][f'yaxis{c+1}'].update(lay['yaxis'])
            fig['layout'][f'xaxis{c+1}'].update(lay['xaxis'])

        # add x and y axis titles
        fig.add_annotation(
            font = {'size': 18, 'color' : 'black'},
            showarrow = False,
            text = yaxis_title['text'],
            x = 0.01,
            xanchor = 'left',
            xref = 'paper',
            y = 0.5,
            yanchor = 'middle',
            yref = 'paper',
            xshift =  -85,
            textangle =  -90
        )
        fig.add_annotation(
            font = {'size': 18, 'color' : 'black'},
            showarrow = False,
            text = xaxis_title['text'],
            x = 0.5,
            xanchor = 'center',
            xref = 'paper',
            y = 0,
            yanchor = 'top',
            yref = 'paper',
            yshift = -30
        )
        return fig
    
    def plot_habituation(self, plot_type, bin_time = 1, num_dtick = 10, response_col =  'has_responded', int_id_col = 'has_interacted', facet_col = None, facet_arg = None, facet_labels = None, secondary = True, title = '', t_column = 't', grids = False):
        """
        A plotting function for AGO or mAGO datasets that have been loaded with the analysing function stimulus_response. 
        A plot to view the response rate to a puff of odour over either the hours (as binned) post the first puff or the consecutive puffs post the first puff.
        This plot is mostly used to understand if the specimen is becoming habituated to the stimulus, it is agnostic of the time of day of the puff or the activity of the specimen.

        Args:
            plot_type (str): Must be either 'time' or 'number'. If time then a plot of the response rate post first puff, if number then the response rate per puff as numbered post first puff.
            bin_time (int, optional): Only needed if plot_type is 'time'. The number of hours you want to bin the response rate to, default is 1 (hour).
            num_dtick (int, optional): The dtick for the x-axis (the number spacing) for when plot_type 'number is chosen. Default is 10.
            response_col (str, optional): The name of the column that contains the boolean response data.
            int_id_col (str, optional): The name of the column conataining the id for the interaction type, which should be either 1 (true interaction) or 2 (false interaction). Default 'has_interacted'.
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
            sceondary (bool, optional): If true then a secondary y-axis is added that contains either the puff cound for 'time' or percentage of flies recieving the puff in 'number'. Default is True
            title (str, optional): The title of the plot. Default is an empty string.
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False
        
        Returns:
            returns a plotly figure objec
        """

        plot_types = ['time', 'number']
        if plot_type not in plot_types:
            raise KeyError(f'plot_type argument must be one of {*plot_types,}')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        if plot_type == 'time':
            yname = 'Puff Count'
            xname = 'Hours post first puff'
            filtname = 'bin_time'
        else:
            yname = 'Percentage revieving puff'
            xname = 'Puff number post first puff'
            filtname = 'puff_count'

        if secondary is False:
            fig = go.Figure() 
        else:
            fig = make_subplots(specs=[[{ "secondary_y" : True}]])
            self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = False, ylabel = yname, title = title, secondary = True, xdomain = 'x1', grid = grids)

        y_range, dtick = self._check_boolean(list(self[response_col]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Response Rate', title = title, secondary = False, grid = grids)

        col_list = self._get_colours(d_list)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = 1, xlabel = xname)

        def get_response(data, ptype, time_window_length):
            if ptype == 'time':
                time_window_length = time_window_length * 60 * 60
                data['bin_time'] = data[t_column].map(lambda t: time_window_length * floor(t / time_window_length)) 
                min_hour = data['bin_time'].min()
                data['bin_time'] = (data['bin_time'] - min_hour) / time_window_length
                gb = data.groupby('bin_time').agg(**{
                            'has_responded' : (response_col, 'mean'),
                            'puff_count' : (response_col, 'count')
                })
                return gb
            elif ptype == 'number':
                tdf = data.sort_values(t_column)
                tdf['puff_count'] = list(range(1, len(tdf)+1))
                return tdf[['puff_count', 'has_responded']]

        max_x = []

        for data, name, col in zip(d_list, facet_labels, col_list):

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue
            
            if len(set(data[int_id_col])) == 1:
                loop_itr = list(set(data[int_id_col]))
            else:
                loop_itr = [2, 1]

            for q in loop_itr:

                if q == 1:
                    qcol = col
                    lab = name
                elif q == 2:
                    qcol = 'grey'
                    lab = f'{name} Spon. Mov'

                tdf = data[data[int_id_col] == q].reset_index()

                if plot_type == 'time':
                    rdf = tdf.groupby('id', group_keys = False).apply(partial(get_response, ptype = plot_type, time_window_length = bin_time))
                    filt_gb = rdf.groupby(filtname).agg(**{
                            'mean' : (response_col, 'mean'),
                            'count' : ('puff_count', 'sum'),
                            'ci' : (response_col, bootstrap)
                    })
                elif plot_type == 'number':
                    rdf = tdf.groupby('id', group_keys = False).apply(partial(get_response, ptype = plot_type, time_window_length = bin_time))
                    filt_gb = rdf.groupby(filtname).agg(**{
                            'mean' : (response_col, 'mean'),
                            'count' : (response_col, 'count'),
                            'ci' : (response_col, bootstrap)
                    })

                max_x.append(np.nanmax(filt_gb.index))

                filt_gb[['y_max', 'y_min']] = pd.DataFrame(filt_gb['ci'].tolist(), index =  filt_gb.index)
                filt_gb.drop('ci', axis = 1, inplace = True)
                filt_gb.reset_index(inplace = True)
                upper, trace, lower, _ = self._plot_line(df = filt_gb, x_col = filtname, name = lab, marker_col = qcol)
                fig.add_trace(upper)
                fig.add_trace(trace) 
                fig.add_trace(lower)

                if secondary is True:
                    if plot_type == 'number':
                        filt_gb['count'] = (filt_gb['count'] / np.max(filt_gb['count'])) * 100

                    fig.add_trace(
                    go.Scatter(
                        legendgroup = lab,
                        x = filt_gb[filtname],
                        y = filt_gb['count'],
                        mode = 'lines',
                        name = f'{lab} count',
                        line = dict(
                            dash = 'longdashdot',
                            shape = 'spline',
                            color = qcol
                            ),
                        ),
                    secondary_y = True
                    )

        fig['layout']['xaxis']['range'] = [0, np.nanmax(max_x)]
        if plot_type == 'number':
            fig['layout']['xaxis']['dtick'] = 10

        return fig

    def plot_response_overtime(self, bin_time = 1, wrapped = False, response_col = 'has_responded', int_id_col = 'has_interacted', facet_col = None, facet_arg = None, facet_labels = None, title = '', day_length = 24, lights_off = 12, secondary = True, t_column = 't', grids = False):
        """
        A plotting function for AGO or mAGO datasets that have been loaded with the analysing function stimulus_response. 
        A plot to view the response rate to a puff over the time of day. Interactions will be binned to a users input (default is 1 hour) and plotted over a ZT hours x-axis. The plot can be the full length of an experiment or wrapped to a singular day.

        Args:
            bin_time (int, optional): The number of hours you want to bin the response rate to, default is 1 (hour).
            wrapped (bool, optional): If true the data is augmented to represent one day, combining data of the same time on consequtive days.
            num_dtick (int, optional): The dtick for the x-axis (the number spacing) for when plot_type 'number is chosen. Default is 10.
            response_col (str, optional): The name of the column that contains the boolean response data.
            int_id_col (str, optional): The name of the column conataining the id for the interaction type, which should be either 1 (true interaction) or 2 (false interaction). Default 'has_interacted'.
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
            title (str, optional): The title of the plot. Default is an empty string.
            day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
            sceondary (bool, optional): If true then a secondary y-axis is added that contains either the puff cound for 'time' or percentage of flies recieving the puff in 'number'. Default is True
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False
        
        Returns:
            returns a plotly figure object
        """

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        fig = make_subplots(specs=[[{ "secondary_y" : True}]])

        max_var = []
        y_range, dtick = self._check_boolean(list(self[response_col]))
        if y_range is not False:
            max_var.append(1)

        if secondary is False:
            fig = go.Figure() 
        else:
            fig = make_subplots(specs=[[{ "secondary_y" : True}]])
            self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = False, ylabel = 'Puff Count', title = title, secondary = True, xdomain = 'x1', grid = grids)

        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Response Rate', title = title, secondary = False, grid = grids)

        col_list = self._get_colours(d_list)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = day_length/4, xlabel = 'ZT (Hours)')

        def get_hourly_response(data, time_window_length):
            data['bin_time'] = data[t_column].map(lambda t: time_window_length * floor(t / time_window_length)) 
            gb = data.groupby(['bin_time', 'has_interacted']).agg(**{
                        'response_rate' : (response_col, 'mean'),
                        'puff_count' : (response_col, 'count')

            })
            return gb

        max_x = []
        min_t = []
        max_t = []

        for data, name, col in zip(d_list, facet_labels, col_list):

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            if wrapped is True:
                data[t_column] = data[t_column] % (60*60*day_length)
            data[t_column] = data[t_column] / (60*60)
                
            min_t.append(int(lights_off * floor(data[t_column].min() / lights_off)))
            max_t.append(int(12 * ceil(data[t_column].max() / 12)) )

            if len(list(set(data.has_interacted))) == 1:
                loop_itr = list(set(data.has_interacted))
            else:
                loop_itr = [2, 1]

            for q in loop_itr:

                if q == 1:
                    qcol = col
                    lab = name
                elif q == 2:
                    qcol = 'grey'
                    lab = f'{name} Spon. Mov'
                
                tdf = data[data[int_id_col] == q].reset_index()
                rdf = tdf.groupby('id', group_keys = False).apply(partial(get_hourly_response, time_window_length = bin_time))

                filt_gb = rdf.groupby('bin_time').agg(**{
                            'mean' : ('response_rate', 'mean'),
                            'count' : ('puff_count', 'sum'),
                            'ci' : ('response_rate', bootstrap)
                })
                filt_gb[['y_max', 'y_min']] = pd.DataFrame(filt_gb['ci'].tolist(), index =  filt_gb.index)
                filt_gb.drop('ci', axis = 1, inplace = True)
                filt_gb.reset_index(inplace = True)

                max_x.append(np.nanmax(filt_gb['bin_time']))

                upper, trace, lower, _ = self._plot_line(df = filt_gb, x_col = 'bin_time', name = lab, marker_col = qcol)
                fig.add_trace(upper)
                fig.add_trace(trace) 
                fig.add_trace(lower)

                if secondary is True:
                    fig.add_trace(
                    go.Scatter(
                        legendgroup = lab,
                        x = filt_gb['bin_time'],
                        y = filt_gb['count'],
                        mode = 'lines',
                        name = f'{lab} count',
                        line = dict(
                            dash = 'longdashdot',
                            shape = 'spline',
                            color = qcol
                            ),
                        ),
                    secondary_y = True
                    )
        # Light-Dark annotaion bars
        bar_shapes, min_bar = circadian_bars(np.nanmin(min_t), np.nanmax(max_t), max_y = np.nanmax(max_var), day_length = day_length, lights_off = lights_off)
        fig.update_layout(shapes=list(bar_shapes.values()))
        fig['layout']['xaxis']['range'] = [1, np.nanmax(max_t)]

        return fig
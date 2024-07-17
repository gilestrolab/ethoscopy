import pandas as pd
import numpy as np 

import plotly.graph_objs as go 
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative

from math import floor, ceil
from scipy.stats import zscore
from functools import partial, update_wrapper
from colour import Color

from ethoscopy.behavpy_draw import behavpy_draw

from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.analyse import max_velocity_detector
from ethoscopy.misc.rle import rle
from ethoscopy.misc.bootstrap_CI import bootstrap
from ethoscopy.misc.hmm_functions import hmm_pct_transition, hmm_mean_length, hmm_pct_state

class behavpy_plotly(behavpy_draw):
    """
    plotly wrapper around behavpy_core
    """

    canvas = 'plotly'

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
    def _plot_meanbox(mean, median, q3, q1, x, colour, showlegend, name, xaxis, CI = True):
        """ For quantify plots, creates a box with a mean line and then extensions showing the confidence intervals 
        """

        trace_box = go.Box(
            showlegend = showlegend,
            median = median,
            mean = mean,
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
        return upper_bound, trace, lower_bound

    def heatmap(self, variable = 'moving', t_column = 't', title = ''):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals using plotly
        
            Args:
                variable (str, optional): The name for the column containing the variable of interest. Default is moving
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                title (str, optional): The title of the plot. Default is an empty string.

        returns
            A plotly heatmap object
        """

        data, time_list, id = self.heatmap_dataset(variable, t_column)

        fig = go.Figure(data=go.Heatmap(
                        z = data,
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

    def plot_overtime(self, variable:str, wrapped:bool = False, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, avg_window:int = 180, day_length:int = 24, lights_off:int = 12, title:str = '', grids:bool = False, t_column:str = 't', col_list = None):
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

        
        Returns:
            returns a plotly figure object

        Note:
            For accurate results, the data should be appropriately preprocessed to ensure that 't' values are
            in the correct format (seconds from time 0) and that 'variable' exists in the DataFrame.
        """
        assert isinstance(wrapped, bool)
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]

        if col_list is None:
            col_list = self._get_colours(d_list)

        fig = go.Figure() 

        min_t = []
        max_t = []

        for data, name, col in zip(d_list, facet_labels, col_list):
            upper, trace, lower, t_min, t_max = self._generate_overtime_plot(data = data, name = name, col = col, var = variable, 
                                                                                    avg_win = avg_window, wrap = wrapped, day_len = day_length, 
                                                                                    light_off= lights_off, t_col = t_column, canvas = 'plotly')
            if upper is None:
                continue

            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

            min_t.append(t_min)
            max_t.append(t_max)

        y_mins = []
        y_maxs = []
        for trace_data in fig.data:
            y_mins.append(min(trace_data.y))
            y_maxs.append(max(trace_data.y))
        ymin = np.nanmin(y_mins) * 0.95
        ymax = np.nanmax(y_maxs) * 1.05

        y_range, dtick = self._check_boolean(list(self[variable]))
        if y_range is not False:
            ymin, ymax = 0, 1.01

        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = day_length/4, xlabel = 'ZT (Hours)')

        # Light-Dark annotaion bars
        bar_shapes, min_bar = circadian_bars(np.nanmin(min_t), np.nanmax(max_t), min_y = ymin, max_y = ymax, day_length = day_length, lights_off = lights_off)
        fig.update_layout(shapes=list(bar_shapes.values()))

        fig['layout']['xaxis']['range'] = [np.nanmin(min_t), np.nanmax(max_t)]
        fig['layout']['yaxis']['range'] = [min_bar, ymax]

        return fig

    def plot_quantify(self, variable:str, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, fun:str = 'mean', title:str = '', z_score = True, grids:bool = False):
        """
        A plot that finds the average (default mean) for a given variable per specimen. The plots will show each specimens average 
        and a box representing the mean and 95% confidence intervals.
        Addtionally, a pandas dataframe is generated that contains the averages per specimen per group for users to perform statistics with.

        Args:
            variable (str): The name of the column you wish to plot from your data. 
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels 
                will be those from the metadata. Default is None.
            fun (str, optional): The function that is applied to the data. Must be one of 'mean', 'median', 'max', 'count'.
            title (str, optional): The title of the plot. Default is an empty string.
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False

        Returns:
            returns a plotly figure object and a pandas DataFrame

        Note:
            Whilst this uses the boxplot functions from plotly, it uses it for formatting and actually plots the mean (dotted line)
            and its 95% confidence intervals, as well as the median (solid line).
        """

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]

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
            mean, median, q3, q1, zlist = self._zscore_bootstrap(gdf[f'{variable}_{fun}'].to_numpy(), z_score = z_score)
            stats_dict[name] = zlist

            fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
            x = [name], colour =  col, showlegend = True, name = name, xaxis = 'x'))

            fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [name], colour = col, 
            showlegend = False, name = name, xaxis = 'x'))

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        if fun != 'mean':
            fig['layout']['yaxis']['autorange'] = True

        return fig, stats_df

    def plot_compare_variables(self, variables, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', z_score:bool = True, grids = False):
        """the first variable in the list is the left hand axis, the last is the right hand axis"""

        assert(isinstance(variables, list))

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)
        
        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]

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

                t_gb = data.analyse_column(column = var, function = fun)
                mean, median, q3, q1, zlist = self._zscore_bootstrap(t_gb[f'{var}_{fun}'].to_numpy(), z_score = z_score)
                stats_dict[f'{name}_{var}'] = zlist

                if len(facet_arg) == 1:
                    col_index = c2
                else:
                    col_index = c

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
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

    def plot_day_night(self, variable, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', day_length = 24, lights_off = 12, title = '', t_column = 't', z_score = True, grids = False):
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

        # if facet_col is not None:
        #     d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        # else:
        #     d_list = [self.copy(deep = True)]

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)
        
        # add the phases for day and night
        data.add_day_phase(day_length = day_length, lights_off = lights_off, t_column = t_column)
        data = data.dropna(subset=[variable])

        if facet_col:
            # merge the facet_col column and replace with the labels
            data = self.facet_merge(data, facet_col, facet_arg, facet_labels)

        fig = go.Figure()
        y_range, dtick = self._check_boolean(list(self[variable]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)

        stats_dict = {}

        for c, phase in enumerate(['light', 'dark']):

            d1 = data[data['phase'] == phase]

            for c2, label in enumerate(facet_labels):
                # print(d1[d1[facet_col]=='1'])
                if facet_col:
                    d2 = d1[d1[facet_col] == label]
                    if len(d2) == 0:
                        print(f'Group {label} has no values and cannot be plotted')
                        continue
                else:
                    d2 = d1
                
                t_gb = d2.analyse_column(column = variable, function = fun)
                mean, median, q3, q1, zlist = self._zscore_bootstrap(t_gb[f'{variable}_mean'].to_numpy(), z_score = z_score)
                stats_dict[f'{label}_{phase}'] = zlist

                if phase == 'light':
                    col = 'goldenrod'
                else:
                    col = 'black'

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [label], colour =  col, showlegend = True, name = label, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [label], colour = col, 
                showlegend = False, name = label, xaxis = f'x{c+1}'))

                domains = np.arange(0, 2, 1/2)
                axis = f'xaxis{c+1}'
                self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = phase, domains = domains[c:c+2], axis = axis)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_anticipation_score(self, mov_variable = 'moving', facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, lights_off = 12, title = '', z_score = True, grids = False):

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        data = data.dropna(subset=[mov_variable])
        data = data.wrap_time()

        dataset = self.anticipation_score(data, mov_variable, day_length, lights_off).set_index('id')

        if facet_col:
            # merge the facet_col column and replace with the labels
            dataset = self.facet_merge(dataset, facet_col, facet_arg, facet_labels)

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 100], t0 = 0, dtick = 20, ylabel = 'Anticipatory Phase Score', title = title, grid = grids)

        palette = self._get_colours(facet_labels)
        for c, name in enumerate(facet_labels):
            _, palette[c], = self._check_grey(name, palette[c]) # change to grey if control
        
        stats_dict = {}

        for c, phase in enumerate(['Lights On', 'Lights Off']):

            d1 = dataset[dataset['phase'] == phase]

            for c2, label in enumerate(facet_labels):
                
                if facet_col:
                    d2 = d1[d1[facet_col] == label]
                    col_index = c2
                    if len(d2) == 0:
                        print(f'Group {label} has no values and cannot be plotted')
                        continue
                else:
                    d2 = d1
                    col_index = c
                
                if z_score is True:
                    zscore_list = d2['anticipation_score'].to_numpy()[np.abs(zscore(d2['anticipation_score'].to_numpy())) < 3]
                else:
                    zscore_list = d2['anticipation_score'].to_numpy()
                q1, q3 = bootstrap(zscore_list)

                stats_dict[f'{label}_{phase}'] = zscore_list

                fig.add_trace(self._plot_meanbox(mean = [np.mean(zscore_list)], median = [np.median(zscore_list)], q3 = [q3], q1 = [q1], 
                x = [label], colour =  palette[col_index], showlegend = True, name = label, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = zscore_list, x = len(zscore_list) * [label], colour = palette[col_index], 
                showlegend = False, name = label, xaxis = f'x{c+1}'))

            domains = np.arange(0, 2, 1/2)
            axis = f'xaxis{c+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = phase, domains = domains[c:c+2], axis = axis)
        
        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))
        return fig, stats_df

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
        title_list = facet_labels

        if facet_col != None:
            root = self._get_subplots(len(facet_arg))
        else:
            facet_arg = [None]
            root =  self._get_subplots(1)

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
    
    def plot_actogram_tile(self, mov_variable = 'moving', bin_window = 15, t_column = 't', labels = None, day_length = 24, title = ''):
        
        if labels is not None:
            if labels not in self.meta.columns.tolist():
                raise KeyError(f'{labels} is not a column in the metadata')
            title_list = self.meta[labels].tolist() 
        else:
            title_list = self.meta.index.tolist()

        facet_arg = self.meta.index.tolist()
        root =  self._get_subplots(len(facet_arg))
        
        data = self.copy(deep=True)

        # make a square subplot domain
        fig = make_subplots(rows=root, cols=root, shared_xaxes = False, vertical_spacing = 0.8/root, subplot_titles = title_list)
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
        fig.update_layout(height=150*root, width=250*6)

        return fig

    # Response AGO/mAGO section

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

                mean, median, q3, q1, zlist = self._zscore_bootstrap(gdf[f'{response_col}_mean'].to_numpy())
                stats_dict[lab] = zlist

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [lab], colour =  qcol, showlegend = False, name = lab, xaxis = 'x'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = qcol, 
                showlegend = False, name = lab, xaxis = 'x'))

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_habituation(self, plot_type, bin_time = 1, num_dtick = 10, response_col =  'has_responded', int_id_col = 'has_interacted', facet_col = None, facet_arg = None, facet_labels = None, display = 'continuous', secondary = True, title = '', t_column = 't', grids = False):
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
                display (str, optional): Choose between two options to display the data, 'continuous' (default) that is a continuous splined line along the x-axis with 95% CI, 'boxplots' is the same data but boxplots with the mean and 95 CI. Default is 'continuous'.
                secondary (bool, optional): If true then a secondary y-axis is added that contains either the puff cound for 'time' or percentage of flies recieving the puff in 'number'. Default is True
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False
        
        Returns:
            returns a plotly figure object
        """

        plot_types = ['time', 'number']
        if plot_type not in plot_types:
            raise KeyError(f'plot_type argument must be one of {*plot_types,}')

        group_types = ['continuous', 'boxplots']
        if display not in group_types:
            raise KeyError(f'display argument must be one of {*group_types,}')

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
            yname = 'Percentage recieving puff'
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
                            'median' : (response_col, 'median'),
                            'count' : ('puff_count', 'sum'),
                            'ci' : (response_col, bootstrap)
                    })
                elif plot_type == 'number':
                    rdf = tdf.groupby('id', group_keys = False).apply(partial(get_response, ptype = plot_type, time_window_length = bin_time))
                    filt_gb = rdf.groupby(filtname).agg(**{
                            'mean' : (response_col, 'mean'),
                            'median' : (response_col, 'median'),
                            'count' : (response_col, 'count'),
                            'ci' : (response_col, bootstrap)
                    })

                max_x.append(np.nanmax(filt_gb.index))

                filt_gb[['y_max', 'y_min']] = pd.DataFrame(filt_gb['ci'].tolist(), index =  filt_gb.index)
                filt_gb.drop('ci', axis = 1, inplace = True)
                filt_gb.reset_index(inplace = True)

                if display == 'continuous':

                    upper, trace, lower = self._plot_line(df = filt_gb, x_col = filtname, name = lab, marker_col = qcol)
                    fig.add_trace(upper)
                    fig.add_trace(trace) 
                    fig.add_trace(lower)


                elif display == 'boxplots':

                    for c in range(len(filt_gb)):

                        fig.add_trace(self._plot_meanbox(mean = [filt_gb['mean'].iloc[c]], median = [filt_gb['mean'].iloc[c]], q3 = [filt_gb['y_min'].iloc[c]], q1 = [filt_gb['y_max'].iloc[c]], 
                        x = [filt_gb[filtname].iloc[c]], colour =  qcol, showlegend = False, name = filt_gb[filtname].iloc[c].astype(str), xaxis = 'x'))


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
            fig['layout']['xaxis']['range'] = [1, np.nanmax(max_x)]
            if np.nanmax(max_x) > 30:
                fig['layout']['xaxis']['dtick'] = 10

        return fig

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
                col_list = self._get_colours(facet_arg)#[tuple(np.array(eval(col[3:])) / 255) for col in ]
                end_colours, start_colours = self._adjust_colours(col_list)
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

            # puff_df[t_column] = puff_df['interaction_t'] % 86400
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
                upper, trace, lower = self._plot_line(df = small_data, x_col = 'previous_activity_count', name = label, marker_col = col)
                fig.add_trace(upper)
                fig.add_trace(trace) 
                fig.add_trace(lower)
                
        fig['layout']['xaxis']['range'] = [1, np.nanmax(max_x)]

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

                upper, trace, lower = self._plot_line(df = filt_gb, x_col = 'bin_time', name = lab, marker_col = qcol)
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

    # HMM section

    def plot_hmm_overtime(self, hmm, variable = 'moving', labels = None, colours = None, wrapped = False, bin = 60, func = 'max', avg_window = 30, day_length = 24, lights_off = 12, title = '', t_column = 't', grids = False):
        """
        Creates a plot of the liklihood of all states overtune, The y-axis shows the liklihood of being in a HMM state and the x-axis showing time in hours.
        The plot is generated through the plotly package

        Params:
        @hmm = hmmlearn.hmm.MultinomialHMM, this should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
        @variable = string, the column heading of the variable of interest. Default is "moving"
        @labels = list[string], the names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake']
        @colours = list[string], the name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red)
        It accepts a specific colour or an array of numbers that are acceptable to plotly
        @wrapped = bool, if True the plot will be limited to a 24 hour day average
        @bin = int, the time in seconds you want to bin the movement data to, default is 60 or 1 minute
        @func = string, when binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
        @avg_window, int, the window in minutes you want the moving average to be applied to. Default is 30 mins
        @circadian_night, int, the hour when lights are off during the experiment. Default is ZT 12
        @save = bool/string, if not False then save as the location and file name of the save file

        returns A plotly figure
        """
        assert isinstance(wrapped, bool)

        df = self.copy(deep = True)

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)

        states_list, time_list = self._hmm_decode(df, hmm, bin, variable, func, t_column)

        df = pd.DataFrame()
        for l, t in zip(states_list, time_list):
            tdf = hmm_pct_state(l, t, list(range(len(labels))), avg_window = int((avg_window * 60)/bin))
            df = pd.concat([df, tdf], ignore_index = True)

        df.rename(columns = dict(zip([f'state_{c}' for c in range(0,len(labels))], labels)), inplace = True)
        melt_df = df.melt('t')
        m = melt_df[['variable']]
        melt_df = melt_df.rename(columns = {'variable' : 'id'}).set_index('id')
        m['id'] = m[['variable']]
        m = m.set_index('id')

        df = self.__class__(melt_df, m)

        return df.plot_overtime(variable='value', wrapped=wrapped, facet_col='variable', facet_arg=labels, avg_window=avg_window, day_length=day_length, 
                                    lights_off=lights_off, title=title, grids=grids, t_column=t_column, col_list = colours)

    def plot_hmm_split(self, hmm, variable = 'moving', labels = None, colours= None, facet_labels = None, facet_col = None, facet_arg = None, wrapped = False, bin = 60, func = 'max', avg_window = 30, day_length = 24, lights_off = 12, title = '', t_column = 't', grids = False):
        """ works for any number of states """

        assert isinstance(wrapped, bool)

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        start_colours, end_colours = self._adjust_colours(colours)

        if len(labels) <= 2:
            nrows = 1
            ncols =2
        else:
            nrows =  2
            ncols = round(len(labels) / 2)

        fig = make_subplots(rows= nrows, cols= ncols, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02, horizontal_spacing=0.02)

        col_list = list(range(1, ncols+1)) * nrows
        row_list = list([i] * ncols for i in range(1, nrows+1))
        row_list = [item for sublist in row_list for item in sublist]

        colour_range_dict = {}
        colours_dict = {'start' : start_colours, 'end' : end_colours}
        for q in range(0,len(labels)):
            start_color = colours_dict.get('start')[q]
            end_color = colours_dict.get('end')[q]
            N = len(facet_arg)
            colour_range_dict[q] = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]

        for c, (arg, n, h, b) in enumerate(zip(facet_arg, facet_labels, h_list, b_list)):   

            if arg != None:
                d = self.xmv(facet_col, arg)
            else:
                d = self.copy(deep = True)

            states_list, time_list = self._hmm_decode(d, h, b, variable, func, t_column)

            analysed_df = pd.DataFrame()
            for l, t in zip(states_list, time_list):
                temp_df = hmm_pct_state(l, t, [0, 1, 2, 3], avg_window = int((avg_window * 60)/b))
                analysed_df = pd.concat([analysed_df, temp_df], ignore_index = False)

            if wrapped is True:
                analysed_df['t'] = analysed_df['t'].map(lambda t: t % (60*60*day_length))
            analysed_df['t'] = analysed_df['t'] / (60*60)

            if 'control' in n.lower() or 'baseline' in n.lower() or 'ctrl' in n.lower():
                black_list = ['black'] * len(facet_arg)
                black_range_dict = {0 : black_list, 1: black_list, 2 : black_list, 3 : black_list}
                marker_col = black_range_dict
            else:
                marker_col = colour_range_dict

            t_min = int(12 * floor(analysed_df.t.min() / 12))
            t_max = int(12 * ceil(analysed_df.t.max() / 12))    
            t_range = [t_min, t_max]  

            for i, (lab, row, col) in enumerate(zip(labels, row_list, col_list)):

                column = f'state_{i}'

                gb_df = analysed_df.groupby('t').agg(**{
                            'mean' : (column, 'mean'), 
                            'SD' : (column, 'std'),
                            'count' : (column, 'count')
                        })

                gb_df['SE'] = (1.96*gb_df['SD']) / np.sqrt(gb_df['count'])
                gb_df['y_max'] = gb_df['mean'] + gb_df['SE']
                gb_df['y_min'] = gb_df['mean'] - gb_df['SE']
                gb_df = gb_df.reset_index()

                upper, trace, lower = self._plot_line(df = gb_df, x_col = 't', name = n, marker_col = marker_col.get(i)[c])
                fig.add_trace(upper,row=row, col=col)
                fig.add_trace(trace, row=row, col=col) 
                fig.add_trace(lower, row=row, col=col)

                if c == 0:
                    fig.add_annotation(xref='x domain', yref='y domain',x=0.1, y=0.9, text = lab, font = {'size': 18, 'color' : 'black'}, showarrow = False,
                    row=row, col=col)

        fig.update_xaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            range = t_range,
            tick0 = 0,
            dtick = 6,
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 2,
            showgrid = False
        )

        fig.update_yaxes(
            zeroline = False, 
            color = 'black',
            linecolor = 'black',
            range = [-0.05, 1], 
            tick0 = 0,
            dtick = 0.2,
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 4,
            showgrid = grids
        )


        fig.update_layout(
            title = title,
            plot_bgcolor = 'white',
            legend = dict(
                bgcolor = 'rgba(201, 201, 201, 1)',
                bordercolor = 'grey',
                font = dict(
                    size = 14
                ),
                x = 1.005,
                y = 0.5
            )
        )

        fig.add_annotation(
                    font = {'size': 18, 'color' : 'black'},
                    showarrow = False,
                    text = 'ZT Time (Hours)',
                    x = 0.5,
                    xanchor = 'center',
                    xref = 'paper',
                    y = 0,
                    yanchor = 'top',
                    yref = 'paper',
                    yshift = -30
                )
        fig.add_annotation(
                    font = {'size': 18, 'color' : 'black'},
                    showarrow = False,
                    text = 'Likelihood to be in sleep state',
                    x = 0,
                    xanchor = 'left',
                    xref = 'paper',
                    y = 0.5,
                    yanchor = 'middle',
                    yref = 'paper',
                    xshift =  -85,
                    textangle =  -90
        )

        # Light-Dark annotaion bars
        bar_shapes, min_bar = circadian_bars(t_min, t_max, min_y = 0, max_y = 1, day_length = day_length, lights_off = lights_off, split = len(labels))
        fig.update_layout(shapes=list(bar_shapes.values()))

        return fig

    def plot_hmm_quantify(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', t_column = 't', grids = False):
        """
        
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func, t_column) for n, d, h, b in zip(facet_arg, df_list, h_list, b_list)}

        def analysis(array_states):
            rows = []
            for array in array_states:
                unique, counts = np.unique(array, return_counts=True)
                row = dict(zip(unique, counts))
                rows.append(row)
            counts_all =  pd.DataFrame(rows)
            counts_all['sum'] = counts_all.sum(axis=1)
            counts_all = counts_all.iloc[:, list_states[0]: list_states[-1]+1].div(counts_all['sum'], axis=0)
            counts_all.fillna(0, inplace = True)
            return counts_all

        analysed_dict = {f'df{n}' : analysis(decoded_dict[f'df{n}'][0]) for n in facet_arg}
        
        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 1.01], t0 = 0, dtick = 0.2, ylabel = 'Fraction of time in each state', title = title, grid = grids)

        stats_dict = {}

        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):
                
                try:
                    mean, median, q3, q1, zlist = self._zscore_bootstrap(analysed_dict[f'df{arg}'][state].to_numpy())
                except KeyError:
                    mean, median, q3, q1, zlist = [0], [0], [0], [np.nan]
                
                stats_dict[f'{arg}_{lab}'] = zlist

                if 'baseline' in i.lower() or 'control' in i.lower() or 'ctrl' in i.lower():
                    if 'rebound' in i.lower():
                        marker_col = 'black'
                    else:
                        marker_col = 'grey'
                else:
                    if 'rebound' in i.lower():
                        marker_col = f'dark{col}'
                    else:
                        marker_col = col

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [i], colour =  marker_col, showlegend = False, name = i, xaxis = f'x{state+1}'))

                label_list = [i] * len(zlist)
                fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                showlegend = False, name = i, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))
        
        return fig, stats_df
    
    def plot_hmm_quantify_length(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', grids = False):
        
        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func, t_column) for n, d, h, b in zip(facet_arg, df_list, h_list, b_list)}

        def analysis(states, t_diff):
            df_lengths = pd.DataFrame()
            for l in states:
                length = hmm_mean_length(l, delta_t = t_diff) 
                df_lengths = pd.concat([df_lengths, length], ignore_index= True)
            return df_lengths

        analysed_dict = {f'df{n}' : analysis(decoded_dict[f'df{n}'][0], b) for n, b in zip(facet_arg, b_list)}

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = 0.69897000433, ylabel = 'Length of state bout (mins)', ytype = 'log', title = title, grid = grids)

        gb_dict = {f'gb{n}' : analysed_dict[f'df{n}'].groupby('state') for n in facet_arg}

        stats_dict = {}

        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):

                try:
                    mean, median, q3, q1, zlist = self._zscore_bootstrap(gb_dict[f'gb{arg}'].get_group(state)['mean_length'].to_numpy())
                except KeyError:
                    mean, median, q3, q1, zlist = [0], [0], [0], [np.nan]
                stats_dict[f'{arg}_{lab}'] = zlist

                if 'baseline' in i or 'control' in i:
                    if 'rebound' in i:
                        marker_col = 'black'
                    else:
                        marker_col = 'grey'
                else:
                    if 'rebound' in i:
                        marker_col = f'dark{col}'
                    else:
                        marker_col = col

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [i], colour =  marker_col, showlegend = False, name = i, xaxis = f'x{state+1}'))

                label_list = [i] * len(zlist)
                fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                showlegend = False, name = i, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)
        
        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_hmm_quantify_length_min_max(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', grids = False):
            
        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func) for n, d, h, b in zip(facet_arg, df_list, h_list, b_list)}

        def analysis(states, t_diff):
            df_lengths = pd.DataFrame()
            for l in states:
                length = hmm_mean_length(l, delta_t = t_diff, raw = True) 
                df_lengths = pd.concat([df_lengths, length], ignore_index= True)
            return df_lengths

        analysed_dict = {f'df{n}' : analysis(decoded_dict[f'df{n}'][0], b) for n, b in zip(facet_arg, b_list)}

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = 0.69897000433, ylabel = 'Length of state bout (mins)', ytype = 'log', title = title, grid = grids)

        gb_dict = {f'gb{n}' : analysed_dict[f'df{n}'].groupby('state') for n in facet_arg}
        stats = []
        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):
                
                try:
                    lnp = gb_dict[f'gb{arg}'].get_group(state)['length_adjusted'].to_numpy()
                    count = lnp.size
                    mean, median, q3, q1, _ = self._zscore_bootstrap(lnp, min_max = True)

                except KeyError:
                    mean, median, q3, q1, count = [0], [0], [0], [0], [0]

                row_dict = {'group' : i, 'state' : lab, 'mean' : mean, 'median' : median, 'min' : q1, 'max' : q3, 'count' : count}
                stats.append(row_dict)
                if 'baseline' in i or 'control' in i:
                    if 'rebound' in i:
                        marker_col = 'black'
                    else:
                        marker_col = 'grey'
                else:
                    if 'rebound' in i:
                        marker_col = f'dark{col}'
                    else:
                        marker_col = col
                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [i], colour =  marker_col, showlegend = False, name = i, xaxis = f'x{state+1}', CI = False))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)
        
        return fig, pd.DataFrame.from_dict(stats).set_index(['group', 'state']).unstack().stack()
            
    def plot_hmm_quantify_transition(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', grids = False):

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func) for n, d, h, b in zip(facet_arg, df_list, h_list, b_list)}

        def analysis(states):
            df_trans = pd.DataFrame()
            for l in states:
                trans = hmm_pct_transition(l, list_states) 
                df_trans = pd.concat([df_trans, trans], ignore_index= True)
            return df_trans

        analysed_dict = {f'df{n}' : analysis(decoded_dict[f'df{n}'][0]) for n in facet_arg}

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 1.05], t0 = 0, dtick = 0.2, ylabel = 'Fraction of runs of each state', title = title, grid = grids)

        stats_dict = {}

        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):
                
                try:
                    mean, median, q3, q1, zlist = self._zscore_bootstrap(analysed_dict[f'df{arg}'][str(state)].to_numpy())  
                except KeyError:
                    mean, median, q3, q1, zlist = [0], [0], [0], [np.nan]

                stats_dict[f'{arg}_{lab}'] = zlist

                if 'baseline' in i.lower() or 'control' in i.lower():
                    if 'rebound' in i.lower():
                        marker_col = 'black'
                    else:
                        marker_col = 'grey'
                else:
                    if 'rebound' in i.lower():
                        marker_col = f'dark{col}'
                    else:
                        marker_col = col

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [i], colour =  marker_col, showlegend = False, name = i, xaxis = f'x{state+1}'))

                label_list = [i] * len(zlist)
                fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                showlegend = False, name = i, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_hmm_raw(self, hmm, variable = 'moving', colours = None, num_plots = 5, bin = 60, mago_df = None, func = 'max', show_movement = False, title = ''):
        """ plots the raw dedoded hmm model per fly (total = num_plots) 
            If hmm is a list of hmm objects, the number of plots will equal the length of that list. Use this to compare hmm models.
            """

        # Get number of states
        if isinstance(hmm, list):
            states = hmm[0].transmat_.shape[0]
        else: 
            states = hmm.transmat_.shape[0]

        if colours is None:
            if isinstance(hmm, list):
                h = hmm[0]
            else:
                h = hmm

            if states == 4:
                colours = self._colours_four
            else:
                raise RuntimeError(f'Your trained HMM is not 4 states, please provide the {h.transmat_.shape[0]} colours for this hmm. See doc string for more info')

        colours_index = {c : col for c, col in enumerate(colours)}

        if mago_df is not None:
            assert isinstance(mago_df, behavpy), 'The mAGO dataframe is not a behavpy class'

        if isinstance(hmm, list):
            num_plots = len(hmm)
            rand_flies = [np.random.permutation(list(set(self.meta.index)))[0]] * num_plots
            h_list = hmm
            if isinstance(bin, list):
                b_list = bin 
            else:
                b_list = [bin] * num_plots
        else:
            rand_flies = np.random.permutation(list(set(self.meta.index)))[:num_plots]
            h_list = [hmm] * num_plots
            b_list = [bin] * num_plots

        df_list = [self.xmv('id', id) for id in rand_flies]
        decoded = [self._hmm_decode(d, h, b, variable, func) for d, h, b in zip(df_list, h_list, b_list)]

        def analyse(data, df_variable):
            states = data[0][0]
            time = data[1][0]
            id = df_variable.index.tolist()
            var = df_variable[variable].tolist()
            previous_states = np.array(states[:-1], dtype = float)
            previous_states = np.insert(previous_states, 0, np.nan)
            previous_var = np.array(var[:-1], dtype = float)
            previous_var = np.insert(previous_var, 0, np.nan)
            all_df = pd.DataFrame(data = zip(id, states, time, var, previous_states, previous_var))
            all_df.columns = ['id','state', 't', 'var','previous_state', 'previous_var']
            all_df.dropna(inplace = True)
            all_df['colour'] = all_df['previous_state'].map(colours_index)
            all_df.set_index('id', inplace = True)
            return all_df

        analysed = [analyse(i, v) for i, v in zip(decoded, df_list)]

        fig = make_subplots(
        rows= num_plots, 
        cols=1,
        shared_xaxes=True, 
        shared_yaxes=True, 
        vertical_spacing=0.02,
        horizontal_spacing=0.02
        )

        for c, (df, b) in enumerate(zip(analysed, b_list)):
            id = df.first_valid_index()
            print(f'Plotting: {id}')
            if mago_df is not None:
                df2 = mago_df.xmv('id', id)
                df2 = df2[df2['has_interacted'] == 1]
                df2['t'] = df2['interaction_t'].map(lambda t:  b * floor(t / b))
                df2.reset_index(inplace = True)
                df = pd.merge(df, df2, how = 'outer', on = ['id', 't'])
                df['colour'] = np.where(df['has_responded'] == True, 'purple', df['colour'])
                df['colour'] = np.where(df['has_responded'] == False, 'lime', df['colour'])
                df['t'] = df['t'].map(lambda t: t / (60*60))
            
            else:
                df['t'] = df['t'].map(lambda t: t / (60*60))

            trace1 = go.Scatter(
                showlegend = False,
                y = df['previous_state'],
                x = df['t'],
                mode = 'markers+lines', 
                marker = dict(
                    color = df['colour'],
                    ),
                line = dict(
                    color = 'black',
                    width = 0.5
                )
                )
            fig.add_trace(trace1, row = c+1, col= 1)

            if show_movement == True:
                df['var'] = np.roll((df['var'] * 2) + 0.5, 1)
                trace2 = go.Scatter(
                    showlegend = False,
                    y = df['var'],
                    x = df['t'],
                    mode = 'lines', 
                    marker = dict(
                        color = 'black',
                        ),
                    line = dict(
                        color = 'black',
                        width = 0.75
                    )
                    )
                fig.add_trace(trace2, row = c+1, col= 1)

        y_range = [-0.2, states-0.8]
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = False, ylabel = '', title = title)

        fig.update_yaxes(
            showgrid = False,
            linecolor = 'black',
            zeroline = False,
            ticks = 'outside',
            range = y_range, 
            tick0 = 0, 
            dtick = 1,
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 4
        )
        fig.update_xaxes(
            showgrid = False,
            color = 'black',
            linecolor = 'black',
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 4
        )

        fig.update_yaxes(
            title = dict(
                text = 'Predicted State',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            row = ceil(num_plots/2),
            col = 1
        )

        fig.update_xaxes(
            title = dict(
                text = 'ZT Hours',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            row = num_plots,
            col = 1
        )

        return fig
    
    def plot_hmm_response(self, mov_df, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, t_bin = 60, facet_labels = None, func = 'max', title = '', grids = False):
        """
        
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
            mov_df_list = [mov_df.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]
            mov_df_list = [mov_df.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func, return_type = 'table') for n, d, h, b in zip(facet_arg, mov_df_list, h_list, b_list)}
        puff_dict = {f'pdf{n}' : d for n, d in zip(facet_arg, df_list)}

        def alter_merge(data, puff):
            puff['bin'] = puff['interaction_t'].map(lambda t:  t_bin * floor(t / t_bin))
            puff.reset_index(inplace = True)

            merged = pd.merge(data, puff, how = 'inner', on = ['id', 'bin'])
            merged['t_check'] = merged.interaction_t + merged.t_rel
            merged['t_check'] = merged['t_check'].map(lambda t:  t_bin * floor(t / t_bin))

            merged['previous_state'] = np.where(merged['t_check'] > merged['bin'], merged['state'], merged['previous_state'])

            interaction_dict = {}
            for i in list(set(merged.has_interacted)):
                filt_merged = merged[merged['has_interacted'] == i]
                big_gb = filt_merged.groupby(['id', 'previous_state']).agg(**{
                            'prop_respond' : ('has_responded', 'mean')
                    })
                interaction_dict[f'int_{int(i)}'] = big_gb.groupby('previous_state')['prop_respond'].apply(np.array)

            return interaction_dict

        analysed_dict = {f'df{n}' : alter_merge(decoded_dict[f'df{n}'], puff_dict[f'pdf{n}']) for n in facet_arg}
        
        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 1.01], t0 = 0, dtick = 0.2, ylabel = 'Response Rate', title = title, grid = grids)

        stats_dict = {}

        for state, col, st_lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):

                for q in [2, 1]:
                    try:
                        mean, median, q3, q1, zlist = self._zscore_bootstrap(analysed_dict[f'df{arg}'][f'int_{q}'][state])
                    except KeyError:
                        continue

                    if q == 2:
                        lab = f'{i} Spon. mov.'
                    else:
                        lab = i

                    stats_dict[f'{i}: {st_lab}'] = zlist

                    if 'baseline' in lab.lower() or 'control' in lab.lower() or 'ctrl' in lab.lower():
                            marker_col = 'black'
                    elif 'spon. mov.' in lab.lower():
                            marker_col = 'grey'
                    else:
                        marker_col = col

                    fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                    x = [lab], colour =  marker_col, showlegend = False, name = lab, xaxis = f'x{state+1}'))

                    label_list = [lab] * len(zlist)
                    fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                    showlegend = False, name = lab, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = st_lab, domains = domains[state:state+2], axis = axis)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))
        
        return fig, stats_df

    def plot_response_hmm_bouts(self, mov_df, hmm, variable = 'moving', labels = None, colours = None, x_limit = 30, t_bin = 60, func = 'max', title = '', grids = False, t_column = 't'):
        """ 
        Generates a plot showing the response rate per time stamp in each HMM bout. Y-axis is between 0-1 and the response rate, the x-axis is the time point
        in each state as per the time the dataset is binned to when decoded.
        This plot is generated through the plotly package.

            Args:
                mov_df (behavpy dataframe): The matching behavpy dataframe containing the movement data from the response experiment
                hmm (hmmlearn.hmm.MultinomialHMM): The accompanying trained hmmlearn model to decode the data.
                variable (str, optional): The name of column that is to be decoded by the HMM. Default is 'moving'.
                labels (list[string], optional): A list of the names of the decoded states, must match the number of states in the given model and colours. 
                    If left as None and the model is 4 states the names will be ['Deep Sleep', 'Light Sleep', 'Quiet Awake', 'Active Awake']. Default is None
                colours (list[string], optional): A list of colours for the decoded states, must match length of labels. If left as None and the 
                    model is 4 states the colours will be ['Dark Blue', 'Light Blue', 'Red', 'Dark Red']. Default is None.
                x_limit (int, optional): A number to limit the x-axis by to remove outliers, i.e. 30 would be 30 minutes or less. Default 30.
                t_bin (int, optional): The time in seconds to bin the time series data to. Default is 60,
                func (str, optional): When binning the time what function to apply the variable column. Default is 'max'.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'

        Returns:
            returns a plotly figure object

        Note:
            This function must be called on a behavpy dataframe that is populated by data loaded in with the stimulus_response
            analysing function.
        """

        labels, colours = df._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))

        data = mov_df.copy(deep=True)
        pdata = self.copy(deep=True)

        decode_df = df._hmm_decode(df, hmm, t_bin, variable, func, t_column, return_type='table')

        # take the states and time per specimen and find the runs of states
        st_gb = decode_df.groupby('id')['state'].apply(np.array)
        time_gb = decode_df.groupby('id')['bin'].apply(np.array)
        all_runs = []
        for m, t, ids in zip(st_gb, time_gb, st_gb.index):
            spec_run = df._find_runs(m, t, ids)
            all_runs.append(spec_run)
        # take the arrays and make a dataframe for merging
        counted_df = pd.concat([pd.DataFrame(specimen) for specimen in all_runs])
        # _find_runs returns the column of interest as 'moving', so changing them for better clarity 
        counted_df.rename(columns = {'moving' : 'state', 'previous_moving' : 'previous_state'}, inplace = True)

        # change the time column to reflect the timing of counted_df
        pdata['t'] = pdata['interaction_t'].map(lambda t:  t_bin * floor(t / t_bin))
        pdata.reset_index(inplace = True)

        # merge the two dataframes on the id and time column and check the response is in the same time bin or the next
        merged = pd.merge(counted_df, pdata, how = 'inner', on = ['id', 't'])
        merged['t_check'] = merged.interaction_t + merged.t_rel
        merged['t_check'] = merged['t_check'].map(lambda t:  t_bin * floor(t / t_bin))
        merged['previous_state'] = np.where(merged['t_check'] > merged['t'], merged['state'], merged['previous_state'])
        merged = merged[merged['previous_activity_count'] <= x_limit]

        # create and style plot
        fig = go.Figure() 
        df._plot_ylayout(fig, yrange = [0, 1], t0 = 0, dtick = 0.2, ylabel = 'Response Rate', title = title, grid = grids)
        df._plot_xlayout(fig, xrange = False, t0 = 0, dtick = t_bin/60, xlabel = f'Consecutive minutes in state')
        
        for state, col, st_lab in zip(list_states, colours, labels):

            loop_df = merged[merged['previous_state'] == state]

            for q in [1, 2]:

                inner_loop = loop_df[loop_df['has_interacted'] == q]

                if q == 2:
                    st_lab = f'{st_lab} spon. mov.'
                    col = 'grey'

                if len(inner_loop) == 0:
                    print(f"No data for {st_lab}")
                    continue

                big_gb = inner_loop.groupby('previous_activity_count').agg(**{
                                        'mean' : ('has_responded', 'mean'),
                                        'count' : ('has_responded', 'count'),
                                        'ci' : ('has_responded', bootstrap)
                            })
                big_gb[['y_max', 'y_min']] = pd.DataFrame(big_gb['ci'].tolist(), index = big_gb.index)
                big_gb.drop('ci', axis = 1, inplace = True)
                big_gb.reset_index(inplace=True)
                big_gb['previous_activity_count'] = big_gb['previous_activity_count'].astype(int)

                upper, trace, lower, _ = self._plot_line(df = big_gb, x_col = 'previous_activity_count', name = st_lab, marker_col = col)
                fig.add_trace(upper)
                fig.add_trace(trace) 
                fig.add_trace(lower)

        return fig

    # Ploty Periodograms

    def plot_periodogram_tile(self, labels = None, find_peaks = False, title = '', grids = False):
        """ Create a tile plot of all the periodograms in a periodogram dataframe"""

        self._validate()

        if labels is not None:
            if labels not in self.meta.columns.tolist():
                raise AttributeError(f'{labels} is not a column in the metadata')
            title_list = self.meta[labels].tolist() 
        else:
            title_list = self.meta.index.tolist()
        
        facet_arg = self.meta.index.tolist()

        data = self.copy(deep = True)

        if find_peaks is True:
            data = data.find_peaks(num_peaks = 2)
        if 'peak' in data.columns.tolist():
            plot_peaks = True

        root =  self._get_subplots(len(data.meta))

        # make a square subplot domain
        fig = make_subplots(rows=root, cols=root, shared_yaxes = True, shared_xaxes = True, vertical_spacing = 0.8/root, subplot_titles = title_list)
        col_list = list(range(1, root+1)) * root
        row_list = list([i] * root for i in range(1, root+1))
        row_list = [item for sublist in row_list for item in sublist]
        
        for arg, col, row in zip(facet_arg, col_list, row_list): 

            d = data.xmv('id', arg)

            fig.append_trace(go.Scatter(
                    showlegend = False,
                    x = d['period'],
                    y = d['power'],
                    mode = 'lines',
                    line = dict(
                    shape = 'spline',
                    color = 'blue'
                ),
                ), row = row, col = col)

            fig.append_trace(go.Scatter(
                    showlegend = False,
                    x = d['period'],
                    y = d['sig_threshold'],
                    mode = 'lines',
                    line = dict(
                    shape = 'spline',
                    color = 'red'
                ),
                ), row = row, col = col)

            if plot_peaks is True:                
                tdf = d[d['peak'] != False]
                fig.append_trace(go.Scatter(
                    showlegend = False,
                    x = tdf['period'],
                    y = tdf['power'],
                    mode = 'markers',
                    marker_symbol = 'x-dot',
                    marker_color = 'gold',
                    marker_size = 8
                ), row = row, col = col)

        tick_6 = np.arange(0,200*6,6)

        fig.update_xaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            tickmode = 'array', 
            tickvals = tick_6,
            ticktext = tick_6,
            range = [int(min(data['period'])), int(max(data['period']))],
            ticks = 'outside',
            showgrid = False
        )

        fig.update_yaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            range = [int(min(data['power'])), int(max(data['power']))],
            ticks = 'outside',
            showgrid = grids,
        )
        
        fig.update_annotations(font_size=8)
        fig['layout']['title'] = title
        fig['layout']['plot_bgcolor'] = 'white'
        fig.update_layout(height=150*root, width=250*6)

        fig.add_annotation(
                    font = {'size': 18, 'color' : 'black'},
                    showarrow = False,
                    text = 'Period Frequency (Hours)',
                    x = 0.5,
                    xanchor = 'center',
                    xref = 'paper',
                    y = 0,
                    yanchor = 'top',
                    yref = 'paper',
                    yshift = -30
                )
        fig.add_annotation(
                    font = {'size': 18, 'color' : 'black'},
                    showarrow = False,
                    text = 'Power',
                    x = 0,
                    xanchor = 'left',
                    xref = 'paper',
                    y = 0.5,
                    yanchor = 'middle',
                    yref = 'paper',
                    xshift =  -85,
                    textangle =  -90
        )

        return fig

    def plot_periodogram(self, facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False):
        """ This function plot the averaged periodograms of the whole dataset or faceted by a metadata column.
        This function should only be used after calling the periodogram function as it needs columns populated
        from the analysis. 
        Periodograms are a good way to quantify through signal analysis the ryhthmicity of your dataset.
        
            Args:
                facet_col (str, optional): The name of the column to use for faceting. Can be main column or from metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. Default is None.
                facet_labels (list, optional): The labels to use for faceting. Default is None.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): True/False to whether the resulting figure should have grids. Default is False.

        Returns:
            A plotly.figure 
        
        """

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)

        variable = 'power'
        max_var = []
        y_range, dtick = self._check_boolean(list(self[variable].dropna()))
        if y_range is False:
            max_var.append(1)
        
        fig = go.Figure() 
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Power', title = title, grid = grids)
        tick_6 = np.arange(0,200*6,6)
        self._plot_xlayout(fig, xrange = [min(self['period']), max(self['period'])], tickvals = tick_6, ticktext = tick_6, xlabel = 'Period Frequency (Hours)')

        for data, name, col in zip(d_list, facet_labels, col_list):

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'grey'

            upper, trace, lower, _, _ = self._generate_overtime_plot(data = data, name = name, col = col, var = 'power', avg_win = False, wrap = False, day_len = False, light_off = False, canvas = 'plotly', t_col = 'period')
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

        return fig
    
    def plot_quantify_periodogram(self, facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False):
        """
        Creates a boxplot and swarmplot of the peaks in circadian rythymn according to a computed periodogram.
        At its core it is just a wrapper of plot_quantify, with some data augmented before being sent to the method.

        Args:
            facet_col (str, optional): The column name used for faceting. Defaults to None.
            facet_arg (list, optional): List of values used for faceting. Defaults to None.
            facet_labels (list, optional): List of labels used for faceting. Defaults to None.
            title (str, optional): Title of the plot. Defaults to ''.
            grids (bool, optional): If True, add a grid to the plot. Defaults to False.
        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.

        Note:
            This function uses seaborn to create a boxplot and swarmplot. It allows to facet the data by a specific column.
            The function to be applied on the data is specified by the `fun` parameter.
        """
        # check it has the right periodogram columns
        self._validate()
        # name for plotting
        power_var = 'period'
        y_label = 'Period (Hours)'
        # find period peaks for plotting
        if 'peak' not in self.columns.tolist():
            self = self.find_peaks(num_peaks = 1)
        # filter by these plot
        self = self[self['peak'] == 1]
        self = self.rename(columns = {power_var : y_label})
    
        return self.plot_quantify(variable = y_label, facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, 
                                    fun='max', title=title, grids=grids)

    def plot_wavelet(self, mov_variable, sampling_rate = 15, scale = 156, wavelet_type = 'morl', t_col = 't', title = '', grids = False):
        """ A formatter and plotter for a wavelet function.
        Wavelet analysis is a windowed fourier transform that yields a two-dimensional plot, both period and time.
        With this you can see how rhythmicity changes in an experimemnt overtime.

            Args:
                mov_variable (str):The name of the column containting the movement data
                sampling_rate (int, optional): The time in minutes the data should be augmented to. Default is 15 minutes
                scale (int optional): The scale facotr, the smaller the scale the more stretched the plot. Default is 156.
                wavelet_type (str, optional): The wavelet family to be used to decompose the sequences. Default is 'morl'.
                t_col (str, optional): The name of the time column in the DataFrame. Default is 't'.
                title (str, optional): The title of the plot. Default is an empty string.

        Returns:
            A plotly figure
        """
        fun, avg_data = self._format_wavelet(mov_variable, sampling_rate, wavelet_type, t_col)

        fig = go.Figure()
        yticks = [1,2,4,6,12,24,36]
        self._plot_ylayout(fig, yrange = np.log2([2,38]), tickvals = np.log2(yticks), ticktext = yticks, title = title, ylabel = 'Period Frequency (Hours)', grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = 12, xlabel = 'ZT (Hours)')

        t, per, power = fun(avg_data, t_col = t_col, var = mov_variable, scale = scale, wavelet_type = wavelet_type)

        trace = go.Contour(
                z=power,
                x=t, 
                y=per,
                contours=dict(
                    start= -3,
                    end= 3,
                    size= 1,
                    # type = 'constraint'
                ),
                colorscale='jet',
                colorbar=dict(nticks=7, ticks='outside',
                        ticklen=5, tickwidth=1,
                        showticklabels=True,
                        tickangle=0, tickfont_size=12)
            )
        fig.add_trace(trace)
        # config = {'staticPlot': True} 
        
        return fig

    # Experimental section

    def make_tile(self, facet_tile, plot_fun, rows = None, cols = None):
        """ 
            *** Warning - this method is experimental and very unpolished *** 
            
        A method to create a tile plot of a repeated but faceted plot.

            Args:
                facet_tile (str): The name of column in the metadata you can to split the tile plot by
                plot_fun (partial function): The plotting method you want per tile with its arguments in the format of partial function. See tutorial.
                rows (int): The number of rows you would like. Note, if left as the default none the number of rows will be the lengh of faceted variables
                cols (int): the number of cols you would like. Note, if left as the default none the number of columns will be 1
                    **Make sure the rows and cols fit the total number of plots your facet should create.**
        
        returns:
            returns a plotly subplot figure
        """

        if facet_tile not in self.meta.columns:
            raise KeyError(f'Column "{facet_tile}" is not a metadata column')

        # find the unique column variables and use to split df into tiled parts
        tile_list = list(set(self.meta[facet_tile].tolist()))

        tile_df = [self.xmv(facet_tile, tile) for tile in tile_list]

        if rows is None:
            rows = len(tile_list)
        if cols is None:
            cols = 1

        # get a list for col number and rows 
        col_list = list(range(1, cols+1)) * rows
        row_list = list([i] * cols for i in range(1, rows+1))
        row_list = [item for sublist in row_list for item in sublist]

        # generate a subplot figure with a single column
        fig = make_subplots(rows=rows, cols=cols, shared_xaxes = True, subplot_titles = tile_list)

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
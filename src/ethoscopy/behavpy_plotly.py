import pandas as pd
import numpy as np 

import plotly.graph_objs as go 
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative

from math import floor, ceil
from scipy.stats import zscore
from functools import partial

from ethoscopy.behavpy_draw import behavpy_draw
from ethoscopy.behavpy_seaborn import behavpy_seaborn

from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.misc.hmm_functions import hmm_pct_state

class behavpy_plotly(behavpy_draw):
    """
    Extends behavpy_draw to make use of data analysis for plotting with Plotly

    Both behavpy_plotly and behavpy_seaborn are two mirrors of each other, containing the same methods (except make_tile).
    However, there are some differences in the way the data is processed and plotted, which is outlined at the 
    appropriate methods. One consistent difference is that quantification plots with plotly give the option to filter
    the data by a z-score threshold which is not available in seaborn. Addtionally 95% CI are calculated using an
    internal function, whereas seaborn uses its own version.
    
    The behavpy_plotly class is not called directly by the user, however it is called in the generator class behavpy, and as
    such this class is directly accessible to the user.

    Attributes:
        canvas (str): tells the inherited classes that this is a plotly class

    **Example Usage:**
        df = behavpy(behavior_data, behavior_meta, check = True,canvas = 'plotly')
        fig = df.heatmap(variable = 'asleep')
    """

    canvas = 'plotly'

    @staticmethod
    def _plot_ylayout(fig, yrange, ylabel, title, t0 = False, dtick = False, secondary = False, xdomain = False, tickvals = False, ticktext = False, 
                      ytype = "-", grid = False):
        """ Static function to configure the y-axis layout for a Plotly figure """
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
    def _plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = False, domains = False, axis = False, tickvals = False, 
                      ticktext = False, type = "-"):
        """ Static function to configure the x-axis layout for a Plotly figure """
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
        """ Static function to create a box with mean and confidence intervals for quantification plots """

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
        """ Static function to plot data points as dots over the mean box in quantification plots """
        trace_box = go.Box(
            showlegend = showlegend,
            y = y, 
            x = x,
            xaxis = xaxis,
            line = dict(
                color = 'rgba(0,0,0,0)' # make the box transparent
            ),
            fillcolor = 'rgba(0,0,0,0)', # make the box transparent
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
        """ Static function to create traces for mean line with 95% confidence intervals in Plotly """

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

    def heatmap(self, variable:str, t_column:str = 't', title = '', lights_off:int|float = 12):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals. 
        This is a great tool to quickly identify changes in the variable of choice over the course of the experiment.
        
        Args:
            variable (str): The name for the column containing the variable of interest
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            title (str, optional): The title of the plot. Default is an empty string.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. 
                Must be number between 0 and day_lenght. Default is 12.

        returns
            fig (plotly.figure.Figure): Figure object of the plot.

        Note:
            For accurate results, the data should be appropriately preprocessed to ensure that 't' values are
            in the correct format (seconds from time 0) and that 'variable' exists in the DataFrame.

        **Example Usage:**
        # Create an heatmap plot for the 'asleep' variable
        fig = df.heatmap(
            variable='asleep',
            t_column='timestamp',
            title='Activity Level Over Time',
            lights_off=12,
        )
        
        # Display the plot
        fig.show()
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
                dtick = lights_off,
                ticks = 'outside',
                tickwidth = 2,
                linewidth = 2)
                )

        return fig

    def plot_overtime(self, variable:str, wrapped:bool = False, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, 
                      avg_window:int = 30, day_length:int = 24, lights_off:int = 12, title:str = '', grids:bool = False, t_column:str = 't', 
                      col_list = None):
        """
        Generate a line plot to visualize the progression of a specified variable over the duration of an experiment or within an experimental day. 
        This plot displays the mean values along with 95% confidence intervals for each group, providing clear insights into temporal trends and 
        group comparisons.
        Additionally, the plot includes visual indicators (white and black boxes) to denote periods when lights are on and off, which can be customized 
        to reflect varying day lengths or different lighting schedules.
        
        Args:
            variable (str): The name of the column you wish to plot from your data.
            wrapped (bool, optional): If `True`, the data is augmented to represent one day by combining data from consecutive days at the same 
                time points. Defaults to `False`.
            facet_col (str, optional): The name of the column to use for faceting, which must exist in the metadata. Defaults to `None`.
            facet_arg (list, optional): Specific arguments to use for faceting. If `None`, all distinct groups are used. Defaults to `None`.
            facet_labels (list, optional): Custom labels for the facets. If `None`, labels from the metadata are used. Defaults to `None`.
            avg_window (int, optional): The number of minutes applied to the rolling smoothing function. Defaults to `30`.
            day_length (int | float, optional): The length of the experimental day in hours. Defaults to `24`.
            lights_off (int | float, optional): The time point when lights are turned off in an experimental day (0 represents lights on). 
                Must be between `0` and `day_length`. Defaults to `12`.
            title (str, optional): The title of the plot. Defaults to an empty string.
            grids (bool, optional): Whether to display grid lines on the plot. Defaults to `False`.
            t_column (str, optional): The name of the column containing timing data in seconds. Defaults to `'t'`.
            col_list (list | None, optional): A list of colors to override the default palette. Must match the number of facets. Defaults to `None`.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.

        **Example Usage:**
        # Create an overtime plot for the 'activity_level' variable
        fig = df.plot_overtime(
            variable='activity_level',
            wrapped=True,
            facet_col='treatment_group',
            facet_arg=['control', 'treated'],
            facet_labels=['Control Group', 'Treated Group'],
            avg_window=15,
            day_length=24,
            lights_off=12,
            title='Activity Level Over Time',
            grids=True,
            t_column='timestamp',
            col_list=['#1f77b4', '#ff7f0e'],
            figsize=(10, 6)
        );
        
        # Display the plot
        plt.show()
        """
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
            upper, trace, lower, t_min, t_max = self._generate_overtime_plot(
                                                    data = data, name = name, col = col, var = variable, 
                                                    avg_win = int((avg_window * 60)/self[t_column].diff().median()), 
                                                    wrap = wrapped, day_len = day_length, light_off= lights_off, t_col = t_column)
            
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

    def plot_quantify(self, variable:str, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, fun:str = 'mean', 
                      title:str = '', z_score:bool = True, grids:bool = False):
        """
        Generate a quantification plot that calculates the average (default is mean) of a specified variable for each specimen. 
        The plot displays each specimen's average value along with a box representing the mean and the 95% confidence intervals.
        Additionally, a pandas DataFrame is generated containing the grouped averages per specimen and group, to facilitate 
        statistical analyses.

        Args:
            variable (str): The name of the column to plot from the dataset.
            facet_col (str, optional): The name of the column in the metadata used for faceting the plot. Must exist in the metadata.
                Defaults to `None`.
            facet_arg (list, optional): Specific values from `facet_col` to include in the plot. If `None`, all distinct groups 
                from `facet_col` are used. Defaults to `None`.
            facet_labels (list, optional): Custom labels for the facets. If `None`, labels from the metadata corresponding 
                to `facet_arg` are used. Defaults to `None`.
            fun (str, optional): The aggregation function to apply to the data. Must be one of `'mean'`, `'median'`, `'max'`, or `'count'`.
                Defaults to `'mean'`.
            title (str, optional): The title of the generated plot. Defaults to an empty string.
            z_score (bool, optional): If `True`, z-scores are calculated for each entry, and entries with z-scores above 
                or below zero are excluded from the plot. Helps in normalising the data and removing outliers.
                Defaults to `True`.
            grids (bool, optional): Determines whether grid lines are displayed on the plot. Defaults to `False`.


        Returns:
            tuple:
                - fig (matplotlib.figure.Figure): The generated matplotlib figure object of the quantification plot.
                - data (pandas.DataFrame): A DataFrame containing the grouped and aggregated data based on the input parameters.

        Raises:
            ValueError: 
                If `fun` is not one of the accepted aggregation functions (`'mean'`, `'median'`, `'max'`, `'count'`), a `ValueError` is raised 
                to enforce the validity of the aggregation operation.
            KeyError: 
                If `facet_col` is specified but does not exist in the metadata, a `KeyError` is raised to indicate the missing column.

        Notes:
            - This function leverages Seaborn's `stripplot` and `pointplot` to create the visualization.
            - Bootstrapping with `n=1000` is used to calculate the 95% confidence intervals for the aggregated data.


        **Example Usage:**
        # Generate a quantification plot for the 'activity_level' variable
        fig, summarised_data = df.plot_quantify(
            variable='activity_level',
            facet_col='treatment_group',
            facet_arg=['control', 'treated'],
            facet_labels=['Control Group', 'Treated Group'],
            fun='mean',
            title='Average Activity Level per Specimen',
            z_score=True,
            grids=True,
            figsize=(12, 8)
        )
        # Display the plot
        fig.show()            
        """

        grouped_data, palette_dict, facet_labels, variable = self._internal_plot_quantify(variable, facet_col, facet_arg, facet_labels, fun)
        plot_column = f'{variable[0]}_{fun}'

        fig = go.Figure() 

        y_range, dtick = self._check_boolean(list(self[variable[0]]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable[0], title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = '')

        if facet_col:
            for lab in facet_labels:

                sub_df = grouped_data[grouped_data[facet_col] == lab]

                if len(sub_df) == 0:
                    print(f'Group {lab} has no values and cannot be plotted')
                    continue

                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [lab], colour =  palette_dict[lab], showlegend = True, name = lab, xaxis = 'x'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = palette_dict[lab], 
                showlegend = False, name = lab, xaxis = 'x'))
        else:
            lab = ' '
            mean, median, q3, q1, zlist = self._zscore_bootstrap(grouped_data[plot_column].to_numpy(dtype=float), z_score = z_score)

            fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
            x = [lab], colour =  list(palette_dict.values())[0], showlegend = True, name = lab, xaxis = 'x'))

            fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = list(palette_dict.values())[0], 
            showlegend = False, name = lab, xaxis = 'x'))

        if fun != 'mean':
            fig['layout']['yaxis']['autorange'] = True

        return fig, grouped_data

    def plot_compare_variables(self, variables:list, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, 
                               fun:str = 'mean', title:str = '', z_score:bool = True, grids:bool = False):
        """ 
        Create a comparative quantification plot for multiple variables, enabling the visualization of multiple data metrics 
        side-by-side within each facet or group. This method extends the functionality of `plot_quantify`.

        When faceting by a group, each variable is plotted collectively per group rather than individually. This approach 
        allows for direct comparison within groups across different variables, enhancing the interpretability of the data.

        Args:
            variables (list[str]): 
                A list of column names representing the variables to be plotted. Must be in the data.
            facet_col (str, optional): The name of the column in the metadata used for faceting the plot. Must exist in the metadata.
                Defaults to `None`.
            facet_arg (list, optional): Specific values from `facet_col` to include in the plot. If `None`, all distinct groups 
                from `facet_col` are used. Defaults to `None`.
            facet_labels (list, optional): Custom labels for the facets. If `None`, labels from the metadata corresponding 
                to `facet_arg` are used. Defaults to `None`.
            fun (str, optional): The aggregation function to apply to the data. Must be one of `'mean'`, `'median'`, `'max'`, or `'count'`.
                Defaults to `'mean'`.
            title (str, optional): The title of the generated plot. Defaults to an empty string.
            z_score (bool, optional): If `True`, z-scores are calculated for each entry, and entries with z-scores above 
                or below zero are excluded from the plot. Helps in normalising the data and removing outliers.
                Defaults to `True`.
            grids (bool, optional): Determines whether grid lines are displayed on the plot. Defaults to `False`.
        
        Returns:
            tuple:
                - fig (matplotlib.figure.Figure): The generated matplotlib figure object of the quantification plot.
                - data (pandas.DataFrame): A DataFrame containing the grouped and aggregated data based on the input parameters.
        
        Raises:
            TypeError: 
                If `variables` is not provided as a list, a `TypeError` is raised to ensure the method receives the correct input type.
            ValueError: 
                If `fun` is not one of the accepted aggregation functions (`'mean'`, `'median'`, `'max'`, `'count'`), a `ValueError` is raised 
                to enforce the validity of the aggregation operation.
            KeyError: 
                If `facet_col` is specified but does not exist in the metadata, a `KeyError` is raised to indicate the missing column.
        
        Notes:
            - When plotting more than two variables, only the last variable in the list is plotted on the secondary y-axis, while 
              all other variables utilize the primary y-axis. This design choice ensures that the plot remains readable and uncluttered.
        
        **Example Usage:**
        # Define the variables to compare
        variables = ['activity_level', 'asleep', 'distance']
            
        # Generate the comparative quantification plot
        fig, summarized_data = df.plot_compare_variables(
            variables=variables,
            facet_col='treatment_group',
            facet_arg=['control', 'treated'],
            facet_labels=['Control Group', 'Treated Group'],
            fun='mean',
            title='Comparison of Activity Metrics Across Treatment Groups',
            z_score=True,
            grids=True
        )
        # Display the plot
        fig.show()
        """

        if isinstance(variables, list) is False:
            raise TypeError('Argument variables must be given as a list')

        grouped_data, palette_dict, facet_labels, variables = self._internal_plot_quantify(variables, facet_col, facet_arg, facet_labels, fun)
        plot_column_list = [f'{var}_{fun}' for var in variables]

        fig = go.Figure() 
        y_range, dtick = self._check_boolean(list(self[variables[0]]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[0], title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = '')
        fig = make_subplots(specs=[[{ "secondary_y" : True}]])

        # generate a list where only the last item is True to then place it on the secondary axis
        bool_list = len(variables) * [False]
        bool_list[-1] = True
        # domains on the x-axis for plotting
        domains = np.arange(0, 1+(1/len(facet_labels)), 1/len(facet_labels))

        for c, lab in enumerate(facet_labels):

            if facet_col:
                sub_df = grouped_data[grouped_data[facet_col] == lab]
            else:
                sub_df = grouped_data

            if len(sub_df) == 0:
                print(f'Group {lab} has no values and cannot be plotted')
                continue

            for c2, (var, secondary) in enumerate(zip(variables, bool_list)):

                if facet_col is None: lab = var # have variable colour change if no facet

                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column_list[c2]].to_numpy(dtype=float), z_score = z_score)

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [var], colour =  palette_dict[lab], showlegend = True, name = var, xaxis = f'x{c+1}'), secondary_y = secondary)

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [var], colour = palette_dict[lab], 
                showlegend = False, name = var, xaxis = f'x{c+1}'), secondary_y = secondary)

                if facet_col is None: lab = facet_labels[c] # revert back for axis naming, not the most efficient \_O_O_/

            axis = f'xaxis{c+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[c:c+2], axis = axis)

        axis_counter = 1
        for i in range(len(facet_labels) * (len(variables) * 2)):
            if i%((len(variables) * 2)) == 0 and i != 0:
                axis_counter += 1
            fig['data'][i]['xaxis'] = f'x{axis_counter}'

        y_range, dtick = self._check_boolean(list(self[variables[0]]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[0], title = title, secondary = False, grid = grids)

        y_range, dtick = self._check_boolean(list(self[variables[-1]]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[-1], title = title, secondary = True, xdomain = f'x{axis_counter}', grid = grids)

        fig['layout'].update(legend =  dict(x = 1.06, y = 0.5)) # shift the legend so it doesn't overlap

        return fig, grouped_data
        
    def plot_day_night(self, variable:str, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, day_length:int|float = 24, lights_off:int|float = 12, z_score:bool = True, title:str = '', t_column:str = 't', grids:bool = False):
        """
        A plot that shows the average of a variable split between the day (lights on) and night (lights off). This a specific version of plot_quantify, using similar backend.
           
        Args:
            variable (str): The name of the column you wish to plot from your data. 
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
            fun (str, optional): The average function that is applied to the data. Must be one of 'mean', 'median', 'count'.
            day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
            z_score (bool, optional): If True (Default) the z-score for each entry is found the those above/below zero are removed. Default is True.                
            title (str, optional): The title of the plot. Default is an empty string.
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.
        """

        grouped_data, palette_dict, facet_labels = self._internal_plot_day_night(variable, facet_col, facet_arg, facet_labels, day_length, lights_off, t_column)
        plot_column = f'{variable}_mean'
        palette_phase = {'light' : 'goldenrod', 'dark' : 'black'} # for use when no facet for extra style

        fig = go.Figure()
        y_range, dtick = self._check_boolean(list(self[variable]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)
        # domains will always be two, light and dark
        domains = np.arange(0, 2, 1/2)

        for c, phase in enumerate(['light', 'dark']):

            sub_df = grouped_data[grouped_data['phase'] == phase]

            if len(sub_df) == 0:
                print(f'Group {lab} has no values and cannot be plotted')
                continue
            
            if facet_col:
                for lab in facet_labels:

                    sub_sub_df = sub_df[sub_df[facet_col] == lab]

                    mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                    fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                    x = [lab], colour =  palette_dict[lab], showlegend = True, name = lab, xaxis = f'x{c+1}'))

                    fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = palette_dict[lab], 
                    showlegend = False, name = lab, xaxis = f'x{c+1}'))
            else:
                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [phase], colour =  palette_phase[phase], showlegend = True, name = phase, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [phase], colour = palette_phase[phase], 
                showlegend = False, name = phase, xaxis = f'x{c+1}'))

            axis = f'xaxis{c+1}'
            if facet_col is None: phase = ' ' # to prevent double labelling of light dark
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = phase, domains = domains[c:c+2], axis = axis)

        return fig, grouped_data

    def plot_anticipation_score(self, variable:str, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, day_length:int|float = 24, lights_off:int|float = 12, z_score:bool = True, title:str = '', t_column:str = 't', grids:bool = False):
        """
        Plots the anticipation scores for lights on and off periods. The anticipation score is calculated as the percentage of activity of the 6 hours prior to lights on/off that occurs in the last 3 hours.
        A higher score towards 100 indicates greater anticipation of the light change.

        Args:
            variable (str): The name of the column containing the variable that measures activity.
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. 
                If None the labels will be those from the metadata. Default is None.
            day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, 
                assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
            z_score (bool, optional): If True (Default) the z-score for each entry is found the those above/below zero are removed. 
                Default is True.
            title (str, optional): The title of the plot. Default is an empty string.
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.
        """
        grouped_data, palette_dict, facet_labels = self._internal_plot_anticipation_score(variable, facet_col, facet_arg, facet_labels, 
                                                                                            day_length, lights_off, t_column)
        palette_phase = {'Lights On' : 'goldenrod', 'Lights Off' : 'black'} # for use when no facet for extra style

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 100], t0 = 0, dtick = 20, ylabel = 'Anticipatory Phase Score', title = title, grid = grids)
        plot_column = 'anticipation_score'
        # domains will always be two, lights on and off
        domains = np.arange(0, 2, 1/2)

        for c, phase in enumerate(['Lights On', 'Lights Off']):

            sub_df = grouped_data[grouped_data['phase'] == phase]

            if len(sub_df) == 0:
                print(f'Group {lab} has no values and cannot be plotted')
                continue
            
            if facet_col:
                for lab in facet_labels:

                    sub_sub_df = sub_df[sub_df[facet_col] == lab]

                    mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                    fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                    x = [lab], colour =  palette_dict[lab], showlegend = True, name = lab, xaxis = f'x{c+1}'))

                    fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = palette_dict[lab], 
                    showlegend = False, name = lab, xaxis = f'x{c+1}'))
            else:
                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [phase], colour =  palette_phase[phase], showlegend = True, name = phase, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [phase], colour = palette_phase[phase], 
                showlegend = False, name = phase, xaxis = f'x{c+1}'))

            axis = f'xaxis{c+1}'
            if facet_col is None: phase = ' ' # to prevent double labelling of light dark
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = phase, domains = domains[c:c+2], axis = axis)

        return fig, grouped_data

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

    def plot_actogram(self, mov_variable:str = 'moving', bin_window:int = 5, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, day_length:int|float = 24, t_column:str = 't', title:str = ''):
        """
        This function creates actogram plots from the provided data. Actograms are useful for visualizing 
        patterns in activity data (like movement or behavior) over time, often with an emphasis on daily 
        (24-hour) rhythms. 

        Args:
            mov_variable (str, optional): The name of the column in the dataframe representing movement 
                data. Default is 'moving'.
            bin_window (int, optional): The bin size for data aggregation in minutes. Default is 5.
            facet_col (str, optional): The name of the column to be used for faceting. If None, no faceting 
                is applied. Default is None.
            facet_arg (list, optional): List of arguments to be used for faceting. If None and if 
                facet_col is not None, all unique values in the facet_col are used. Default is None.
            facet_labels (list, optional): List of labels to be used for the facets. If None and if 
                facet_col is not None, all unique values in the facet_col are used as labels. Default is None.
            day_length (int, optional): The length of the day in hours. Default is 24.
            t_column (str, optional): The name of the time column in the DataFrame. Default is 't'.
            title (str, optional): The title of the plot. Default is an empty string.

        Returns:
            matplotlib.figure.Figure: If facet_col is provided, returns a figure that contains subplots for each 
            facet. If facet_col is not provided, returns a single actogram plot.

        Note:
            Due to plotlys' interactive nature it doesn't generate great actograms. So now the default plot is a matplotlib plot.

        Example:
            >>> instance.plot_actogram(mov_variable='movement', bin_window=10, 
            ...                        t_column='time', facet_col='activity_type')
        """

        seaborn_df = behavpy_seaborn(data=self, meta=self.meta, palette='deep', long_palette='deep', check=False, index=None, columns=None, dtype=None, copy=True)

        return seaborn_df.plot_actogram(mov_variable=mov_variable, bin_window=bin_window, facet_col=facet_col, facet_arg=facet_arg, day_length=day_length,
                                                    t_column=t_column, title=title)

        # facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)
        # title_list = facet_labels

        # if facet_col != None:
        #     root = self._get_subplots(len(facet_arg))
        # else:
        #     facet_arg = [None]
        #     root =  self._get_subplots(1)

        # # make a square subplot domain
        # fig = make_subplots(rows=root, cols=root, shared_xaxes = False, subplot_titles = title_list)
        # col_list = list(range(1, root+1)) * root
        # row_list = list([i] * root for i in range(1, root+1))
        # row_list = [item for sublist in row_list for item in sublist]

        # data = self.copy(deep=True)
        # data = data.bin_time(mov_variable, bin_window*60, t_column = t_column)
        # data.add_day_phase(t_column = f'{t_column}_bin')

        # for arg, col, row in zip(facet_arg, col_list, row_list): 

        #     if facet_col is not None:
        #         d = data.xmv(facet_col, arg)

        #         if len(d) == 0:
        #             print(f'Group {arg} has no values and cannot be plotted')
        #             continue

        #         d = d.groupby(f'{t_column}_bin').agg(**{
        #             'moving_mean' : ('moving_mean', 'mean'),
        #             'day' : ('day', 'max')
        #         })
        #         d.reset_index(inplace = True)
        #         d[f'{t_column}_bin'] = (d[f'{t_column}_bin'] % (day_length*60*60)) / (60*60)
        #     else:
        #         d = data.wrap_time(24, time_column = f'{t_column}_bin')
        #         d[f'{t_column}_bin'] = d[f'{t_column}_bin'] / (60*60)

        #     self._actogram_plot(fig = fig, data = d, mov = mov_variable, day = day_length, row = row, col = col)

        # fig.update_xaxes(
        #     zeroline = False,
        #     color = 'black',
        #     linecolor = 'black',
        #     gridcolor = 'black',
        #     range = [0,48],
        #     tick0 = 0,
        #     dtick = 6,
        #     ticks = 'outside',
        #     tickfont = dict(
        #         size = 12
        #     ),
        #     showgrid = False
        # )

        # fig.update_yaxes(
        #     zeroline = False,
        #     color = 'black',
        #     linecolor = 'black',
        #     gridcolor = 'black',
        #     range = [0,int(data['day'].max())],
        #     tick0 = 0,
        #     dtick = 1,
        #     ticks = 'outside',
        #     showgrid = True,
        #     autorange =  'reversed'
        # )
        
        # if facet_col == None:
        #     fig.update_annotations(font_size=8)
        # fig['layout']['title'] = title
        # fig['layout']['plot_bgcolor'] = 'white'

        # return fig
    
    def plot_actogram_tile(self, mov_variable:str = 'moving', labels:None|str = None, bin_window:int = 15, day_length:int = 24, t_column:str = 't', title:str = ''):
        """
        This function creates a grid or tile actogram plot of all specimens in the provided data. Actograms are useful for visualizing 
        patterns in activity data (like movement or behavior) over time, often with an emphasis on daily 
        (24-hour) rhythms. 

        Args:
            mov_variable (str, optional): The name of the column in the dataframe representing movement 
                data. Default is 'moving'.
            labels (str, optional): The name of the column in the metadata with the labels per specimen. If None then
                the ids in the index are used. Default is None.
            bin_window (int, optional): The bin size for data aggregation in minutes. Default is 15.
            day_length (int, optional): The length of the day in hours. Default is 24.
            t_column (str, optional): The name of the column in the dataframe representing time data.
                Default is 't'.
            title (str, optional): The title of the plot. Default is an empty string.

        Returns:
            matplotlib.figure.Figure
        Raises:
            KeyError: If labels column is not in the metadata.
        Note:
            Due to plotlys' interactive nature it doesn't generate great actograms. So now the default plot is a matplotlib plot.
        Example:
            >>> instance.plot_actogram_tile(mov_variable='movement', lables='labels_column', 
                                            bin_window=10, t_column='time')
            ...                        
        """  

        seaborn_df = behavpy_seaborn(data=self, meta=self.meta, palette='deep', long_palette='deep', check=False, index=None, columns=None, dtype=None, copy=True)

        return seaborn_df.plot_actogram_tile(mov_variable=mov_variable, labels=labels, bin_window=bin_window, day_length=day_length,
                                                    t_column=t_column, title=title)

        # if labels is not None:
        #     if labels not in self.meta.columns.tolist():
        #         raise KeyError(f'{labels} is not a column in the metadata')
        #     title_list = self.meta[labels].tolist() 
        # else:
        #     title_list = self.meta.index.tolist()

        # facet_arg = self.meta.index.tolist()
        # root =  self._get_subplots(len(facet_arg))
        
        # data = self.copy(deep=True)

        # # make a square subplot domain
        # fig = make_subplots(rows=root, cols=root, shared_xaxes = False, vertical_spacing = 0.8/root, subplot_titles = title_list)
        # col_list = list(range(1, root+1)) * root
        # row_list = list([i] * root for i in range(1, root+1))
        # row_list = [item for sublist in row_list for item in sublist]

        # data = data.bin_time(mov_variable, bin_window*60, t_column = t_column)
        # data.add_day_phase(t_column = 't_bin')

        # for arg, col, row in zip(facet_arg, col_list, row_list): 

        #     d = data.xmv('id', arg)
        #     d = d.wrap_time(24, time_column = 't_bin')
        #     d['t_bin'] = d['t_bin'] / (60*60)

        #     self._actogram_plot(fig = fig, data = d, mov = mov_variable, day = day_length, row = row, col = col)

        # fig.update_xaxes(
        #     zeroline = False,
        #     color = 'black',
        #     linecolor = 'black',
        #     gridcolor = 'black',
        #     range = [0,48],
        #     tick0 = 0,
        #     dtick = 6,
        #     ticks = 'outside',
        #     tickfont = dict(
        #         size = 12
        #     ),
        #     showgrid = False
        # )

        # fig.update_yaxes(
        #     zeroline = False,
        #     color = 'black',
        #     linecolor = 'black',
        #     gridcolor = 'black',
        #     range = [0,int(data['day'].max())],
        #     tick0 = 0,
        #     dtick = 1,
        #     ticks = 'outside',
        #     showgrid = True,
        #     autorange =  'reversed'
        # )

        # fig.update_annotations(font_size=8)
        # fig['layout']['title'] = title
        # fig['layout']['plot_bgcolor'] = 'white'
        # fig.update_layout(height=150*root, width=250*6)

        # return fig

    def survival_plot(self, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, repeat:bool|str = False, day_length:int = 24, lights_off:int = 12, title:str = '', t_column:str = 't', grids:bool = False):
        """
        Generates a plot of the percentage of animals in a group present / alive over the course of an experiment. This method does not calculate or r
        emove flies that are dead. It is recommended you use the method .curate_dead_animals() to do this. If you have repeats, 
        signposted in the metadata, call the column in the repeat parameter and the standard error will be plotted.
        
        Args:
            facet_col (str, optional): The name of the column to use for faceting. Can be main column or from metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. Default is None.
            facet_labels (list, optional): The labels to use for faceting. Default is None.
            repeat (bool/str, optional): If False the function won't look for a repeat column. If wanted the user should change the argument to the 
                column in the metadata that contains repeat information, which could be increasing integers indicating experiment repeat, 
                i.e. 1, 2, or 3. Default is False
            day_length (int, optional): The length of the day in hours for wrapping. Default is 24.
            lights_off (int, optional): The time of "lights off" in hours. Default is 12.
            title (str, optional): The title of the plot. Default is an empty string.
            t_column (str, optional): The name of the time column in the DataFrame. Default is 't'.
            grids (bool, optional): If True, horizontal grid lines will be displayed on the plot. Default is False.
        
        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
        Note:
            Percentage alive at each timepoint is calculated by taking the highest timepoint per group and 
                checking from start to x_max how many of the group have data points at each hour.        
        """

        if repeat is True:
            if repeat not in self.meta.columns:
                raise KeyError(f'Column "{repeat}" is not a metadata column, please check and add if you want repeat data')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg and repeat:
            data = self.xmv(facet_col, facet_arg).merge(self.meta[[facet_col, repeat]], left_index=True, right_index=True)
            sur_df = data.groupby(facet_col, group_keys = False).apply(partial(self._time_alive, facet_col = facet_col, repeat = repeat, t_column = t_column))
        elif facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg).merge(self.meta[[facet_col]], left_index=True, right_index=True)
            sur_df = data.groupby(facet_col, group_keys = False).apply(partial(self._time_alive, facet_col = facet_col, repeat = repeat, t_column = t_column))
        else:
            data = self.copy(deep=True)
            sur_df = self._time_alive(df = data, facet_col = facet_col, repeat = repeat, t_column = t_column)

        x_ticks = np.arange(0, sur_df['hour'].max() + day_length, day_length)

        fig = go.Figure() 
        self._plot_ylayout(fig, yrange = [0,101], t0 = 0, dtick = 20, ylabel = "Survival (%)", title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = day_length, xlabel = 'ZT (Hours)')

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        if facet_col is not None:
            map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
            sur_df['label'] = sur_df['label'].map(map_dict)
            for lab in facet_labels:
                tdf = sur_df[sur_df['label'] == lab]
                upper, trace, lower, _, _ = self._generate_overtime_plot(data = sur_df, var = 'survived', name = lab, col = palette_dict[lab], avg_win = False,
                                                                    wrap = False, day_len = False, light_off = False, t_col = 'hour')
                fig.add_trace(upper)
                fig.add_trace(trace) 
                fig.add_trace(lower)
        else:
            upper, trace, lower, _, _ = self._generate_overtime_plot(data = sur_df, var = 'survived', name = facet_labels[0], col = palette_dict[facet_labels[0]], avg_win = False,
                                                                wrap = False, day_len = False, light_off = False, t_col = 'hour')
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

        # Light-Dark annotaion bars
        bar_shapes, min_bar = circadian_bars(np.nanmin(x_ticks), np.nanmax(x_ticks), min_y = 0, max_y = 100, day_length = day_length, lights_off = lights_off)
        fig.update_layout(shapes=list(bar_shapes.values()))

        fig['layout']['xaxis']['range'] = [np.min(x_ticks), np.max(x_ticks)]
        fig['layout']['yaxis']['range'] = [min_bar, 101]

        return fig

    # Response AGO/mAGO section

    def plot_response_quantify(self, response_col:str = 'has_responded', facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, z_score:bool = True, title:str = '', grids:bool = False): 
        """ 
        A plotting function for AGO or mAGO datasets that have been loaded with the analysing function stimulus_response.
        A augmented version of plot_quanitfy that finds the average response to a stimulus and the average response
        from a mock stimulus. Must contain the column 'has_interacted' with 1 = True stimulus, 2 = Mock stimulus.
        
        Args:
            response_col = string, the name of the column in the data with the response per interaction, column data should be in boolean form
            facet_col = string, the name of the column in the metadata you wish to filter the data by
            facet_arg = list, if not None then a list of items from the column given in facet_col that you wish to be plotted
            facet_labels = list, if not None then a list of label names for facet_arg. If not provided then facet_arg items are used
            z_score (bool, optional): If True (Default) the z-score for each entry is found the those above/below zero are removed. Default is True.
            title = string, a title for the plotted figure
            grids = bool, true/false whether the resulting figure should have grids

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.
        Notes:
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function or one that mimics it.
        
        """

        grouped_data, h_order, palette_dict =  self._internal_plot_response_quantify(response_col, facet_col, facet_arg, facet_labels) 
        plot_column = f'{response_col}_mean'

        fig = go.Figure() 
        y_range, dtick = self._check_boolean(list(self[response_col]))

        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Resonse Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = '')


        for lab in h_order:

            sub_df = grouped_data[grouped_data['facet_col'] == lab]
            if len(sub_df) == 0:
                print(f'Group {lab} has no values and cannot be plotted')
                continue

            mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column].to_numpy(), z_score = zscore)

            fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
            x = [lab], colour = palette_dict[lab], showlegend = True, name = lab, xaxis = 'x'))

            fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = palette_dict[lab], 
            showlegend = False, name = lab, xaxis = 'x'))

        fig.update_xaxes(showticklabels=False) # remove labels as they're too bulky

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data[facet_col] = grouped_data['facet_col']
            grouped_data = grouped_data[[plot_column, f'{response_col}_std', facet_col]].sort_values(facet_col)

        return fig, grouped_data

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

    def plot_habituation(self, plot_type:str, t_bin_hours:int = 1, response_col:str = 'has_responded', interaction_id_col:str = 'has_interacted', stim_count:bool = True, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None,  x_limit:bool = False, t_column:str = 't', title:str = '', grids:bool = False):
        """ 
        Generate a plot which shows how the response response rate changes over either repeated stimuli (number) or hours post first stimuli (time).
        If false stimuli are given and represented in the interaction_id column, they will be plotted seperately in grey.

        Args:
            plot_type (str): The type of habituation being plotter, either 'number' (the response rate for every stimuli in sequence, i.e. 1st, 2nd, 3rd, ..)
                or 'time' (the response rate per hour(s) post the first stimuli.)
            t_bin_hours (int, optional): The number of hours you want to bin the response rate to. Default is 1 (hour).
            response_col (str, optional): The name of the coloumn that has the responses per interaction. Must be a column of bools. Default is 'has_responded'.
            interaction_id_col (str, optional): The name of the column conataining the id for the interaction type, which should be either 1 (true interaction) or 2 (false interaction). Default 'has_interacted'.
            stim_count (bool, optional): If True statistics for the stimuli are plotted on the secondary y_axis. For 'number' the percentage of specimen revieving
                that number of stimuli is plotted. If 'time', the raw number of stimuli per hour(s) is plotted. False Stimuli are discarded. Default is True
            facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
            facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
            x_limit (int, optional): A number to limit the x-axis by to remove outliers, i.e. 50 would be 50 stimuli for 'number'. Default False.
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'.
            title (str, optional): The title of the plot. Default is an empty string.
            grids (bool, optional): true/false whether the resulting figure should have grids. Default is False

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
            
        Notes:
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function. Contain columns such as 'has_responded' and 'has_interacted'.
            The stimulus count plot only calculates the percentage or total for the true stimulus, discarding the false
                for visual clarity.
        """  

        seconday_label = {'time' : f'No. of stimulus (absolute)', 'number' : '% recieving stimulus'}

        # call the internal method to curate and analse data, see behavpy_draw
        grouped_data, h_order, palette_dict, x_max, plot_choice = self._internal_plot_habituation(plot_type=plot_type, t_bin_hours=t_bin_hours, response_col=response_col, interaction_id_col=interaction_id_col,
                                                                                        facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, x_limit=x_limit, t_column=t_column)
        # create and style plot
        if stim_count is False:
            fig = go.Figure() 
        else:
            fig = make_subplots(specs=[[{ "secondary_y" : True}]])
            if plot_type == 'number':
                yran = [0,101]
            else:
                yran = False
            self._plot_ylayout(fig, yrange = yran, t0 = 0, dtick = False, ylabel = seconday_label[plot_type], title = title, secondary = True, xdomain = 'x1', grid = grids)
            
        self._plot_ylayout(fig, yrange = [0, 1], t0 = 0, dtick = 0.2, ylabel = 'Response Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = [0, x_max], t0 = 0, dtick = False, xlabel = plot_choice)

        for hue in h_order:
            sub_df = grouped_data[grouped_data.index == hue]

            upper, trace, lower = self._plot_line(df = sub_df, x_col = plot_choice, name = hue, marker_col = palette_dict[hue])
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

            if stim_count is True and '-True Stimulus' in hue:
                if plot_type == 'number':
                    sub_df['count'] = (sub_df['count'] / np.max(sub_df['count'])) * 100
                else:
                    sub_df['count'] = sub_df['stim_count']

                fig.add_trace(
                go.Scatter(
                    legendgroup = hue,
                    x = sub_df[plot_choice],
                    y = sub_df['count'],
                    mode = 'lines',
                    name = f'{hue} count',
                    line = dict(
                        dash = 'longdashdot',
                        shape = 'spline',
                        color = palette_dict[hue]
                        ),
                    ),
                secondary_y = True
                )
        
        fig['layout'].update(legend =  dict(x = 1.06, y = 0.5)) # shift the legend so it doesn't overlap

        return fig

    def plot_response_over_activity(self, mov_df, activity:str, variable:str = 'moving', response_col:str = 'has_responded', facet_col:None|str = None,
        facet_arg:None|list = None, facet_labels:None|list = None, x_limit:int = 30, t_bin:int = 60, title:str = '', t_column:str = 't', grids:bool = False):
        """ 
        A plotting function for AGO or mAGO datasets that have been loaded with the analysing function stimulus_response.
        Generate a plot which shows how the response response rate changes over time inactive or active.

            Args:
                mov_df (behavpy dataframe): The matching behavpy dataframe containing the movement data from the response experiment
                activity (str): A choice to display reponse rate for continuous bounts of inactivity, activity, or both. Choice one of ['inactive', 'active', 'both']
                variable (str, optional): The name of column in the movement dataframe that has the boolean movement values. Default is 'moving'.
                response_col (str, optional): The name of the coloumn that has the responses per interaction. Must be a column of bools. Default is 'has_responded'.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. 
                    Default is None.
                x_limit (int, optional): A number to limit the x-axis by to remove outliers, i.e. 30 would be 30 minutes or less if t_bin is 60. Default 30.
                t_bin (int, optional): The time in seconds to bin the time series data to. Default is 60.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'.
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
            
        Notes:
            This plotting method can show the response rate for both active and inactive bouts for the 
                whole dataset, but only for one or the other if you want to facet by a column, i.e. facet_col.
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function.
        """        

        # call the internal method to curate and analse data, see behavpy_draw
        grouped_data, h_order, palette_dict = self._internal_bout_activity(mov_df=mov_df, activity=activity, variable=variable, response_col=response_col, 
                                    facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, x_limit=x_limit, t_bin=t_bin, t_column=t_column)
        # create and style plot
        fig = go.Figure() 
        self._plot_ylayout(fig, yrange = [0, 1], t0 = 0, dtick = 0.2, ylabel = 'Response Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = 1, xlabel = f'Consecutive minutes in behavioural bout ({activity})')

        for hue in h_order:

            sub_df = grouped_data[grouped_data['label_col'] == hue]
            # if no data, such as no false stimuli, skip the plotting
            if len(sub_df) == 0:
                continue

            upper, trace, lower = self._plot_line(df = sub_df, x_col = 'previous_activity_count', name = hue, marker_col = palette_dict[hue])
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)
        
        return fig

    def plot_response_overtime(self, t_bin_hours:int = 1, wrapped:bool = False, response_col:str = 'has_responded', interaction_id_col:str = 'has_interacted', 
        facet_col:None|str = None, facet_arg:None|list = None, facet_labels:None|list = None, day_length:int = 24, lights_off:int = 12, func:str = 'mean', 
        t_column:str = 't', title:str = '', grids:bool = False):
        """ 
        A plotting function for AGO or mAGO datasets that have been loaded with the analysing function stimulus_response.
        Generate a plot which shows how the response response rate changes over a day (wrapped) or the course of the experiment.
        If false stimuli are given and represented in the interaction_id column, they will be plotted seperately.

            Args:
                t_bin_hours (int, optional): The number of hours you want to bin the response rate to per specimen. Default is 1 (hour).
                wrapped (bool, optional): If true the data is augmented to represent one day, combining data of the same time on consequtive days.
                response_col (str, optional): The name of the coloumn that has the responses per interaction. Must be a column of bools. Default is 'has_responded'.
                interaction_id_col (str, optional): The name of the column conataining the id for the interaction type, 
                    which should be either 1 (true interaction) or 2 (false interaction). Default 'has_interacted'.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. 
                    If None the labels will be those from the metadata. Default is None.
                day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
                lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. 
                    Must be number between 0 and day_lenght. Default is 12.
                func (str, optional): When binning the time what function to apply the variable column. Default is 'max'.                
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
            
        Notes:
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function. Contain columns such as 'has_responded' and 'has_interacted'.
        """  
        df, h_order, palette = self._internal_plot_response_overtime(t_bin_hours=t_bin_hours, response_col=response_col, interaction_id_col=interaction_id_col, 
                                                facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, func=func, t_column=t_column)

        return df.plot_overtime(variable='Response Rate', wrapped=wrapped, facet_col='new_facet', facet_arg=h_order, facet_labels=h_order,
                                avg_window=5, day_length=day_length, lights_off=lights_off, title=title, grids=grids, t_column='t_bin', 
                                col_list = palette)

    # Possibly add the puff count on the secondary access in the future 

    #         fig = make_subplots(specs=[[{ "secondary_y" : True}]])
    #         self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = False, ylabel = 'Puff Count', title = title, secondary = True, xdomain = 'x1', grid = grids)

    #             if secondary is True:
    #                 fig.add_trace(
    #                 go.Scatter(
    #                     legendgroup = lab,
    #                     x = filt_gb['bin_time'],
    #                     y = filt_gb['count'],
    #                     mode = 'lines',
    #                     name = f'{lab} count',
    #                     line = dict(
    #                         dash = 'longdashdot',
    #                         shape = 'spline',
    #                         color = qcol
    #                         ),
    #                     ),
    #                 secondary_y = True
    #                 )

    # Ploty Periodograms

    def plot_periodogram_tile(self, labels:None|str = None, find_peaks:bool = False, title:str = '', grids:bool = False):
        """ 
        Generates a periodogram plot for every specimen in the metdata. Periodograms show the power of each frequency (rythmn) within the data. 
        For a normal specimen a peak in power at 24 hours is expected. A threshold line (in red) is also plotted, power above this line is 
        significant given the alpha value when calculating frequency power with .periodogram().

            Args:
                labels (str, optional): The name of the column in the metadata that contains unique labels per specimen. If None then
                    the index ids are used. Default is None.
                find_peaks (bool, optional): If True then the highest frequency that is signifcant is found and marked with an X.
                    Default is False.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
        Raises:
            AttributeError is this method is called on a beahvpy object that has not been augmented by .periodogram()
                or doesn't have the columns "power" or "period".
        """

        self._validate()

        if labels is not None:
            if labels not in self.meta.columns.tolist():
                raise AttributeError(f'{labels} is not a column in the metadata')
            title_list = self.meta[labels].tolist() 
        else:
            title_list = self.meta.index.tolist()
        
        facet_arg = self.meta.index.tolist()

        data = self.copy(deep = True)

        # Find the peaks if True
        if find_peaks is True:
            if 'peak' not in data.columns.tolist():
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
                tdf = d[~d['peak'].isna()]
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

    def plot_periodogram(self, facet_col:None|str = None, facet_arg:None|list = None, facet_labels:None|list = None, 
        title:str = '', grids:bool = False):
        """ 
        Generates a periodogram plot that is averaged over the whole dataset or faceted by a metadata column.
        Periodograms are a good way to quantify through signal analysis the ryhthmicity of your dataset.
        
            Args:
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. 
                    Default is None.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): True/False to whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
        Raises:
            AttributeError is this method is called on a beahvpy object that has not been augmented by .periodogram()
                or doesn't have the columns "power" or "period".
        """
        # check if the dataset has the needed columns from .periodogram()
        self._validate()
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]

        col_list = self._get_colours(d_list)

        power_var = 'power'
        period_var = 'period'        
        
        max_var = []
        y_range, dtick = self._check_boolean(list(self[power_var].dropna()))
        if y_range is False:
            max_var.append(1)
        
        fig = go.Figure() 
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Power', title = title, grid = grids)
        tick_6 = np.arange(0,200*6,6)
        self._plot_xlayout(fig, xrange = [min(self[period_var]), max(self[period_var])], tickvals = tick_6, ticktext = tick_6, xlabel = 'Period Frequency (Hours)')

        for data, name, col in zip(d_list, facet_labels, col_list):

            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'grey'

            upper, trace, lower, _, _ = self._generate_overtime_plot(data = data, name = name, col = col, var = power_var, avg_win = False, wrap = False, day_len = False, light_off = False, t_col = period_var)
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

        return fig
    
    def plot_periodogram_quantify(self, facet_col:None|str = None, facet_arg:None|list = None, facet_labels:None|list = None, title:str = '', 
        grids:bool = False):        
        """
        Creates a boxplot and swarmplot of the peaks in circadian rythymn according to a computed periodogram.
        At its core it is just a wrapper of plot_quantify, with some data augmented before being sent to the method.

            Args:
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. 
                    Default is None.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): True/False to whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure to be plotted as (width, height). If set to 
                    (0,0), the size is determined automatically. Default is (0,0).
        
        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.
        Raises:
            AttributeError is this method is called on a beahvpy object that has not been augmented by .periodogram()
                or doesn't have the columns "power" or "period".
        """
        # check it has the right periodogram columns
        self._validate()

        # name for plotting
        power_var = 'period'
        y_label = 'Period (Hours)'

        # find period peaks for plotting
        if 'peak' not in self.columns.tolist():
            self = self.find_peaks(num_peaks = 1)
        self = self[self['peak'] == 1]
        self = self.rename(columns = {power_var : y_label})
    
        return self.plot_quantify(variable = y_label, facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, 
                                    fun='max', title=title, grids=grids)

    def plot_wavelet(self, mov_variable:str, sampling_rate:int = 15, scale:int = 156, wavelet_type:str = 'morl', t_col:str = 't', title:str = '', grids:bool = False):
        """ 
        Analyses a dataset with a movement column using wavelets, which preserve the time dimension.
        Plots contain the time of the experiment on the x-axis, frequency on the y-axis, and power on the z-axis.
        With this you can see how rhythmicity changes in an experimemnt overtime.

            Args:
                mov_variable (str):The name of the column containting the movement data.
                sampling_rate (int, optional): The time in minutes the data should be augmented to. Default is 15 minutes
                scale (int optional): The scale facotr, the smaller the scale the more stretched the plot. Default is 156.
                wavelet_type (str, optional): The wavelet family to be used to decompose the sequences. Default is 'morl'.
                    A list of the types of wavelets can be generated by calling .wavelet_types(). Head to https://pywavelets.readthedocs.io/en/latest/
                    for the latest information on the package used in the backend, pywavelets.
                t_col (str, optional): The name of the time column in the DataFrame. Default is 't'.
                title (str, optional): The title of the plot. Default is an empty string.
                
        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
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


    # HMM section

    def plot_hmm_overtime(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, wrapped:bool = False, t_bin:int = 60, 
        func:str = 'max', avg_window:int = 30, day_length:int = 24, lights_off:int = 12, title:str = '', t_column:str = 't', 
        grids:bool = False):
        """
        Generates a plot of the occurance of each state as a percentage at each time point for the whole dataset. The is the go to plot for understanding 
        how states change over the day.

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. 
                    If None and not 4 states then generic labels are generated, i.e. 'state-1, state-2, state-n'.
                    Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. 
                    If None the colours are by default for 4 states (blue and red), if not 4 then colours from the palette are chosen. 
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn. Default is None.
                wrapped (bool, optional). If True the plot will be limited to a 24 hour day average. Default is False.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable
                avg_window (int, optioanl): The window in minutes you want the moving average to be applied to. Default is 30 mins
                day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
                lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. 
                    Must be number between 0 and day_lenght. Default is 12.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
                Generated by the .plot_overtime() method
        Note:
            The method decodes and augments the dataset, which is then fed into the plot_overtime method.

        """
        assert isinstance(wrapped, bool)

        df = self.copy(deep = True)

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)

        states_list, time_list = self._hmm_decode(df, hmm, t_bin, variable, func, t_column)

        df = pd.DataFrame()
        for l, t in zip(states_list, time_list):
            tdf = hmm_pct_state(l, t, list(range(len(labels))), avg_window = int((avg_window * 60)/t_bin))
            df = pd.concat([df, tdf], ignore_index = True)

        df.rename(columns = dict(zip([f'state_{c}' for c in range(0,len(labels))], labels)), inplace = True)
        melt_df = df.melt('t')
        m = melt_df[['variable']]
        melt_df = melt_df.rename(columns = {'variable' : 'id'}).set_index('id')
        m['id'] = m[['variable']]
        m = m.set_index('id')

        df = self.__class__(melt_df, m)
        return df.plot_overtime(variable='value', wrapped=wrapped, facet_col='variable', facet_arg=labels, avg_window=avg_window, day_length=day_length, 
                                    lights_off=lights_off, title=title, grids=grids, t_column='t', col_list = colours)

    def plot_hmm_split(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|str = None, facet_arg:None|list = None, 
        facet_labels:None|list = None, wrapped:bool = False, t_bin:int|list = 60, func:str = 'max', avg_window:int = 30, day_length:int = 24, lights_off:int = 12,
        title:str = '', t_column:str = 't', grids:bool = False):
        """
        Generates a plot similar to plot_hmm_overtime(). However, each state is plotted on its own in a grid arrangement. This plot is best used for when 
        comparing multiple groups, as with the use of faceting. If you want to compare HMMs you can provide a list of HMMs which will each be applied either 
        to the whole dataset or to the given groups in facet_arg provided. Please ensure the list of HMMS equals the facet_arg list.

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the correct hidden states and emission 
                    states for your dataset. Can be a single HMM or a list of HMMs. If a list then it can be used in two ways. If a list and facet_col
                    remains None, then each HMM is applied to the whole dataset with a given label 'HMM-X' dependend on order. If a with and facet_col 
                    is given, then a HMM will be applied to each group given the order of facet_arg. Due to this you must provide a list of the groups
                    to facet_arg when giving a facet_col.
                variable (str, optional): The column heading of the variable of you wish to decode. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. 
                    If None and not 4 states then generic labels are generated, i.e. 'state-1, state-2, state-n'.
                    Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. 
                    If None the colours are by default for 4 states (blue and red), if not 4 then colours from the palette are chosen. 
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn. Default is None.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. 
                    Default is None.
                wrapped (bool, optional). If True the plot will be limited to a 24 hour day average. Default is False.
                t_bin (int|list, optional): The time in seconds you want to bin the movement data to. If giving a list of HMMs
                    this must also be a list. Default is 60 or 1 minute
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable
                avg_window (int, optioanl): The window in minutes you want the moving average to be applied to. Default is 30 mins
                day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
                lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. 
                    Must be number between 0 and day_lenght. Default is 12.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
                Generated by the .plot_overtime() method       
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        Note:
            If providing multiple HMMs but no facet_col, then leave labels as None. Generic labels will be generated
            for the largest state model.
        """
        assert isinstance(wrapped, bool)

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        if facet_col is None and isinstance(hmm, list): facet_col = True # for when testing different HMMs on the same dataset

        if len(labels) <= 2:
            nrows = 1
            ncols =2
        else:
            nrows =  2
            ncols = round(len(labels) / 2)

        fig = make_subplots(rows= nrows, cols= ncols, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.04, horizontal_spacing=0.02)

        col_list = list(range(1, ncols+1)) * nrows
        row_list = list([i] * ncols for i in range(1, nrows+1))
        row_list = [item for sublist in row_list for item in sublist]

        if facet_col is not None:
            colours = self._get_colours(facet_labels)
            colours = [self._check_grey(name, colours[c])[1] for c, name in enumerate(facet_labels)] # change to grey if control

        df = self.copy(deep=True)

        if isinstance(hmm, list) is False: # if only 1 hmm with either facet_col or not, decode as whole for efficiency
            df = self.__class__(self._hmm_decode(df, hmm, t_bin, variable, func, t_column, return_type='table'), df.meta, check=True)
            h_list = [hmm] * len(facet_arg) # change to list for loop
            b_list = [t_bin] * len(facet_arg)

        for c, (arg, n, h, b) in enumerate(zip(facet_arg, facet_labels, h_list, b_list)):   

            if arg != None:
                sub_df = df.xmv(facet_col, arg)
            else:
                sub_df = df

            if isinstance(hmm, list) is True: # call the decode here if multiple HMMs
                states_list, time_list = self._hmm_decode(sub_df, h, b, variable, func, t_column)
            else:
                states_list = sub_df.groupby(sub_df.index)['state'].apply(np.array)
                time_list = sub_df.groupby(sub_df.index)['bin'].apply(list)

            analysed_df = pd.DataFrame()
            for l, t in zip(states_list, time_list):
                temp_df = hmm_pct_state(l, t, [0, 1, 2, 3], avg_window = int((avg_window * 60)/b))
                analysed_df = pd.concat([analysed_df, temp_df], ignore_index = False)

            if wrapped is True:
                analysed_df['t'] = analysed_df['t'].map(lambda t: t % (60*60*day_length))
            analysed_df['t'] = analysed_df['t'] / (60*60)

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

                if facet_col is None:
                    upper, trace, lower = self._plot_line(df = gb_df, x_col = 't', name = n, marker_col = colours[i])
                else:
                    upper, trace, lower = self._plot_line(df = gb_df, x_col = 't', name = n, marker_col = colours[c])

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
                    yshift = -48
                )
        fig.add_annotation(
                    font = {'size': 18, 'color' : 'black'},
                    showarrow = False,
                    text = 'Probability of state',
                    x = 0,
                    xanchor = 'left',
                    xref = 'paper',
                    y = 0.5,
                    yanchor = 'middle',
                    yref = 'paper',
                    xshift =  -80,
                    textangle =  -90
        )

        # Light-Dark annotaion bars
        bar_shapes, _ = circadian_bars(t_min, t_max, min_y = 0, max_y = 1, day_length = day_length, lights_off = lights_off, split = len(labels))
        fig.update_layout(shapes=list(bar_shapes.values()))

        return fig

    def plot_hmm_quantify(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|str = None, 
        facet_arg:None|list = None, facet_labels:None|list = None, t_bin:int = 60, col_uniform:bool = False, func:str = 'max', 
        z_score:bool = True, title:str = '', t_column:str = 't', grids:bool = False):
        """
        Generates a quantification plot of how much a state is predicted as a percentage of the whole time. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the 
                    correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. 
                    If None and not 4 states then generic labels are generated, i.e. 'state-1, state-2, state-n'.
                    Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. 
                    If None the colours are by default for 4 states (blue and red), if not 4 then colours from the palette are chosen. 
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn. Default is None.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. 
                    Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. 
                    If None the labels will be those from the metadata. Default is None.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                col_uniform (bool, optional): Unique to the plotly version of this method. 
                    When True the colour scheme is based on the states and not the faceted column. 
                    When false the colour palette is used instead as in the Seaborn version. Default is False.
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                z_score (bool, optional): If True (Default) the z-score for each entry is found the those above/below zero are removed. 
                    Default is True.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        """

        grouped_data, labels, colours, facet_col, facet_labels, palette_dict = self._internal_plot_hmm_quantify(hmm, variable, labels, colours, facet_col, 
                                                                                    facet_arg, facet_labels, t_bin, func, t_column)
        
        if col_uniform: # change the palette to be state labels rather than facet labe
            palette_dict = {name : self._check_grey(name, colours[c])[1] for c, name in enumerate(labels)} # change to grey if control  

        plot_column = 'Fraction of time in each State'
        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 1.01], t0 = 0, dtick = 0.2, ylabel = plot_column, title = title, grid = grids)

        for c, state in enumerate(labels):

            sub_df = grouped_data[grouped_data['state'] == state]

            if facet_col:
                for lab in facet_labels:

                    sub_sub_df = sub_df[sub_df[facet_col] == lab]

                    if col_uniform: col = palette_dict[state]
                    else: col = palette_dict[lab]

                    try:
                        mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)
                    except IndexError:
                        mean, median, q3, q1, zlist = 0, 0, 0, 0, [0]

                    fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                    x = [lab], colour =  col, showlegend = True, name = lab, xaxis = f'x{c+1}'))

                    fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = col, 
                    showlegend = False, name = lab, xaxis = f'x{c+1}'))
            else:
                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [state], colour =  palette_dict[state], showlegend = True, name = state, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [state], colour = palette_dict[state], 
                showlegend = False, name = state, xaxis = f'x{c+1}'))

            axis = f'xaxis{c+1}'
            if facet_col is None: state = ' ' # to prevent double labelling of light dark
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = state, domains = domains[c:c+2], axis = axis)

        # reorder dataframe for stats output
        grouped_data.drop(columns=['bin', 'previous_state'], inplace=True)
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data
    
    def plot_hmm_quantify_length(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|str = None, 
        facet_arg:None|list = None, facet_labels:None|list = None, scale:str = 'log', t_bin:int = 60, col_uniform:bool = False, func:str = 'max', 
        z_score:bool =  True, title:str = '', t_column:str = 't', grids:bool = False):
        """
        Generates a quantification plot of the average length of each state per individual per group. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the 
                    correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. 
                    If None and not 4 states then generic labels are generated, i.e. 'state-1, state-2, state-n'.
                    Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. 
                    If None the colours are by default for 4 states (blue and red), if not 4 then colours from the palette are chosen. 
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn. Default is None.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. 
                    Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. 
                    If None the labels will be those from the metadata. Default is None.
                scale (str, optional): The scale of the y-axis. Default is 'log'.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                col_uniform (bool, optional): Unique to the plotly version of this method. 
                    When True the colour scheme is based on the states and not the faceted column. 
                    When false the colour palette is used instead as in the Seaborn version. Default is False.
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                z_score (bool, optional): If True (Default) the z-score for each entry is found the those above/below zero are removed. 
                    Default is True.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        """
        grouped_data, labels, colours, facet_col, facet_labels, palette_dict = self._internal_plot_hmm_quantify_length(hmm, variable, labels, colours, facet_col, 
                                                                                    facet_arg, facet_labels, t_bin, func, t_column)
        if col_uniform: # change the palette to be labels rather than facet
            palette_dict = {name : self._check_grey(name, colours[c])[1] for c, name in enumerate(labels)} # change to grey if control  
        plot_column = 'Length of state bout (mins)'
        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = 0.69897000433, ylabel = plot_column, ytype = scale, title = title, grid = grids)

        for c, state in enumerate(labels):

            sub_df = grouped_data[grouped_data['state'] == state]

            if facet_col:
                for lab in facet_labels:

                    sub_sub_df = sub_df[sub_df[facet_col] == lab]

                    if col_uniform: col = palette_dict[state]
                    else: col = palette_dict[lab]

                    mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                    fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                    x = [lab], colour =  col, showlegend = True, name = lab, xaxis = f'x{c+1}'))

                    fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = col, 
                    showlegend = False, name = lab, xaxis = f'x{c+1}'))
            else:
                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [state], colour =  palette_dict[state], showlegend = True, name = state, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [state], colour = palette_dict[state], 
                showlegend = False, name = state, xaxis = f'x{c+1}'))

            axis = f'xaxis{c+1}'
            if facet_col is None: state = ' ' # to prevent double labelling of light dark
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = state, domains = domains[c:c+2], axis = axis)

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_hmm_quantify_length_min_max(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|str = None, 
        facet_arg:None|list = None, facet_labels:None|list = None, scale:str = 'log', plot_points:bool = False, t_bin:int = 60, col_uniform:bool = False, 
        func:str = 'max', z_score:bool =  True, title:str = '', t_column:str = 't', grids:bool = False):
        """
        Generates a plot of the average length of each state per individual per group. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the 
                    correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. 
                    If None and not 4 states then generic labels are generated, i.e. 'state-1, state-2, state-n'.
                    Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. 
                    If None the colours are by default for 4 states (blue and red), if not 4 then colours from the palette are chosen. 
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn. Default is None.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. 
                    Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. 
                    If None the labels will be those from the metadata. Default is None.
                scale (str, optional): The scale of the y-axis. Default is 'log'.
                plot_points (bool, optional); If True then the length of each run is plotted. However, as it is every state run within each specimen
                    this can generate many points making the notebook run slowly or crash . Default is False.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                col_uniform (bool, optional): Unique to the plotly version of this method. 
                    When True the colour scheme is based on the states and not the faceted column. 
                    When false the colour palette is used instead as in the Seaborn version. Default is False.
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                z_score (bool, optional): If True (Default) the z-score for each entry is found the those above/below zero are removed. 
                    Default is True.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        Notes:
            In processing the first and last bouts of the variable fed into the HMM are trimmed to prevent them affecting the result. 
            Any missing data points will also affect the end quantification.
        """

        grouped_data, labels, colours, facet_col, facet_labels, palette_dict = self._internal_plot_hmm_quantify_length_min_max(hmm, variable, labels, colours, facet_col, 
                                                                                    facet_arg, facet_labels, t_bin, func, t_column)
        if col_uniform: # change the palette to be labels rather than facet
            palette_dict = {name : self._check_grey(name, colours[c])[1] for c, name in enumerate(labels)} # change to grey if control  

        plot_column = 'Length of state bout (mins)'
        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = 0.69897000433, ylabel = plot_column, ytype = scale, title = title, grid = grids)

        for c, state in enumerate(labels):

            sub_df = grouped_data[grouped_data['state'] == state]

            if facet_col:
                for lab in facet_labels:

                    sub_sub_df = sub_df[sub_df[facet_col] == lab]

                    if col_uniform: col = palette_dict[state]
                    else: col = palette_dict[lab]

                    mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_sub_df[plot_column].to_numpy(dtype=float), min_max = True, z_score = False)

                    fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                    x = [lab], colour =  col, showlegend = True, name = lab, xaxis = f'x{c+1}'))
                    if plot_points:
                        fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = col, 
                        showlegend = False, name = lab, xaxis = f'x{c+1}'))
            else:
                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column].to_numpy(dtype=float), min_max = True, z_score = False)

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [state], colour =  palette_dict[state], showlegend = True, name = state, xaxis = f'x{c+1}'))
                if plot_points:
                    fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [state], colour = palette_dict[state], 
                    showlegend = False, name = state, xaxis = f'x{c+1}'))

            axis = f'xaxis{c+1}'
            if facet_col is None: state = ' ' # to prevent double labelling of light dark
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = state, domains = domains[c:c+2], axis = axis)

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data
            
    def plot_hmm_quantify_transition(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|str = None, 
        facet_arg:None|list = None, facet_labels:None|list = None, t_bin:int = 60, col_uniform:bool = False, 
        func:str = 'max', z_score:bool =  True, title:str = '', t_column:str = 't', grids:bool = False):        
        """
        Generates a plot of the average amount each state is transitioned into. This is a good plot to compare with plot_hmm_quantify() to understand the interplay
        between frequent transitions into a state and staying in a state.

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the 
                    correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. 
                    If None and not 4 states then generic labels are generated, i.e. 'state-1, state-2, state-n'.
                    Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. 
                    If None the colours are by default for 4 states (blue and red), if not 4 then colours from the palette are chosen. 
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn. Default is None.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. 
                    Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. 
                    If None the labels will be those from the metadata. Default is None.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                col_uniform (bool, optional): Unique to the plotly version of this method. 
                    When True the colour scheme is based on the states and not the faceted column. 
                    When false the colour palette is used instead as in the Seaborn version. Default is False.
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                z_score (bool, optional): If True (Default) the z-score for each entry is found the those above/below zero are removed. 
                    Default is True.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        Notes:
            In processing the first and last bouts of the variable fed into the HMM are trimmed to prevent them affecting the result. 
            Any missing data points will also affect the end quantification.
        """

        grouped_data, labels, colours, facet_col, facet_labels, palette_dict = self._internal_plot_hmm_quantify_transition(hmm, variable, labels, colours, facet_col, 
                                                                                    facet_arg, facet_labels, t_bin, func, t_column)
        if col_uniform: # change the palette to be labels rather than facet
            palette_dict = {name : self._check_grey(name, colours[c])[1] for c, name in enumerate(labels)} # change to grey if control  

        plot_column = 'Fraction of transitions into each state'
        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = 0.69897000433, ylabel = plot_column, title = title, grid = grids)

        for c, state in enumerate(labels):

            sub_df = grouped_data[grouped_data['state'] == state]

            if facet_col:
                for lab in facet_labels:

                    sub_sub_df = sub_df[sub_df[facet_col] == lab]

                    if col_uniform: col = palette_dict[state]
                    else: col = palette_dict[lab]

                    mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                    fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                    x = [lab], colour =  col, showlegend = True, name = lab, xaxis = f'x{c+1}'))

                    fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = col, 
                    showlegend = False, name = lab, xaxis = f'x{c+1}'))
            else:
                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_df[plot_column].to_numpy(dtype=float), z_score = z_score)

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [state], colour =  palette_dict[state], showlegend = True, name = state, xaxis = f'x{c+1}'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [state], colour = palette_dict[state], 
                showlegend = False, name = state, xaxis = f'x{c+1}'))

            axis = f'xaxis{c+1}'
            if facet_col is None: state = ' ' # to prevent double labelling of light dark
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = state, domains = domains[c:c+2], axis = axis)

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_hmm_raw(self, hmm, variable:str = 'moving', colours:list = None, num_plots:int = 5, stim_df = None, show_movement:bool = False, 
        t_bin:int = 60, func:str = 'max', day_length:int = 24, lights_off:int = 12, t_column:str = 't', title:str = '', grids:bool = False):
        """
        Generates a plot that represents each predicted state over the experiment. This plot is great for getting a quick feel to your HMM and useful
        for comparing multiple HMMs trained with different architecture on the same dataset. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM | list([hmmlearn.hmm.CategoricalHMM ])): This should be a trained HMM Learn object with the 
                    correct hidden states and emission states for your dataset. If as a list the number of plots will be the same as its length,
                    with seach plot being the same specimens data decoded with each HMM.
                variable (str, optional): The column heading of the variable of interest for decoding. Default is "moving"
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states.
                    If None the colours are a default for 4 states (blue and red) or the first X number of colours in the given palette. Default is None.
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn.
                    If given a list of HMMs, the largest n state model is found and colours checked against.
                numm_plots (int, optional): The number of plots as rows in a subplot. If a list of HMMs is given num_plots will be overridden
                    by the length of that list. Default is 5.
                stim_df (behavpy df, optional): A corresponding dataframe with responses to stimuli as loaded in by stimulus_response. 
                    When present True responses will be purple and False responses will be lime. Only for true stimuli, not mock. Default is None.
                show_movement (bool, optional): If True each plot will be overlayed with the given variable, not decoded, see note below. Default is False.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute.
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable
                day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
                lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. 
                    Default is 12.    
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                title (str, optional): The title of the plot. Default is an empty string.  
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            A plotly figure that is composed of scatter plots.
        Raises:
            TypeError:
                Can be thrown sometimes when the filtered dataframe contains no data. If this occurs run the method again to randomnly select a 
                    different specimen for plotting.
        Note:
            If show_movement is true the categories of your variable must be less than the number of states. If greater it 
                will be plotted off the plot.
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = None, col = colours)

        colours_index = {c : col for c, col in enumerate(colours)}

        if stim_df is not None:
            assert isinstance(stim_df, self.__class__), 'The stim_df dataframe is not behavpy dataframe'

        if isinstance(hmm, list):
            num_plots = len(hmm)
            rand_flies = [np.random.permutation(list(set(self.meta.index)))[0]] * num_plots
            h_list = hmm
            if isinstance(t_bin, list):
                b_list = t_bin 
            else:
                b_list = [t_bin] * num_plots
        else:
            rand_flies = np.random.permutation(list(set(self.meta.index)))[:num_plots]
            h_list = [hmm] * num_plots
            b_list = [t_bin] * num_plots

        df_list = [self.xmv('id', id) for id in rand_flies]
        decoded = [self._hmm_decode(d, h, b, variable, func, t_column, return_type='table').dropna().set_index('id') for d, h, b in zip(df_list, h_list, b_list)]

        fig = make_subplots(
        rows= num_plots, 
        cols=1,
        shared_xaxes=True, 
        shared_yaxes=True, 
        vertical_spacing=0.02,
        horizontal_spacing=0.02
        )

        y_range = [-0.2, len(labels)-0.8]
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = False, ylabel = '', title = title)

        min_t = []
        max_t = []

        for c, (df, b) in enumerate(zip(decoded, b_list)):

            df['colour'] = df['previous_state'].map(colours_index)
            id = df.first_valid_index()
            print(f'Plotting: {id}')

            if stim_df is not None:
                df2 = stim_df.xmv('id', id)
                df2 = df2[df2['has_interacted'] == 1]
                df2['bin'] = df2['interaction_t'].map(lambda t:  b * floor(t / b))
                df2.reset_index(inplace = True)
                df = pd.merge(df, df2, how = 'outer', on = ['id', 'bin'])
                df['colour'] = np.where(df['has_responded'] == True, 'purple', df['colour'])
                df['colour'] = np.where(df['has_responded'] == False, 'lime', df['colour'])
            
            df['bin'] = df['bin'] / (60*60) # change time to be a fraction of an hour

            trace1 = go.Scatter(
                showlegend = False,
                y = df['previous_state'],
                x = df['bin'],
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
                df[variable] = np.roll((df[variable] * 2) + 0.5, 1)
                trace2 = go.Scatter(
                    showlegend = False,
                    y = df[variable],
                    x = df['bin'],
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

            min_t.append(int((day_length/lights_off) * floor(df['bin'].min() / (day_length/lights_off))))
            max_t.append(int((day_length/lights_off) * floor(df['bin'].max() / (day_length/lights_off))))

            # Light-Dark annotaion bars
            # bar_shapes, min_bar = circadian_bars(np.nanmin(df['bin']), np.nanmax(df['bin']), min_y = y_range[0], max_y = y_range[1], day_length = day_length, lights_off = lights_off)
            # fig.update_layout(shapes=list(bar_shapes.values()))

        if np.max(max_t) - np.min(min_t) <= 24:
            lights_off = None

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
            dtick = lights_off,
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
    
    def plot_hmm_response(self, mov_df, hmm, variable:str = 'moving', response_col:str = 'has_responded', labels:list = None, colours:list = None, 
        facet_col:None|str = None, facet_arg:None|list = None, facet_labels:None|list = None, t_bin:int = 60, col_uniform:bool = False, func:str = 'max', 
        z_score:bool = True, title:str = '', t_column:str = 't', grids:bool = False):
        """
        Generates a plot to explore the response rate to a stimulus per hidden state from a Hidden markov Model. 
        Y-axis is the average response rate per group / state / True or mock interactions


            Args:
                mov_df (behavpy dataframe): The matching behavpy dataframe containing the movement data from the response experiment
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the 
                    correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                response_col (str, optional): The name of the coloumn that has the responses per interaction. 
                    Must be a column of bools. Default is 'has_responded'.
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. 
                    If None and not 4 states then generic labels are generated, i.e. 'state-1, state-2, state-n'.
                    Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. 
                    If None the colours are by default for 4 states (blue and red), if not 4 then colours from the palette are chosen. 
                    It accepts a specific colour or an array of numbers that are acceptable to Plotly. Default is None.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. 
                    Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. 
                    If None the labels will be those from the metadata. Default is None.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                col_uniform (bool, optional): Unique to the plotly version of this method. 
                    When True the colour scheme is based on the states and not the faceted column. 
                    When false the colour palette is used instead as in the Plotly version. Default is False.
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                z_score (bool, optional): If True (Default) the z-score for each entry is found the those above/below zero are removed. 
                    Default is True.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.

        Returns:
            fig (plotly.figure.Figure): Figure object of the plot.
        Raises:
            KeyError:
                If a column for the respones is not a column.   
            Multiple assertion of ValueErrors and KeyErrors in regards to faceting and HMM lists
        Note:
            This function must be called on a behavpy dataframe that is populated by data loaded in with the stimulus_response
            analysing function.
        """

        if response_col not in self.columns.tolist():
            raise KeyError(f'The column you gave {response_col}, is not in the data. Check you have analysed the dataset with stimulus_response')

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, hmm, t_bin = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        if isinstance(hmm, list) and facet_col is None: # End method is trying to have mutliple HMMs with no facet
            raise RuntimeError('This method does not support multiple HMMs and no facet_col')
        
        plot_column = f'{response_col}_mean'

        grouped_data, palette_dict, h_order = self._hmm_response(mov_df, hmm, variable, response_col, labels,
                                                                    colours, facet_col, facet_arg, facet_labels, 
                                                                    t_bin, func, t_column)
        
        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
        if facet_col is None:
            facet_col = ''
            col_uniform = True # make it use the state colours if no facet

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 1.01], t0 = 0, dtick = 0.2, ylabel = 'Response Rate', title = title, grid = grids)


        for c, (col, state) in enumerate(zip(colours, labels)):

            sub_df = grouped_data[grouped_data['state'] == state]

            for lab in h_order:

                sub_np = sub_df[plot_column][sub_df[facet_col] == lab].to_numpy()
                if len(sub_np) == 0: continue # normally used when no Spon. Mov. is present

                mean, median, q3, q1, zlist = self._zscore_bootstrap(sub_np, z_score = z_score)

                if col_uniform == True:
                    if 'Spon. Mov.' in lab:
                        marker_col = palette_dict[lab]
                    else:
                        marker_col = col
                else:
                    marker_col = palette_dict[lab]

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [lab], colour =  marker_col, showlegend = True, name = lab, xaxis = f'x{c+1}'))

                label_list = [lab] * len(zlist)
                fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                showlegend = False, name = lab, xaxis = f'x{c+1}'))

            axis = f'xaxis{c+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = state, domains = domains[c:c+2], axis = axis)
            # I don't know why the above plots the x-labels at the top. Will just have to leave it

        fig.update_xaxes(showticklabels=False) # remove labels as they're too bulky
        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)
        grouped_data = grouped_data.drop(columns = ['previous_state'])        

        return fig, grouped_data

    def plot_response_over_hmm_bouts(self, mov_df, hmm, variable:str = 'moving', response_col:str = 'has_responded', labels:list = None, colours:list = None, 
        x_limit:int = 30, t_bin:int = 60, func:str = 'max', title:str = '', grids:bool = False, t_column:str = 't'):
        """ 
        Generates a plot showing the response rate per time stamp in each HMM bout. Y-axis is between 0-1 and the response rate, the x-axis is the time point
        in each state as per the time the dataset is binned to when decoded.

            Args:
                mov_df (behavpy dataframe): The matching behavpy dataframe containing the movement data from the response experiment
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the 
                    correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                response_col (str, optional): The name of the coloumn that has the responses per interaction. 
                    Must be a column of bools. Default is 'has_responded'.
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. 
                    If None and not 4 states then generic labels are generated, i.e. 'state-1, state-2, state-n'.
                    Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. 
                    If None the colours are by default for 4 states (blue and red), if not 4 then colours from the palette are chosen. 
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn. Default is None.
                x_limit (int, optional): A number to limit the x-axis by to remove outliers, i.e. 30 would be 30 minutes or less if t_bin is 60. 
                    Default 30.
                t_bin (int, optional): The time in seconds to bin the time series data to. Default is 60,
                func (str, optional): When binning the time what function to apply the variable column. Default is 'max'.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
        Note:
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function.
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)

        grouped_data, palette_dict, h_order = self._bouts_response(mov_df=mov_df, hmm=hmm, variable=variable, response_col=response_col, labels=labels, colours=colours, 
                                            x_limit=x_limit, t_bin=t_bin, func=func, t_col=t_column)

        # create and style plot
        fig = go.Figure() 
        self._plot_ylayout(fig, yrange = [0, 1], t0 = 0, dtick = 0.2, ylabel = 'Response Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = t_bin/60, xlabel = f'Consecutive minutes in state')

        for hue in h_order:
            sub_df = grouped_data[grouped_data['label_col'] == hue]

            upper, trace, lower = self._plot_line(df = sub_df, x_col = 'previous_activity_count', name = hue, marker_col = palette_dict[hue])
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)
        
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
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from math import floor
from functools import partial

from ethoscopy.behavpy_draw import behavpy_draw

from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.misc.hmm_functions import hmm_pct_state

class behavpy_seaborn(behavpy_draw):
    """
    Extends behavpy_draw to make use of data analysis for plotting with Seaborn, a plotting library for Python built upon matplotlib.

    Both behavpy_seaborn and behavpy_plotly are two mirrors of each other, containing the same methods (except make_tile).
    However, there are some differences in the way the data is processed and plotted, which is outlined at the 
    appropriate methods. One consistent difference is that quantification plots with plotly give the option to filter
    the data by a z-score threshold which is not available in seaborn. Addtionally 95% CI are calculated using an
    internal function, whereas seaborn uses its own version.
    
    The behavpy_seaborn class is not called directly by the user, however it is called in the generator class behavpy, and as
    such this class is directly accessible to the user.

    Attributes:
        canvas (str): tells the inherited classes that this is a seaborn class

    **Example Usage:**
        df = behavpy(behavior_data, behavior_meta, check = True,canvas = 'seaborn')
        fig = df.heatmap(variable = 'asleep')
    """

    canvas = 'seaborn'
    error = 'se'

    def heatmap(self, variable:str, t_column:str = 't', title = '', lights_off:int|float = 12, figsize:tuple = (0,0)):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals. 
        This is a great tool to quickly identify changes in the variable of choice over the course of the experiment.
        
        Args:
            variable (str): The name for the column containing the variable of interest
            t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
            title (str, optional): The title of the plot. Default is an empty string.
            lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. 
                Must be number between 0 and day_lenght. Default is 12.
            figsize (tuple, optional): The size of the figure. Default is (0, 0) which auto-adjusts the size. Default is (0,0).
        
        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.

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
            figsize=(10, 6)
        );
        """
        
        data, time_list, id = self.heatmap_dataset(variable, t_column)

        data = pd.DataFrame(data.tolist())

        # n = 12
        # t_min = int(n * floor(time_list.min() / n))
        # t_max = int(n * ceil(time_list.max() / n)) 

        # Set every nth x-tick label, in this example every 12th label
        n = lights_off
        # x_labels = time_list[::n].astype(int)  # Get every nth label
        x_ticks = np.arange(0, len(time_list), n)  # Get every nth location

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (0.5*len(x_ticks), 0.1*len(id))

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, cmap="viridis", ax=ax)


        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)
        plt.xlabel("ZT Time (Hours)")

        plt.yticks(ticks=np.arange(0, len(id), 2), labels=id[::-2], rotation=0)

        if title: fig.suptitle(title, fontsize=16, y = 1)

        return fig

    def plot_overtime(self, variable:str, wrapped:bool = False, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, 
                        avg_window:int = 30, day_length:int|float = 24, lights_off:int|float  = 12, title:str = '', grids:bool = False, t_column:str = 't', 
                        col_list:list|None = None, figsize:tuple = (0,0)):
        """
        Generate a line plot to visualize the progression of a specified variable over the duration of an experiment or within an experimental day. 
        This plot displays the mean values along with 95% confidence intervals for each group, providing clear insights into temporal trends and 
        group comparisons.
        Additionally, the plot includes visual indicators (white and black boxes) to denote col = np.where(st == c, colours[c], np.NaN)ds when lights are on and off, which can be customized 
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
            figsize (tuple, optional): The size of the figure as `(width, height)`. If set to `(0, 0)`, the size is auto-adjusted. Defaults to `(0, 0)`.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.

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
        """

        # If facet_col is provided but facet arg is not, will automatically fill facet_arg and facet_labels with all the possible values
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]

        if col_list is None:
            col_list = self._get_colours(d_list)
        else:
            if len(col_list) != len(facet_labels):
                raise ValueError("Given col_list is of different length to the facet_arg list")

        min_t = []
        max_t = []

        fig, ax = plt.subplots(figsize=figsize)

        for data, name, col in zip(d_list, facet_labels, col_list):

            gb_df, t_min, t_max, col, _ = self._generate_overtime_plot(
                                            data = data, name = name, col = col, var = variable, 
                                            avg_win = int((avg_window * 60)/self[t_column].diff().median()),
                                            wrap = wrapped, day_len = day_length, light_off= lights_off, t_col = t_column)

            if gb_df is None:
                continue

            plt.plot(gb_df[t_column], gb_df["mean"], label=name, color=col)
            plt.fill_between(
            gb_df[t_column], gb_df["y_min"], gb_df["y_max"], alpha = 0.25, color=col
            )

            min_t.append(t_min)
            max_t.append(t_max)

        if lights_off%2 == 0:
            x_ticks = np.arange(np.min(min_t), np.max(max_t), lights_off/2, dtype = int)
        else:
            x_ticks = np.arange(np.min(min_t), np.max(max_t), lights_off/2)

        if figsize == (0,0):
            figsize = ( 6 + 1/3 * len(x_ticks), 
                        4 + 1/24 * len(x_ticks) 
                        )
            fig.set_size_inches(figsize)

        # Customise legend values
        if facet_labels[0] != '':
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)

        yr, dt =  self._check_boolean(data[variable].tolist())
        if yr is not False:
            plt.ylim(0, yr[1])
        ymin, ymax = ax.get_ylim()
        plt.xlim(t_min, t_max)

        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)
        plt.xlabel("ZT Time (Hours)")
        plt.ylabel(variable)
        if facet_col: plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) # legend outside of area if faceted 
        plt.title(title)

        if grids:
            plt.grid(axis='y')

        # For every 24 hours, draw a rectangle from 0-12 (daytime) and another from 12-24 (nighttime)
        bar_range, thickness = circadian_bars(t_min, t_max, min_y = ymin, max_y = ymax, day_length = day_length, 
                                              lights_off = lights_off, canvas = 'seaborn')
        # lower range of y-axis to make room for the bars 
        ax.set_ylim(ymin-thickness, ymax)

        # iterate over the bars and add them to the plot
        for i in bar_range:
            # Daytime patch
            if i % day_length == 0:
                ax.add_patch(mpatches.Rectangle((i, ymin-thickness), lights_off, thickness, color='black', alpha=0.4, clip_on=False, fill=None))
            else:
                # Nighttime patch
                ax.add_patch(mpatches.Rectangle((i, ymin-thickness), day_length-lights_off, thickness, color='black', alpha=0.8, clip_on=False))

        return fig

    def plot_quantify(self, variable:str, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, fun:str = 'mean', 
                      title:str = '', grids:bool = False, figsize:tuple = (0,0)):
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
            grids (bool, optional): Determines whether grid lines are displayed on the plot. Defaults to `False`.
            figsize (tuple, optional): The size of the figure as `(width, height)`. If set to `(0, 0)`, the size is auto-adjusted. Defaults to `(0, 0)`.


        Returns:
            tuple:
                - fig (matplotlib.figure.Figure): The generated matplotlib figure object of the quantification plot.
                - data (pandas.DataFrame): A DataFrame containing the grouped and aggregated data based on the input parameters.

        Raises:
            ValueError: If `fun` is not a valid function or 'facet_col' is not a valid column in the metadata.

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
        );
        # Display the plot
        fig          
        """

        grouped_data, palette_dict, facet_labels, variable = self._internal_plot_quantify(variable, facet_col, facet_arg, facet_labels, fun)

        # BOXPLOT
        fig_rows = len(variable)
        fig_cols = 1

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (2*len(facet_labels), 4*fig_rows)

        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize)

        # axes is
        #  matplotlib.axes._axes.Axes if only one subplot
        #  numpy.ndarray if multiple subplots
        # Flatten the axes list, in case we have more than one row
        if fig_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for ax, var in zip(axes, variable):
        
            plot_column = f'{var}_{fun}'

            y_range, dtick = self._check_boolean(list(self[var]))

            if y_range: 
                ax.set_ylim(y_range)

            if facet_col:
                
                sns.stripplot(data=grouped_data, x=facet_col, y=plot_column, order = facet_labels, hue=facet_col, palette=palette_dict, ax=ax, alpha=0.5, legend=False,)
                sns.pointplot(data=grouped_data, x=facet_col, y=plot_column, order = facet_labels, hue=facet_col, palette=palette_dict, ax=ax, estimator = 'mean',
                                linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

                ## ax.set_xticklabels(facet_labels)
                # Customise legend values
                # handles, _ = ax.get_legend_handles_labels()
                # ax.legend(labels=facet_labels)

            else:
                sns.stripplot(data=grouped_data, y=plot_column, color = list(palette_dict.values())[0], ax=ax, alpha=0.5, legend=False,)
                sns.pointplot(data=grouped_data, y=plot_column, color = list(palette_dict.values())[0], ax=ax, estimator = 'mean',
                                linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

            ax.set_ylabel(var)

        if grids: plt.grid(axis='y')
        plt.title(title)
        plt.tight_layout()

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_compare_variables(self, variables:list, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None,
                                fun:str = 'mean', title:str = '', grids:bool = False, figsize:tuple = (0,0)):
        """ 
        Create multiple sets of quantification plots of each variable to compare. 
        When faceting by a group, each variable is plotted on its own figure, unlike the Plotly version which plots all variables on the same figure.

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
            grids=True
        );
        # Display the plot
        fig
        """

        if isinstance(variables, list) is False:
            raise TypeError('Argument variables must be given as a list')

        return self.plot_quantify(variable = variables, facet_col = facet_col, facet_arg = facet_arg, facet_labels = facet_labels, fun = fun, title = title, grids = grids, figsize=figsize)

    def plot_day_night(self, variable:str, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, day_length:int|float = 24, 
        lights_off:int|float = 12, title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple = (0,0)):
        """
        A plot that shows the average of a variable split between the day (lights on) and night (lights off). This a specific version of plot_quantify, using similar backend.

            Args:
                variable (str): The name of the column you wish to plot from your data. 
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. 
                    Default is None.
                fun (str, optional): The average function that is applied to the data. Must be one of 'mean', 'median', 'count'.
                day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
                lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. 
                    Default is 12.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False
                figsize (tuple, optional): The size of the figure. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.
        
        Note:


        """

        grouped_data, palette_dict, facet_labels = self._internal_plot_day_night(variable, facet_col, facet_arg, facet_labels, day_length, lights_off, t_column)
        plot_column = f'{variable}_mean'

        # BOXPLOT
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_labels)+2, 4+2)

        fig, ax = plt.subplots(figsize=figsize)

        y_range, dtick = self._check_boolean(list(self[variable]))
        if y_range:
            plt.ylim(y_range)

        if facet_col:
            sns.stripplot(data=grouped_data, x='phase', y=plot_column, order=['light', 'dark'], hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge = 0.8 - 0.8 / len(facet_labels))
            sns.pointplot(data=grouped_data, x='phase', y=plot_column, order=['light', 'dark'], hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.8 - 0.8 / len(facet_labels))

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)

        else:
            sns.stripplot(data=grouped_data, x='phase', y=plot_column, order=['light', 'dark'], ax=ax, hue ='phase', palette={"light" : "gold", "dark" : "darkgrey"}, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='phase', y=plot_column, order=['light', 'dark'], ax=ax, hue ='phase', palette={"light" : "gold", "dark" : "darkgrey"}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)
        
        ax.set_ylabel(variable)
        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_anticipation_score(self, variable:str, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, day_length:int|float = 24, 
        lights_off:int|float = 12, title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple = (0,0)):
        """
        Plots the anticipation scores for lights on and off periods. The anticipation score is calculated as the percentage of activity of the 6 hours prior 
        to lights on/off that occurs in the last 3 hours. A higher score towards 100 indicates greater anticipation of the light change.

            Args:
                variable (str): The name of the column containing the variable that measures activity.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. 
                    Default is None.
                day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
                lights_off (int, optional): The time point when the lights are turned off in an experimental day, 
                    assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False
                figsize (tuple, optional): The size of the figure. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.
        """

        grouped_data, palette_dict, facet_labels = self._internal_plot_anticipation_score(variable, facet_col, facet_arg, facet_labels, 
                                                                                            day_length, lights_off, t_column)

        # BOXPLOT
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_labels)+2, 6+2)

        fig, ax = plt.subplots(figsize=figsize)

        if facet_col:

            sns.stripplot(data=grouped_data, x='phase', y="anticipation_score", order=['Lights On', 'Lights Off'], ax=ax, hue =facet_col, hue_order=facet_labels, palette=palette_dict, alpha=0.5, legend=False, dodge = 0.8 - 0.8 / len(facet_labels))
            sns.pointplot(data=grouped_data, x='phase', y="anticipation_score", order=['Lights On', 'Lights Off'], ax=ax, hue =facet_col, hue_order=facet_labels, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge =  0.8 - 0.8 / len(facet_labels))

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)

        else:
            sns.stripplot(data=grouped_data, x='phase', y="anticipation_score", order=['Lights On', 'Lights Off'], ax=ax, hue ='phase', palette={"Lights On" : "gold", "Lights Off" : "darkgrey"}, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='phase', y="anticipation_score", order=['Lights On', 'Lights Off'], ax=ax, hue ='phase', palette={"Lights On" : "gold", "Lights Off" : "darkgrey"}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        if grids: plt.grid(axis='y')
        plt.title(title)
        plt.tight_layout()
        ax.set_ylabel('Anticipatory Phase Score')

        # The score is in %
        plt.ylim(0,100)

        # reorder dataframe for stats output
        if facet_col: grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data
    
    @staticmethod
    def _plot_single_actogram(dt, figsize, days, title, day_length, size):

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (10, len(days)/2)

        fig, axes = plt.subplots(len(days)-1, 1, figsize=figsize, sharex=True)
        axes[0].set_title(title, size = size)

        for ax, day in zip(axes, days[:-1]):

            subset = dt[dt['day'].isin([day, day+1])].copy()
            subset.loc[subset['day'] == day+1, 'hours'] += 24

            # Remove x and y axis labels and ticks
            ax.set(yticklabels=[])
            ax.tick_params(axis='both', which='both', length=0)

            #ax.step(subset["hours"], subset["moving_mean"], alpha=0.5) 
            ax.fill_between(subset["hours"], subset["moving_mean"], step="pre", color="black", alpha=1.0)

        plt.xticks(range(0, day_length*2+1, int(day_length*2/8)))
        plt.xlim(0,48)
        plt.xlabel("ZT (Hours)")
        #plt.tight_layout()

        sns.despine(left=True, bottom=True)
        return fig 

    @staticmethod
    def _internal_actogram(data, mov_variable, bin_window, t_column, facet_col):
        """
        An internal function to augment and setup the data for plotting actograms
        """

        data = data.bin_time(mov_variable, bin_window*60, t_column = t_column)
        data.add_day_phase(t_column = f'{t_column}_bin')
        days = data["day"].unique()

        data = data.merge(data.meta, left_index=True, right_index=True)

        data["hours"] = (data[f'{t_column}_bin'] / (60*60))
        data["hours"] = data["hours"] - (data["day"]*24)
        data.reset_index(inplace = True)
        if facet_col:
            data = data.groupby([f'{t_column}_bin', facet_col]).agg(**{
                'moving_mean' : ('moving_mean', 'mean'),
                'day' : ('day', 'max'),
                'hours': ('hours', 'max')

            }).reset_index()
        else:
            data = data.groupby(f'{t_column}_bin').agg(**{
                'moving_mean' : ('moving_mean', 'mean'),
                'day' : ('day', 'max'),
                'hours': ('hours', 'max')

            }).reset_index()
        return data, days

    def plot_actogram(self, mov_variable:str = 'moving', bin_window:int = 5, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, 
        day_length:int|float = 24, t_column:str = 't', title:str = '', figsize:tuple=(0,0)):
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
                figsize (tuple, optional): The size of the figure to be plotted as (width, height). If set to 
                    (0,0), the size is determined automatically. Default is (0,0).

        Returns:
            matplotlib.figure.Figure: If facet_col is provided, returns a figure that contains subplots for each 
            facet. If facet_col is not provided, returns a single actogram plot.

        Example:
            >>> instance.plot_actogram(mov_variable='movement', bin_window=10, 
            ...                        t_column='time', facet_col='activity_type')
        """

        # If facet_col is provided but facet arg is not, will automatically fill facet_arg and facet_labels with all the possible values
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        # call the internal actogram augmentor
        data, days = self._internal_actogram(data, mov_variable, bin_window, t_column, facet_col)

        if facet_col:

            sub_title_size = 20


            figs = []
            for subplot in facet_arg:

                dt = data.loc [data[facet_col] == subplot]
                title = "%s - %s" % (title, subplot)
                fig = self._plot_single_actogram(dt, figsize, days, title, day_length, sub_title_size)
                plt.close()

                figs.append(fig)

            
            # Create a new figure to combine the figures
            cols, rows = 3, -(-len(facet_arg) // 3)
            c = []

            if figsize == (0,0):
                figsize = (16*rows, 2*len(days))

            combined_fig = plt.figure(figsize = figsize )
            
            for pos, f in enumerate(figs):

                c.append( combined_fig.add_subplot(rows, cols, pos+1))
                c[-1].axis('off')  # Turn off axis
                c[-1].imshow( self._fig2img (f) )

            # Adjust the layout of the subplots in the combined figure
            #combined_fig.tight_layout()

            return combined_fig

        else:
            sub_title_size = 25
            return self._plot_single_actogram(data, figsize, days, title, day_length, sub_title_size)

    def plot_actogram_tile(self, mov_variable:str = 'moving', labels:None|str = None, bin_window:int = 15, day_length:int = 24, t_column:str = 't', 
        title:str = '', figsize:tuple = (0,0)):
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
                figsize (tuple, optional): The size of the figure to be plotted as (width, height). If set to 
                    (0,0), the size is determined automatically. Default is (0,0).

        Returns:
            matplotlib.figure.Figure: If facet_col is provided, returns a figure that contains subplots for each 
            facet. If facet_col is not provided, returns a single actogram plot.

        Raises:
            ValueError: If facet_arg is provided but facet_col is None.
            SomeOtherException: If some other condition is met.

        Example:
            >>> instance.plot_actogram_tile(mov_variable='movement', bin_window=10, 
            ...                        t_column='time', facet_col='activity_type')
        """

        # If there are no lablels then populate with index IDs
        if labels is not None:
            if labels not in self.meta.columns.tolist():
                raise KeyError(f'{labels} is not a column in the metadata')
            title_list = self.meta[labels].tolist() 
        else:
            title_list = self.meta.index.tolist()

        facet_arg = self.meta.index.tolist()

        # call the internal actogram augmentor
        data, days = self._internal_actogram(self, mov_variable, bin_window, t_column, facet_col='id')

        # get the nearest square number to make a grid plot
        root =  self._get_subplots(len(title_list))
        sub_title_size = 30

        figs = []
        for subplot, label in zip(facet_arg, title_list):

            dt = data.loc[data['id'] == subplot]
            subtitle = "%s" % (label)        
            fig = self._plot_single_actogram(dt, figsize, days, subtitle, day_length, sub_title_size)
            plt.close()

            figs.append(fig)

        
        # Create a new figure to combine the figures
        cols, rows = root, root
        c = []

        if figsize == (0,0):
            figsize = (6*rows, 2*len(days))

        combined_fig = plt.figure(figsize = figsize )
        for pos, f in enumerate(figs):

            c.append( combined_fig.add_subplot(rows, cols, pos+1))
            c[-1].axis('off')  # Turn off axis
            c[-1].imshow( self._fig2img (f) )

        combined_fig.suptitle(title, size = 5.5*root, y=1)

        # Adjust the layout of the subplots in the combined figure
        #combined_fig.tight_layout()

        return combined_fig

    def survival_plot(self, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, repeat:bool|str = False, day_length:int = 24, 
        lights_off:int = 12, title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple = (0,0)):
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
            fig (matplotlib.figure.Figure): Figure object of the plot.
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

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = ( 6 + 1/4 * len(x_ticks), 
                        4 + 1/32 * len(x_ticks) 
                        )

        fig, ax = plt.subplots(figsize=figsize)
        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        # sns.set_style("ticks")
        if facet_col is not None:
            map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
            sur_df['label'] = sur_df['label'].map(map_dict)
            sns.lineplot(data = sur_df, x = "hour", y = "survived", hue = 'label', hue_order = facet_labels, ax=ax, palette = palette_dict, errorbar = self.error)
        else:
            sns.lineplot(data = sur_df, x = "hour", y = "survived", ax=ax, palette = palette, errorbar = self.error)

        plt.xlim(np.min(x_ticks), np.max(x_ticks))
        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)

        plt.ylabel("Survival (%)")
        plt.xlabel("ZT (Hours)")
        plt.title(title)

        if facet_col: plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) # legend outside of area if faceted 

        if grids:
            plt.grid(axis='y')
        ymin, ymax = 0, 101
        bar_range, thickness = circadian_bars(0, sur_df['hour'].max(), min_y = ymin, max_y = ymax, day_length = day_length, lights_off = lights_off, canvas = 'seaborn')
        # lower range of y-axis to make room for the bars 
        ax.set_ylim(ymin-thickness, ymax)
        # For every 24 hours, draw a rectangle from 0-12 (daytime) and another from 12-24 (nighttime)
        for i in bar_range:
            # Daytime patch
            if i % day_length == 0:
                ax.add_patch(mpatches.Rectangle((i, ymin-thickness), lights_off, thickness, color='black', alpha=0.4, clip_on=False, fill=None))
            else:
                # Nighttime patch
                ax.add_patch(mpatches.Rectangle((i, ymin-thickness), day_length-lights_off, thickness, color='black', alpha=0.8, clip_on=False))

        return fig

    # Response AGO/mAGO section

    def plot_response_quantify(self, response_col:str = 'has_responded', facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, title:str = '', 
        grids:bool = False, figsize:tuple = (0,0)):
        """ 
        A plotting function for AGO or mAGO datasets that have been loaded with the analysing function stimulus_response.
        A augmented version of plot_quanitfy that finds the average response to a stimulus and the average response
        from a mock stimulus. Must contain the column 'has_interacted' with 1 = True stimulus, 2 = Mock stimulus.
        
            Args:
                response_col = string, the name of the column in the data with the response per interaction, column data should be in boolean form
                facet_col = string, the name of the column in the metadata you wish to filter the data by
                facet_arg = list, if not None then a list of items from the column given in facet_col that you wish to be plotted
                facet_labels = list, if not None then a list of label names for facet_arg. If not provided then facet_arg items are used
                title = string, a title for the plotted figure
                grids = bool, true/false whether the resulting figure should have grids
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.
        Notes:
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function or one that mimics it.
        """

        grouped_data, h_order, palette_dict =  self._internal_plot_response_quantify(response_col, facet_col, facet_arg, facet_labels) 
        plot_column = f'{response_col}_mean'

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (1.5*len(h_order), 10)

        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim(0, 1.01)
        plt.ylabel("Response rate")

        if facet_col:
            # this one splits by the original facet_col and then plots thw conjoined order. However, they sit on top of each other and normal dodging doesn't work.
            # sns.stripplot(data=grouped_data, x=facet_col, y=plot_column, order=facet_labels, hue='facet_col', hue_order=h_order, ax=ax, palette=palette_dict, alpha=0.5, legend=False)
            # sns.pointplot(data=grouped_data, x=facet_col, y=plot_column, order=facet_labels, hue='facet_col', hue_order=h_order, ax=ax, palette=palette_dict, estimator = 'mean',
            #                 linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

            sns.stripplot(data=grouped_data, x='facet_col', y=plot_column, order=h_order, hue='facet_col', hue_order=h_order, ax=ax, palette=palette_dict, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='facet_col', y=plot_column, order=h_order, hue='facet_col', hue_order=h_order, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, legend=True)
            ax.set(xticklabels=[])

            # Customise legend values
            # handles, _ = ax.get_legend_handles_labels()
            # ax.legend(handles=handles, labels=h_order)

            plt.xticks(rotation=45)


        else:
            sns.stripplot(data=grouped_data, x='facet_col', y=plot_column, order=h_order, ax=ax, hue = 'facet_col', hue_order = h_order, palette=palette_dict, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='facet_col', y=plot_column, order=h_order, ax=ax, hue = 'facet_col', hue_order = h_order, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)
        
        plt.xlabel(facet_col)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data[facet_col] = grouped_data['facet_col']
            grouped_data = grouped_data[[plot_column, f'{response_col}_std', facet_col]].sort_values(facet_col)

        return fig, grouped_data

    def plot_habituation(self, plot_type:str, t_bin_hours:int = 1, response_col:str = 'has_responded', interaction_id_col:str = 'has_interacted', stim_count:bool = True, 
        facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None,  x_limit:bool = False, t_column:str = 't', title:str = '', grids:bool = False, figsize:tuple = (0,0)):
        """
        Generate a plot which shows how the response response rate changes over either repeated stimuli (number) or hours post first stimuli (time).
        If false stimuli are given and represented in the interaction_id column, they will be plotted seperately in grey.

            Args:
                plot_type (str): The type of habituation being plotter, either 'number' (the response rate for every stimuli in sequence, i.e. 1st, 2nd, 3rd, ..)
                    or 'time' (the response rate per hour(s) post the first stimuli.)
                t_bin_hours (int, optional): The number of hours you want to bin the response rate to. Default is 1 (hour).
                response_col (str, optional): The name of the coloumn that has the responses per interaction. Must be a column of bools. Default is 'has_responded'.
                interaction_id_col (str, optional): The name of the column conataining the id for the interaction type,
                    which should be either 1 (true interaction) or 2 (false interaction). Default 'has_interacted'.
                stim_count (bool, optional): If True statistics for the stimuli are plotted on the secondary y_axis. For 'number' the percentage of specimen revieving
                    that number of stimuli is plotted. If 'time', the raw number of stimuli per hour(s) is plotted. False Stimuli are discarded. Default is True
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. 
                    Default is None.
                x_limit (int, optional): A number to limit the x-axis by to remove outliers, i.e. 50 would be 50 stimuli for 'number'. Default False.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
            
        Notes:
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function. Contain columns such as 'has_responded' and 'has_interacted'.
            The stimulus count plot only calculates the percentage or total for the true stimulus, discarding the false
                for visual clarity.
        """  

        seconday_label = {'time' : f'No. of stimulus (absolute)', 'number' : '% recieving stimulus'}

        # call the internal method to curate and analse data, see behavpy_draw
        grouped_data, h_order, palette_dict, x_max, plot_choice = self._internal_plot_habituation(plot_type=plot_type, t_bin_hours=t_bin_hours, response_col=response_col, 
                                                                                    interaction_id_col=interaction_id_col, facet_col=facet_col, facet_arg=facet_arg, 
                                                                                    facet_labels=facet_labels, x_limit=x_limit, t_column=t_column)

        fig, ax = plt.subplots(figsize=figsize)
        if stim_count is True:
            ax2 = ax.twinx()  
        ax.set_ylim([0, 1.01])
        plt.xlim([0, x_max])

        if figsize == (0,0):
            figsize = ( 6 + 1/2 * x_max, 
                        8
                        )
            fig.set_size_inches(figsize)

        for hue in h_order:
            sub_df = grouped_data[grouped_data.index == hue]
            
            if len(sub_df) == 0:
                continue

            ax.plot(sub_df[plot_choice], sub_df["mean"], label = hue, color = palette_dict[hue])
            ax.fill_between(
            sub_df[plot_choice], sub_df["y_min"], sub_df["y_max"], alpha = 0.25, color = palette_dict[hue]
            )

            if stim_count is True and '-True Stimulus' in hue:
                if plot_type == 'number':
                    sub_df['count'] = (sub_df['count'] / np.max(sub_df['count'])) * 100
                else:
                    sub_df['count'] = sub_df['stim_count']

                ax2.plot(sub_df[plot_choice], sub_df["count"], label = hue, color = palette_dict[hue], linestyle='--')

        # # Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles)

        ax.set_xlabel(plot_choice)
        ax.set_ylabel("Response Rate")
        if facet_col: ax.legend(bbox_to_anchor=(1.06, 1), loc='upper left', borderaxespad=0) # legend outside of area if faceted 

        if stim_count is True:
            if plot_type == 'time':
                ax2.autoscale(axis = 'y')
            else:
                ax2.set_ylim([0,101])
            ax2.set_ylabel(seconday_label[plot_type])

        plt.title(title)

        if grids:
            ax.grid(axis='y')

        return fig

    def plot_response_over_activity(self, mov_df, activity:str, variable:str = 'moving', response_col:str = 'has_responded', facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, 
        x_limit:int = 30, t_bin:int = 60, title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple = (0,0)):
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
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
            
        Notes:
            This plotting method can show the response rate for both active and inactive bouts for the 
                whole dataset, but only for one or the other if you want to facet by a column, i.e. facet_col.
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function or one that mimics it.
        """        

        # call the internal method to curate and analse data, see behavpy_draw
        grouped_data, h_order, palette_dict = self._internal_bout_activity(mov_df=mov_df, activity=activity, variable=variable, response_col=response_col, 
                                    facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, x_limit=x_limit, t_bin=t_bin, t_column=t_column)
                                    
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim([0, 1.01])
        plt.xlim([1, x_limit])
        x_ticks = np.arange(1, x_limit+1, 1, dtype = int)

        if figsize == (0,0):
            figsize = ( 6 + 1/2 * len(x_ticks), 
                        8
                        )
            fig.set_size_inches(figsize)

        for hue in h_order:

            sub_df = grouped_data[grouped_data['label_col'] == hue]
            # if no data, such as no false stimuli, skip the plotting
            if len(sub_df) == 0:
                continue
            plt.plot(sub_df["previous_activity_count"], sub_df["mean"], label = hue, color = palette_dict[hue])
            plt.fill_between(
            sub_df["previous_activity_count"], sub_df["y_min"], sub_df["y_max"], alpha = 0.25, color = palette_dict[hue]
            )

        # # Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles)#, labels=h_order)

        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)
        plt.xlabel(f'Consecutive minutes in behavioural bout ({activity})')
        plt.ylabel("Response rate")

        plt.title(title)

        if grids:
            plt.grid(axis='y')

        return fig

    def plot_response_overtime(self, t_bin_hours:int = 1, wrapped:bool = False, response_col:str = 'has_responded', interaction_id_col:str = 'has_interacted', facet_col:None|str = None, facet_arg:None|str = None, 
        facet_labels:None|str = None, day_length:int = 24, lights_off:int = 12, func:str = 'mean', t_column:str = 't', title:str = '', grids:bool = False, figsize:tuple = (0,0)):
        """ 
        Generate a plot which shows how the response response rate changes over a day (wrapped) or the course of the experiment.
        If false stimuli are given and represented in the interaction_id column, they will be plotted seperately in grey.

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
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
            
        Notes:
            This function must be called on a behavpy dataframe that is populated with data loaded with the stimulus_response
                analysing function. Contain columns such as 'has_responded' and 'has_interacted'.
        """  
        df, h_order, palette = self._internal_plot_response_overtime(t_bin_hours=t_bin_hours, response_col=response_col, interaction_id_col=interaction_id_col, 
                                                facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, func=func, t_column=t_column)

        return df.plot_overtime(variable='Response Rate', wrapped=wrapped, facet_col='new_facet', facet_arg=h_order, facet_labels=h_order,
                                avg_window=5, day_length=day_length, lights_off=lights_off, title=title, grids=grids, t_column='t_bin', 
                                col_list = palette, figsize=(0,0))

    # Seaborn Periodograms

    def plot_periodogram_tile(self, labels = None, find_peaks = False, title = '', grids = False, figsize=(0,0)):
        """ 
        Generates a periodogram plot for every specimen in the metdata. Periodograms show the power of each frequency (rythmn) within the data. 
        For a normal specimen a peak in power at 24 hours is expected. A threshold line (in red) is also plotted, power above this line is 
        significant given the alpha value when calculating frequency power with .periodogram().

            Args:
                label (str, optional): The name of the column in the metadata that contains unique labels per specimen. If None then
                    the index ids are used. Default is None.
                find_peaks (bool, optional): If True then the highest frequency that is signifcant is found and marked with an X.
                    Default is False.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure to be plotted as (width, height). If set to 
                    (0,0), the size is determined automatically. Default is (0,0).
        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
        Raises:
            AttributeError is this method is called on a beahvpy object that has not been augmented by .periodogram()
                or doesn't have the columns "power" or "period".
        """
        self._validate()

        if labels is not None:
            if labels not in self.meta.columns.tolist():
                raise KeyError(f'{labels} is not a column in the metadata')
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

        # get the nearest square number to make a grid plot
        root =  self._get_subplots(len(data.meta))
        col_list = list(range(0, root)) * root
        row_list = list([i] * root for i in range(0, root))
        row_list = [item for sublist in row_list for item in sublist]
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (2*len(facet_arg), 4*root)
        # create the subplot
        fig, axes = plt.subplots(root, root, figsize=figsize)

        for subplot, col, row, label in zip(facet_arg, col_list, row_list, title_list): 
            # filter by index for subplot
            dt = data.loc[data.index == subplot]
            # plot the power (blue) and signigicant threshold (red)
            axes[row, col].plot(dt['period'], dt['power'], color = 'blue')
            axes[row, col].plot(dt['period'], dt['sig_threshold'], color = 'red')
            axes[row, col].set_title(label)

            if plot_peaks is True:                
                tdf = dt[dt['peak'] == 1]
                axes[row, col].plot(tdf['period'], tdf['power'], marker='x', markersize=40, color="black")

        for ax in axes.flat:
            ax.set(xlabel='Period (Hours)', ylabel='Power')
        fig.suptitle(title, size = 7*root, y=1)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axes.flat:
            ax.label_outer()

        return fig

    def plot_periodogram(self, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, title:str = '', grids:bool = False, figsize:tuple=(0,0)):
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
                figsize (tuple, optional): The size of the figure to be plotted as (width, height). If set to 
                    (0,0), the size is determined automatically. Default is (0,0).
        
        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
        Raises:
            AttributeError is this method is called on a beahvpy object that has not been augmented by .periodogram()
                or doesn't have the columns "power" or "period".
        """

        # check if the dataset has the needed columns from .periodogram()
        self._validate()
        # check the facet_col and args are in the dataset, populate if not
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col is not None:
            d_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            d_list = [self.copy(deep = True)]

        power_var = 'power'
        period_var = 'period'

        fig, ax = plt.subplots(figsize=figsize)

        for data, name, col in zip(d_list, facet_labels, self._get_colours(d_list)):

            gb_df, _, _, col, _ = self._generate_overtime_plot(data = data, name = name, col = col, var = power_var, 
                                                                                    avg_win = False, wrap = False, day_len = False, 
                                                                                    light_off= False, t_col = period_var)
            if gb_df is None:
                continue

            plt.plot(gb_df[period_var], gb_df["mean"], label=name, color=col)
            plt.fill_between(
            gb_df[period_var], gb_df["y_min"], gb_df["y_max"], alpha = 0.25, color=col
            )
        
        x_ticks = np.arange(0,36*6,6, dtype=int)

        if figsize == (0,0):
            figsize = ( 6 + 1/3 * len(x_ticks), 
                        4 + 1/24 * len(x_ticks) 
                        )
            fig.set_size_inches(figsize)

        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)
        plt.xlim(np.nanmin(self[period_var]), np.nanmax(self[period_var]))
        plt.xlabel('Period Frequency (Hours)')
        plt.ylabel(power_var)

        plt.title(title)

        if grids:
            plt.grid(axis='y')

        return fig
    
    def plot_periodogram_quantify(self, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, title:str = '', 
        grids:bool = False, figsize:tuple=(0,0)):
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
            fig (matplotlib.figure.Figure): Figure object of the plot.
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

        # call the plot quantify method
        return self.plot_quantify(variable = y_label, facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, 
                                    fun='max', title=title, grids=grids, figsize=figsize)

    def plot_wavelet(self, mov_variable:str, sampling_rate:int = 15, scale:int = 156, wavelet_type:str = 'morl', t_col:str = 't', title:str = '', grids:bool = False, figsize:tuple = (0,0)):
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
            fig (matplotlib.figure.Figure): Figure object of the plot.
        """
        # format the data for the wavelet function
        fun, avg_data = self._format_wavelet(mov_variable, sampling_rate, wavelet_type, t_col)
        # call the wavelet function
        t, per, power = fun(avg_data, t_col = t_col, var = mov_variable, scale = scale, wavelet_type = wavelet_type)

        # fig, ax = plt.subplots(figsize=figsize)
        fig, ax = plt.subplots()

        # set contour levels between -3 and 3
        levels = np.arange(-3, 4)

        CS = ax.contourf(t, per, power, levels=levels, cmap = self.attrs['sh_pal'], extend = 'min')

        plt.colorbar(CS, label='Power')

        if figsize == (0,0):
            figsize = ( 15 + (1/24 * len(t)), 
                        12 
                        )
            fig.set_size_inches(figsize)

        # set y ticks in log 2
        y_ticks = [1,2,4,6,12,24,36]
        y_ticks_log = np.log2(y_ticks)
        plt.yticks(ticks=y_ticks_log, labels=y_ticks, fontsize=20, rotation=0)
        plt.ylim(np.log2([2,38]))

        # set x ticks to every 12
        x_ticks = np.arange(0, 24*200, 12)
        plt.xticks(ticks=x_ticks, labels=x_ticks, fontsize=20, rotation=0)
        plt.xlim(np.min(t), np.max(t))

        plt.title(title)
        plt.ylabel("Period Frequency (Hours)", fontsize=20)
        plt.xlabel("ZT (Hours)", fontsize=20)

        if grids:
            plt.grid(axis='y')

        return fig

    # HMM plots

    def plot_hmm_overtime(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, wrapped:bool = False, t_bin:int = 60, 
        func:str = 'max', avg_window:int = 30, day_length:int = 24, lights_off:int = 12, title:str = '', t_column:str = 't', grids:bool = False, 
        figsize:tuple=(0,0)):
        """
        Generates a plot of the occurance of each state as a percentage at each time point for the whole dataset. The is the go to plot for understanding 
        how states change over the day.

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of you wish to decode. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. 
                    If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. Default is None.
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
            fig (matplotlib.figure.Figure): Figure object of the plot.
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
            tdf = hmm_pct_state(l, t, list(range(len(labels))), avg_window = 5)
            df = pd.concat([df, tdf], ignore_index = True)

        df.rename(columns = dict(zip([f'state_{c}' for c in range(0,len(labels))], labels)), inplace = True)
        melt_df = df.melt('t')
        m = melt_df[['variable']]
        melt_df = melt_df.rename(columns = {'variable' : 'id'}).set_index('id')
        melt_df = melt_df.rename(columns = {'value' : 'Probability of state'})
        m['id'] = m[['variable']]
        m = m.set_index('id')

        df = self.__class__(melt_df, m)

        return df.plot_overtime(variable='Probability of state', wrapped=wrapped, facet_col='variable', facet_arg=labels, avg_window=avg_window, day_length=day_length, 
                                    lights_off=lights_off, title=title, grids=grids, t_column='t', col_list = colours, figsize=figsize)

    def plot_hmm_split(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|str = None, facet_arg:None|list = None, 
        facet_labels:None|list = None, wrapped:bool = False, t_bin:int|list = 60, func:str = 'max', avg_window:int = 30, day_length:int = 24, lights_off:int = 12,
        title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple=(0,0)):
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
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Static image of the plot.
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        Note:
            If providing multiple HMMs with different number of states, then leave labels as None. Generic labels will be generated
            for the largerst state model.
            This plotting method workds much better in the Plotly version due to having to save each subplot as an image.
        """

        assert isinstance(wrapped, bool)

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            df = self.xmv(facet_col, facet_arg)
        else:
            df = self.copy(deep=True)

        if facet_col is None and isinstance(hmm, list): facet_col = True # for when testing different HMMs on the same dataset

        if facet_col is None:  # decode the whole dataset
            df = self.__class__(self._hmm_decode(df, hmm, t_bin, variable, func, t_column, return_type='table'), df.meta, check=True)
        else:
            if isinstance(hmm, list) is False: # if only 1 hmm but is faceted, decode as whole for efficiency
                df = self.__class__(self._hmm_decode(df, hmm, t_bin, variable, func, t_column, return_type='table'), df.meta, check=True)

        states_dict = {k : [] for k in labels}

        # iterate over the faceted column. Decode and augment to be ready to plot
        for c, arg in enumerate(facet_arg):
            
            if arg != None:
                sub_df = df.xmv(facet_col, arg)
            else:
                sub_df = df
                if not isinstance(hmm, list): arg = ' '
                else: 
                    arg = facet_labels[c]
                    facet_arg[c] = arg

            if isinstance(hmm, list) is True: # call the decode here if multiple HMMs
                sub_df = self._hmm_decode(sub_df, h_list[c], b_list[c], variable, func, t_column, return_type='table').set_index('id')

            states_list = sub_df.groupby(sub_df.index)['state'].apply(np.array)
            time_list = sub_df.groupby(sub_df.index)['bin'].apply(list)

            # calculate the % per state
            states_df = pd.DataFrame()
            for l, t in zip(states_list, time_list):
                tdf = hmm_pct_state(l, t, list(range(len(labels))), avg_window = 5)
                states_df = pd.concat([states_df, tdf], ignore_index = True)

            # melt to make ready for plot
            states_df.rename(columns = dict(zip([f'state_{c}' for c in range(0,len(labels))], labels)), inplace = True)
            melt_df = states_df.melt('t')
            melt_df['facet_col'] = len(melt_df) * [arg]
            melt_df = melt_df.rename(columns = {'value' : 'Probability of state'})

            # filter by each state and append to a list in a dict
            for state in labels:
                sub_states = melt_df[melt_df['variable'] == state]
                states_dict[state].append(sub_states)

        # calculate rows and columns, maximum 2 rows and then more columns as more states
        if len(labels) <= 2:
            nrows = 1
            ncols = 2
        else:
            nrows =  2
            ncols = round(len(labels) / 2)

        figs = []

        for c, state in enumerate(labels):

            plot_df = pd.concat(states_dict[state])
            plot_m = pd.DataFrame(data = {'id' : list(set(plot_df['facet_col'])), 'facet_col' : list(set(plot_df['facet_col']))})
            plot_df.rename(columns = {'facet_col' : 'id'}, inplace = True)
            plot_bh = self.__class__(plot_df, plot_m, check=True)

            if facet_col is None: # add colours to each state if no facet
                fig = plot_bh.plot_overtime(variable='Probability of state', wrapped=wrapped, avg_window=avg_window, day_length=day_length, 
                                            lights_off=lights_off, title=state, grids=grids, t_column='t', col_list = [colours[c]])      
            else:
                fig = plot_bh.plot_overtime(variable='Probability of state', wrapped=wrapped, facet_col='facet_col', facet_arg=facet_arg, facet_labels=facet_labels,
                                        avg_window=avg_window, day_length=day_length, lights_off=lights_off, title=state, grids=grids, t_column='t') #, col_list = [colours[c]])             
            ax = plt.gca()
            ax.set_ylim([-0.02525, 1])            
            plt.close()
            figs.append(fig)

        c = []

        if figsize == (0,0):
            figsize = (8*nrows, 10*ncols)

        combined_fig = plt.figure(figsize = figsize)

        for pos, f in enumerate(figs):
            c.append( combined_fig.add_subplot(nrows, ncols, pos+1))
            c[-1].axis('off')  # Turn off axis
            c[-1].imshow( self._fig2img (f) )

        # Adjust the layout of the subplots in the combined figure
        combined_fig.subplots_adjust(wspace=0.05, hspace=-0.55)
        
        return combined_fig

    def plot_hmm_quantify(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|list = None, 
        facet_arg:None|list = None, facet_labels:None|list = None, t_bin:int = 60, func:str = 'max', 
        title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple=(0,0)):        
        """
        Creates a quantification plot of how much a predicted state appears per individual. 

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
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.   
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        """

        grouped_data, labels, _, facet_col, facet_labels, palette_dict = self._internal_plot_hmm_quantify(hmm, variable, labels, colours, facet_col, 
                                                                                    facet_arg, facet_labels, t_bin, func, t_column)
        plot_column = 'Fraction of time in each State'

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_labels)+2, 4+2)
        
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim([0, 1.01])

        if facet_col:
            # merge the facet_col column and replace with the labelsBinned
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge =  0.8 - 0.8 / len(facet_labels))
            sns.pointplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge =  0.8 - 0.8 / len(facet_labels))

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue ='state', palette=palette_dict, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue ='state', palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        grouped_data.drop(columns=['bin', 'previous_state'], inplace=True)
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)
            
        return fig, grouped_data

    def plot_hmm_quantify_length(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|list = None, 
        facet_arg:None|list = None, facet_labels:None|list = None, scale:str = 'log', t_bin:int = 60, func:str = 'max', 
        title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple=(0,0)):
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
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.   
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        """

        grouped_data, labels, _, facet_col, facet_labels, palette_dict = self._internal_plot_hmm_quantify_length(hmm, variable, labels, colours, facet_col, 
                                                                                    facet_arg, facet_labels, t_bin, func, t_column)
        plot_column = 'Length of state bout (mins)'

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_labels)+2, 4+2)
        
        fig, ax = plt.subplots(figsize=figsize)
        if scale is not None:
            plt.yscale(scale)

        if facet_col:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge =  0.8 - 0.8 / len(facet_labels))
            sns.pointplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge =  0.8 - 0.8 / len(facet_labels))

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue ='state', palette=palette_dict, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue ='state', palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_hmm_quantify_length_min_max(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|list = None, 
        facet_arg:None|list = None, facet_labels:None|list = None, scale:str = 'log', t_bin:int = 60, func:str = 'max', 
        title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple=(0,0)):
        """
        Generates a plot of every run of each state. Use when you'd like to know at what point in time one state becomes another.
        The Seaborn version plots every point as a dot plus a box plot of the interquartile range.

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
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.   
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        Notes:
            In processing the first and last bouts of the variable fed into the HMM are trimmed to prevent them affecting the result. 
            Any missing data points will also affect the end quantification.
        """

        grouped_data, labels, _, facet_col, facet_labels, palette_dict = self._internal_plot_hmm_quantify_length_min_max(hmm, variable, labels, colours, facet_col, 
                                                                                    facet_arg, facet_labels, t_bin, func, t_column)
        plot_column = 'Length of state bout (mins)'

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_labels)+2, 4+2)
        
        fig, ax = plt.subplots(figsize=figsize)
        if scale is not None:
            plt.yscale(scale)

        if facet_col:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge =  0.8 - 0.8 / len(facet_labels))
            sns.boxplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict,
                            showcaps=False, showfliers=False, whiskerprops={'linewidth':0}, dodge =  0.8 - 0.8 / len(facet_labels))

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue ='state', palette=palette_dict, alpha=0.5, legend=False)
            sns.boxplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue ='state', palette=palette_dict,
                            showcaps=False, showfliers=False, whiskerprops={'linewidth':0})

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_hmm_quantify_transition(self, hmm, variable:str = 'moving', labels:list = None, colours:list = None, facet_col:None|list = None, 
        facet_arg:None|list = None, facet_labels:None|list = None, t_bin:int = 60, func:str = 'max', 
        title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple=(0,0)):
        """
        Generates a plot of every run of each state. Use when you'd like to know at what point in time one state becomes another.
        The Seaborn version plots every point as a dot plus a box plot of the interquartile range.

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
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.   
        Raises:
            Multiple assertion of ValueErrors in regards to faceting and HMM lists
        Notes:
            In processing the first and last bouts of the variable fed into the HMM are trimmed to prevent them affecting the result. 
            Any missing data points will also affect the end quantification.
        """

        grouped_data, labels, _, facet_col, facet_labels, palette_dict = self._internal_plot_hmm_quantify_transition(hmm, variable, labels, colours, facet_col, 
                                                                                    facet_arg, facet_labels, t_bin, func, t_column)
        plot_column = 'Fraction of transitions into each state'

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_labels)+2, 4+2)
        
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim([0, 1.01])

        if facet_col:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge =  0.8 - 0.8 / len(facet_labels))
            sns.pointplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.8 - 0.8 / len(facet_labels))

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue ='state', palette=palette_dict, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue ='state', palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_hmm_raw(self, hmm, variable:str = 'moving', colours:list = None, num_plots:int = 5, t_bin:int = 60, func:str = 'max', day_length:int = 24, 
        lights_off:int = 12, t_column:str = 't', title:str = '', grids:bool = False, figsize:tuple=(0,0)):
        """
        Generates a plot that represents each predicted state over the experiment. This plot is great for getting a quick feel to your HMM and useful
        for comparing multiple HMMs trained with different architecture on the same dataset. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM | list([hmmlearn.hmm.CategoricalHMM ])): This should be a trained HMM Learn object with the 
                    correct hidden states and emission states for your dataset. If as a list, the number of plots will be the same as 
                    its length, with each plot being the same specimens data decoded with each HMM.
                variable (str, optional): The column heading of the variable of interest for decoding. Default is "moving"
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states.
                    If None the colours are a default for 4 states (blue and red) or the first X number of colours in the given palette. Default is None.
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn.
                    If given a list of HMMs, the largest n state model is found and colours checked against.
                numm_plots (int, optional): The number of plots as rows in a subplot. If a list of HMMs is given num_plots will be overridden
                    by the length of that list. Default is 5.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute.
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable
                day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
                lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. 
                    Default is 12.    
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                title (str, optional): The title of the plot. Default is an empty string.  
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            A matplotlib figure that is a combination of scatter and a line plot.
        Raises:
            TypeError:
                Can be thrown sometimes when the filtered dataframe contains no data. If this occurs run the method again to randomnly select a 
                    different specimen for plotting.
        Note:
            Plotting with the results of a stimulus experiment is only avaible in the plotly version of plot_hmm_raw.
            Plotting the movement variable is only available in the plotly version due to how cluttered it makes the plot.
        """
        d_copy = self.copy(deep=True)
        labels, colours = self._check_hmm_shape(hm = hmm, lab = None, col = colours)
        y_ticks = list(range(len(labels)))

        colours_index = {c : col for c, col in enumerate(colours)}

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

        # BOXPLOT
        fig_rows = num_plots
        fig_cols = 1

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (20, 4*(num_plots)+2)
        fig, axes = plt.subplots(fig_rows, fig_cols, sharex=True, figsize=figsize)

        if fig_rows == 1:
            axes = [axes]

        df_list = [d_copy.xmv('id', id) for id in rand_flies]
        decoded = [self._hmm_decode(d, h, b, variable, func, t_column, return_type='table').dropna().set_index('id') for d, h, b in zip(df_list, h_list, b_list)]

        min_t = []
        max_t = []

        for ax, data in enumerate(decoded):
            
            print(f'Plotting: {data.index[0]}')

            data['bin'] = data['bin'] / (60*60)
            st = data['state'].to_numpy()
            time = data['bin'].to_numpy()

            for c, loop_col in enumerate(colours):
                if c == 0:
                    col = np.where(st == c, loop_col, '')
                else:
                    col = np.where(st == c, loop_col, col)

            axes[ax].scatter(time, st, s=25, marker="o", c=col)
            axes[ax].plot(
                time,
                st,
                marker="o",
                markersize=0,
                mfc="white",
                mec="white",
                c="black",
                lw=0.25,
                ls="-",
            )
            min_t.append(int((day_length/lights_off) * floor(data['bin'].min() / (day_length/lights_off))))
            max_t.append(int((day_length/lights_off) * floor(data['bin'].max() / (day_length/lights_off))))
            axes[ax].set_yticks(y_ticks) 

        if np.max(max_t) - np.min(min_t) >= 24:
            x_ticks = np.arange(np.min(min_t), np.max(max_t) + lights_off, lights_off, dtype = int)
            axes[0].set_xticks(x_ticks) 
            axes[0].set_xticklabels(x_ticks, fontsize=12)  

        if title: fig.suptitle(title, fontsize=20)
        fig.supxlabel("ZT Hours")
        fig.supylabel("Predicted State")
        return fig

    def plot_hmm_response(self, mov_df, hmm, variable:str = 'moving', response_col:str = 'has_responded', labels:list = None, colours:list = None, 
        facet_col:None|str = None, facet_arg:None|list = None, facet_labels:None|list = None, t_bin:int = 60, func:str = 'max', 
        title:str = '', t_column:str = 't', grids:bool = False, figsize:tuple=(0,0)):
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
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn. Default is None.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. 
                    Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. 
                    If None the labels will be those from the metadata. Default is None.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                func (str, optional): When binning to the above what function should be applied to the grouped data. 
                    Default is "max" as is necessary for the "moving" variable.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.    
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

        labels, _ = self._check_hmm_shape(hm = hmm, lab = labels, col = None)
        facet_arg, facet_labels, hmm, t_bin = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        if isinstance(hmm, list) and facet_col is None: # End method is trying to have mutliple HMMs with no facet
            raise RuntimeError('This method does not support multiple HMMs and no facet_col')

        plot_column = f'{response_col}_mean'

        grouped_data, palette_dict, h_order = self._hmm_response(mov_df, hmm, variable, response_col, labels,
                                                                    colours, facet_col, facet_arg, facet_labels, 
                                                                    t_bin, func, t_column)

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_arg)+2, 8)

        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim([0, 1.01])

        if facet_col:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=h_order, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge = 0.8 - 0.8 / len(h_order))
            sns.pointplot(data=grouped_data, x='state', y=plot_column, order=labels, hue=facet_col, hue_order=h_order, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.8 - 0.8 / len(h_order))

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=h_order)

        else:
            sns.stripplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue = '', hue_order = h_order, palette=palette_dict, alpha=0.5, legend=False, dodge = 0.8 - 0.8 / len(h_order))
            sns.pointplot(data=grouped_data, x='state', y=plot_column, order=labels, ax=ax, hue = '', hue_order = h_order, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.8 - 0.8 / len(h_order))

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)
        grouped_data.drop(columns=['previous_state'], inplace=True)

        return fig, grouped_data

    def plot_response_over_hmm_bouts(self, mov_df, hmm, variable:str = 'moving', response_col:str = 'has_responded', labels:list = None, colours:list = None, 
        x_limit:int = 30, t_bin:int = 60, func:str = 'max', title:str = '', grids:bool = False, t_column:str = 't', figsize:tuple = (0,0)):
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

        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim([0, 1.01])
        plt.xlim([1, x_limit])
        x_ticks = np.arange(1, x_limit+1, 1, dtype = int)

        if figsize == (0,0):
            figsize = ( 6 + 1/2 * x_limit, 
                        8
                        )
            fig.set_size_inches(figsize)

        for hue in h_order:

            sub_df = grouped_data[grouped_data['label_col'] == hue]

            plt.plot(sub_df["previous_activity_count"], sub_df["mean"], label = hue, color = palette_dict[hue])
            plt.fill_between(
            sub_df["previous_activity_count"], sub_df["y_min"], sub_df["y_max"], alpha = 0.25, color = palette_dict[hue]
            )

        # Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=h_order)

        plt.xlabel("Consecutive minutes in state")
        plt.ylabel("Response rate")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) # legend outside of area if faceted 

        plt.title(title)

        if grids:
            plt.grid(axis='y')

        return fig
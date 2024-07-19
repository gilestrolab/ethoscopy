import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from math import floor, ceil, sqrt
from scipy.stats import zscore
from functools import partial, update_wrapper
from colour import Color

from ethoscopy.behavpy_draw import behavpy_draw

from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.analyse import max_velocity_detector
from ethoscopy.misc.rle import rle
from ethoscopy.misc.bootstrap_CI import bootstrap
from ethoscopy.misc.hmm_functions import hmm_pct_transition, hmm_mean_length, hmm_pct_state

class behavpy_seaborn(behavpy_draw):
    """
    seaborn wrapper around behavpy_core
    """

    canvas = 'seaborn'
    error = 'se'

    def heatmap(self, variable = 'moving', t_column = 't', title = '', figsize = (0,0)):
        """
        Plots a heatmap of a chosen variable.

        Args:
            variable (str, optional): The variable from the DataFrame to plot. Default is 'moving'.
            t_column (str, optional): The name of the time column in the DataFrame. Default is 't'.
            title (str, optional): The title of the plot. Default is an empty string.
            figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.
        Returns:
            matplotlib.figure.Figure: The created figure.

        Note:
            For accurate results, the data should be appropriately preprocessed to ensure that 't' values are
            in the correct format (seconds from time 0) and that 'variable' exists in the DataFrame.
        """
        
        data, time_list, id = self.heatmap_dataset(variable, t_column)

        data = pd.DataFrame(data.tolist())

        n = 12
        t_min = int(n * floor(time_list.min() / n))
        t_max = int(n * ceil(time_list.max() / n)) 

        # Set every nth x-tick label, in this example every 12th label
        n = 12
        x_labels = time_list[::n].astype(int)  # Get every nth label
        x_ticks = np.arange(0, len(time_list), n)  # Get every nth location

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (0.5*len(x_ticks), 0.1*len(id))

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, cmap="viridis", ax=ax)


        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)
        plt.xlabel("ZT Time (Hours)")

        plt.yticks(ticks=np.arange(0, len(id), 2), labels=id[::-2], rotation=0)

        if title: fig.suptitle(title, fontsize=16)

        return fig

    def plot_overtime(self, variable:str, wrapped:bool = False, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, facet_tile:None|str = None, avg_window:int = 30, day_length:int = 24, lights_off:int = 12, title:str = '', grids:bool = False, t_column:str = 't', col_list:list|None = None, figsize:tuple = (0,0)):
        """
        Plots a line hypnogram using seaborn, displaying rolling averages of a chosen variable over time.
        Optionally, the plot can be wrapped and faceted.

        Args:
            variable (str): The name of the variable in the DataFrame to plot. eg: asleep, moving
            wrapped (bool, optional): If True, time data will be wrapped to daylength. Default is False.
            facet_col (str, optional): The name of the column to use for faceting. Can be main column or from metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. Default is None.
            facet_labels (list, optional): The labels to use for faceting. Default is None.
            avg_window (int, optional): The number, in minutes, that is applied to the rolling smoothing function. The default is 30 minutes, which for a t_diff of 10 would be a window of 180.
            day_length (int, optional): The length of the day in hours for wrapping. Default is 24.
            lights_off (int, optional): The time of "lights off" in hours. Default is 12.
            title (str, optional): The title of the plot. Default is an empty string.
            grids (bool, optional): If True, horizontal grid lines will be displayed on the plot. Default is False.
            t_column (str, optional): The name of the time column in the DataFrame. Default is 't'.
            figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            matplotlib.figure.Figure: The created figure.

        Note:
            For accurate results, the data should be appropriately preprocessed to ensure that 't' values are
            in the correct format (seconds from time 0) and that 'variable' exists in the DataFrame.
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

            gb_df, t_min, t_max, col, _ = self._generate_overtime_plot(data = data, name = name, col = col, var = variable, 
                                                                                    avg_win = int((avg_window * 60)/self[t_column].diff().median()), wrap = wrapped, day_len = day_length, 
                                                                                    light_off= lights_off, t_col = t_column, canvas = 'seaborn')
            if gb_df is None:
                continue

            plt.plot(gb_df["t"], gb_df["mean"], label=name, color=col)
            plt.fill_between(
            gb_df["t"], gb_df["y_min"], gb_df["y_max"], alpha = 0.25, color=col
            )

            min_t.append(t_min)
            max_t.append(t_max)

        if isinstance(lights_off, float):
            x_ticks = np.arange(np.min(min_t), np.max(max_t), lights_off/2)
        else:
            x_ticks = np.arange(np.min(min_t), np.max(max_t), lights_off/2, dtype = int)

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

        plt.title(title)

        if grids:
            plt.grid(axis='y')

        # For every 24 hours, draw a rectangle from 0-12 (daytime) and another from 12-24 (nighttime)
        bar_range, thickness = circadian_bars(t_min, t_max, min_y = ymin, max_y = ymax, day_length = day_length, lights_off = lights_off, canvas = 'seaborn')
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

    def plot_quantify(self, variable:str, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, fun:str = 'mean', title:str = '', grids:bool = False, figsize = (0,0)):
        """
        Creates a boxplot and swarmplot for the given data with the option to facet by a specific column.

        Args:
            variable (str): The variable column name to be plotted. e.g. asleep
            facet_col (str, optional): The column name used for faceting. Defaults to None.
            facet_arg (list, optional): List of values used for faceting. Defaults to None.
            facet_labels (list, optional): List of labels used for faceting. Defaults to None.
            fun (str, optional): Function to apply on the data, e.g., 'mean'. Defaults to 'mean'.
            title (str, optional): Title of the plot. Defaults to ''.
            grids (bool, optional): If True, add a grid to the plot. Defaults to False.
            figsize (tuple, optional): Tuple specifying the figure size. Default is (0, 0) which auto-adjusts the size.
        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
            data (pandas.DataFrame): DataFrame with grouped data based on the input parameters.

        Raises:
            ValueError: If `fun` is not a valid function.

        Note:
            This function uses seaborn to create a stripplot and pointplot. It uses boostrapping with n 1000 to calculate 95%
            confidence intervals. It allows to facet the data by a specific column.
            The function to be applied on the data is specified by the `fun` parameter.
        """

        # If facet_col is provided but facet arg is not, will automatically fill facet_arg and facet_labels with all the possible values
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if not isinstance(variable, list):
            variable = [variable]

        data = self.copy(deep=True)

        data_summary = {}
        for var in variable:
            data_summary.update( {
                f"{var}_{fun}" : (var, fun),
                f"{var}_std" : (var, 'std')
                } )

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)
            
        # applt the averaging function by index per variable
        grouped_data = data.groupby(data.index).agg(**data_summary)

        # BOXPLOT
        fig_rows = len(variable)
        fig_cols = 1

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (2*len(facet_arg), 4*fig_rows)

        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize)

        # axes is
        #  matplotlib.axes._axes.Axes if only one subplot
        #  numpy.ndarray if multiple subplots
        # Flatten the axes list, in case we have more than one row
        if fig_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        grouped_data = self.facet_merge(grouped_data, facet_col, facet_arg, facet_labels)

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
                sns.stripplot(data=grouped_data, y=plot_column, color = palette[0], ax=ax, alpha=0.5, legend=False,)
                sns.pointplot(data=grouped_data, y=plot_column, color = palette[0], ax=ax, estimator = 'mean',
                                linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

            ax.set_ylabel(var)

        if grids: plt.grid(axis='y')
        plt.title(title)
        plt.tight_layout()

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_compare_variables(self, variables, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False):

        return self.plot_quantify(variable = variables, facet_col = facet_col, facet_arg = facet_arg, facet_labels = facet_labels, fun = fun, title = title, grids = grids)

    def plot_day_night(self, variable, facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, lights_off = 12, title = '', t_column = 't', grids = False, figsize=(0,0)):
        """
        Plot day and night data.

        Parameters:
            variable (str): The variable to plot.
            facet_col (str, optional): The column to facet the data by. Default is None.
            facet_arg (list, optional): The list of values to use for faceting. Default is None.
            facet_labels (list, optional): The list of labels for the facet_arg values. Default is None.
            day_length (int, optional): The length of the day in hours. Default is 24.
            lights_off (int, optional): The hour when lights are turned off. Default is 12.
            title (str, optional): The title of the plot. Default is an empty string.
            grids (bool, optional): Whether to display grid lines. Default is False.
            figsize (tuple, optional): The size of the figure. Default is (0, 0) which auto-adjusts the size.

        Returns:
            tuple: A tuple containing the figure object and the processed data.

        Raises:
            None

        """
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)
        plot_column = f'{variable}_mean'

        data_summary = {
            "%s_mean" % variable : (variable, 'mean'),
            "%s_std" % variable : (variable, 'std'),
            }

        data = self.copy(deep=True)

        #Add phase information to the data
        data.add_day_phase(day_length = day_length, lights_off = lights_off, t_column = t_column)

        # takes subset of data if requested
        if facet_col and facet_arg:
            # Add the specified columns from metadata
            data = data.xmv(facet_col, facet_arg)

        grouped_data = data.groupby([data.index, 'phase'], observed = True).agg(**data_summary).reset_index(1)

        # BOXPLOT
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_arg)+2, 4+2)

        fig, ax = plt.subplots(figsize=figsize)

        y_range, dtick = self._check_boolean(list(self[variable]))
        if y_range:
            plt.ylim(y_range)

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        if facet_col:
            # merge the facet_col column and replace with the labelsBinned
            grouped_data = self.facet_merge(grouped_data, facet_col, facet_arg, facet_labels)

            sns.stripplot(data=grouped_data, x='phase', y=plot_column, order=['light', 'dark'], hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge = True)
            sns.pointplot(data=grouped_data, x='phase', y=plot_column, order=['light', 'dark'], hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.4)

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:

            sns.stripplot(data=grouped_data, x='phase', y=plot_column, order=['light', 'dark'], ax=ax, hue ='phase', palette={"light" : "gold", "dark" : "darkgrey"}, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='phase', y=plot_column, order=['light', 'dark'], ax=ax, hue ='phase', palette={"light" : "gold", "dark" : "darkgrey"}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_anticipation_score(self, mov_variable = 'moving', facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, lights_off = 12, title = '', grids = False, figsize=(0,0)):
        """
        Plots the anticipation scores for lights on and off periods, separately for each category defined by facet_col.
        
        This function calculates the anticipation scores for lights off and on conditions and then plots a seaborn boxplot.

        Parameters
        ----------
        mov_variable : str, optional
            The movement variable to consider for calculating the anticipation score, by default 'moving'.
        facet_col : str, optional
            The column name to be used for faceting, by default None.
        facet_arg : list, optional
            List of arguments to consider for faceting, by default None.
        facet_labels : list, optional
            Labels for the facet arguments, by default None.
        day_length : int, optional
            The length of the day in hours, by default 24.
        lights_off : int, optional
            The time in hours when the lights are turned off, by default 12.
        title : str, optional
            The title of the plot, by default ''.
        grids : bool, optional
            If True, the grid is displayed on the plot, by default False.
        figsize : tuple, optional
            Tuple indicating the size of the figure (width, height), by default (0,0) which indicates automatic size.

        Returns
        -------
        tuple
            A tuple containing the figure and the dataset used for plotting.
        """


        # If facet_col is provided but facet arg is not, will automatically fill facet_arg and facet_labels with all the possible values
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        data = data.dropna(subset=[mov_variable])
        data = data.wrap_time()

        dataset = self.anticipation_score(data, mov_variable, day_length, lights_off).set_index('id')

        # BOXPLOT
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_arg)+2, 6+2)

        fig, ax = plt.subplots(figsize=figsize)

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)}

        if facet_col:
            # merge the facet_col column and replace with the labels
            dataset = self.facet_merge(dataset, facet_col, facet_arg, facet_labels)

            sns.stripplot(data=dataset, x='phase', y="anticipation_score", order=['Lights On', 'Lights Off'], ax=ax, hue =facet_col, hue_order=facet_labels, palette=palette_dict, alpha=0.5, legend=False, dodge = True)
            sns.pointplot(data=dataset, x='phase', y="anticipation_score", order=['Lights On', 'Lights Off'], ax=ax, hue =facet_col, hue_order=facet_labels, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.4)

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)

        else:
            sns.stripplot(data=dataset, x='phase', y="anticipation_score", order=['Lights On', 'Lights Off'], ax=ax, hue ='phase', palette={"Lights On" : "gold", "Lights Off" : "darkgrey"}, alpha=0.5, legend=False)
            sns.pointplot(data=dataset, x='phase', y="anticipation_score", order=['Lights On', 'Lights Off'], ax=ax, hue ='phase', palette={"Lights On" : "gold", "Lights Off" : "darkgrey"}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        if grids: plt.grid(axis='y')
        plt.title(title)
        plt.tight_layout()

        # The score is in %
        plt.ylim(0,100)

        # reorder dataframe for stats output
        if facet_col: dataset = dataset.sort_values(facet_col)

        return fig, dataset
    
    @staticmethod
    def _plot_single_actogram(dt, figsize, days, title, day_length):

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (10, len(days)/2)

        fig, axes = plt.subplots(len(days)-1, 1, figsize=figsize, sharex=True)
        axes[0].set_title(title)

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

    def plot_actogram(self, mov_variable = 'moving', bin_window = 5, t_column = 't', facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, title = '', figsize=(0,0)):
        """
        This function creates actogram plots from the provided data. Actograms are useful for visualizing 
        patterns in activity data (like movement or behavior) over time, often with an emphasis on daily 
        (24-hour) rhythms. 

        Args:
            mov_variable (str, optional): The name of the column in the dataframe representing movement 
                data. Default is 'moving'.
            bin_window (int, optional): The bin size for data aggregation in minutes. Default is 5.
            t_column (str, optional): The name of the column in the dataframe representing time data.
                Default is 't'.
            facet_col (str, optional): The name of the column to be used for faceting. If None, no faceting 
                is applied. Default is None.
            facet_arg (list, optional): List of arguments to be used for faceting. If None and if 
                facet_col is not None, all unique values in the facet_col are used. Default is None.
            facet_labels (list, optional): List of labels to be used for the facets. If None and if 
                facet_col is not None, all unique values in the facet_col are used as labels. Default is None.
            day_length (int, optional): The length of the day in hours. Default is 24.
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
            figs = []
            for subplot in facet_arg:

                dt = data.loc [data[facet_col] == subplot]
                title = "%s - %s" % (title, subplot)
                fig = self._plot_single_actogram(dt, figsize, days, title, day_length)
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
            return self._plot_single_actogram(data, figsize, days, title, day_length)

    def plot_actogram_tile(self, mov_variable = 'moving', bin_window = 15, t_column = 't', labels = None, day_length = 24, title = '', figsize=(0,0)):
        """
        This function creates a grid or tile actogram plot of all specimens in the provided data. Actograms are useful for visualizing 
        patterns in activity data (like movement or behavior) over time, often with an emphasis on daily 
        (24-hour) rhythms. 

        Args:
            mov_variable (str, optional): The name of the column in the dataframe representing movement 
                data. Default is 'moving'.
            bin_window (int, optional): The bin size for data aggregation in minutes. Default is 5.
            t_column (str, optional): The name of the column in the dataframe representing time data.
                Default is 't'.
            facet_col (str, optional): The name of the column to be used for faceting. If None, no faceting 
                is applied. Default is None.
            facet_arg (list, optional): List of arguments to be used for faceting. If None and if 
                facet_col is not None, all unique values in the facet_col are used. Default is None.
            facet_labels (list, optional): List of labels to be used for the facets. If None and if 
                facet_col is not None, all unique values in the facet_col are used as labels. Default is None.
            day_length (int, optional): The length of the day in hours. Default is 24.
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
                raise AttributeError(f'{labels} is not a column in the metadata')
            title_list = self.meta[labels].tolist() 
        else:
            title_list = self.meta.index.tolist()

        facet_arg = self.meta.index.tolist()

        # call the internal actogram augmentor
        data, days = self._internal_actogram(self, mov_variable, bin_window, t_column, facet_col='id')

        # get the nearest square number to make a grid plot
        root =  self._get_subplots(len(title_list))

        figs = []
        for subplot, label in zip(facet_arg, title_list):

            dt = data.loc[data['id'] == subplot]
            subtitle = "%s" % (label)
            fig = self._plot_single_actogram(dt, figsize, days, subtitle, day_length)
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

        combined_fig.suptitle(title, size = 7*root)

        # Adjust the layout of the subplots in the combined figure
        #combined_fig.tight_layout()

        return combined_fig

    def survival_plot(self, facet_col = None, facet_arg = None, facet_labels = None, repeat = False, day_length = 24, lights_off = 12, title = '', grids = False, t_column = 't', figsize=(0,0)):
        """
        Currently only returns a data frame that can be used with seaborn to plot whilst we go through the changes
        
            Args:
                facet_col (str, optional): The name of the column to use for faceting. Can be main column or from metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. Default is None.
                facet_labels (list, optional): The labels to use for faceting. Default is None.
                repeat (bool/str, optional): If False the function won't look for a repeat column. If wanted the user should change the argument to the column in the metadata that contains repeat information. Default is False
                day_length (int, optional): The length of the day in hours for wrapping. Default is 24.
                lights_off (int, optional): The time of "lights off" in hours. Default is 12.
                title (str, optional): The title of the plot. Default is an empty string.
                grids (bool, optional): If True, horizontal grid lines will be displayed on the plot. Default is False.
                t_column (str, optional): The name of the time column in the DataFrame. Default is 't'.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.
        
        returns:
            A Pandas DataFrame with columns hour, survived, and label. It is formatted to fit a Seaborn plot

        """

        if repeat is True:
            if repeat not in self.meta.columns:
                raise KeyError(f'Column "{repeat}" is not a metadata column, please check and add if you want repeat data')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg and repeat:
            data = self.xmv(facet_col, facet_arg).merge(self.meta[[facet_col, repeat]], left_index=True, right_index=True)
        elif facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg).merge(self.meta[[facet_col]], left_index=True, right_index=True)
        else:
            data = self.copy(deep=True)

        sur_df = data.groupby(facet_col, group_keys = False).apply(partial(self._time_alive, facet_col = facet_col, repeat = repeat, t_column = t_column))

        x_ticks = np.arange(0, sur_df['hour'].max() + day_length, day_length)

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = ( 6 + 1/4 * len(x_ticks), 
                        4 + 1/32 * len(x_ticks) 
                        )

        fig, ax = plt.subplots(figsize=figsize)

        # sns.set_style("ticks")
        sns.lineplot(data = sur_df, x = "hour", y = "survived", errorbar = self.error, hue = 'label', hue_order = facet_arg, ax=ax, palette = self._palette) # add a style option too to differentiate multiple controls from exp

        # x_major_locator=MultipleLocator(day_length) 
        # ax=plt.gca() #ax is an instance of two coordinate axes
        # ax.xaxis.set_major_locator(x_major_locator) #Set the main scale of the x-axis to a multiple of 1
        plt.xlim(np.min(x_ticks), np.max(x_ticks))
        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)

        plt.ylabel("Survival (%)")
        plt.xlabel("ZT (Hours)")

        plt.title(title)

        if grids:
            plt.grid(axis='y')
        ymin, ymax = 0, 105
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

    def plot_response_quantify(self, response_col = 'has_responded', facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False, figsize = (0,0)):
        """
        """
        if response_col not in self.columns.tolist():
            raise KeyError(f'The column you gave {response_col}, is not in the data. Check you have analyed the dataset with puff_mago')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        plot_column = f'{response_col}_mean'

        data_summary = {
            "%s_mean" % response_col : (response_col, 'mean'),
            "%s_std" % response_col : (response_col, 'std'),
            }

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)
            # apply the specified operation and add the specified columns from metadata
            grouped_data = data.groupby([data.index, 'has_interacted']).agg(**data_summary).reset_index(level = 1).merge(self.meta[[facet_col]], left_index=True, right_index=True)
            grouped_data[facet_col] = grouped_data[facet_col].astype('category')

        # this applies in case we want to apply the specified information to ALL the data
        else:
            grouped_data = self.groupby([self.index, 'has_interacted']).agg(**data_summary).copy(deep=True).reset_index()

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (6*len(facet_arg), 10)

        fig, ax = plt.subplots(figsize=figsize)

        plt.ylim(0, 1.01)
        map_dict = {1 : 'True Stimulus', 2 : 'Spon. Mov'}
        grouped_data['has_interacted'] = grouped_data['has_interacted'].map(map_dict)

        palette = self._get_colours(facet_labels)
        # palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        if facet_col:

            map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
            grouped_data[facet_col] = grouped_data[facet_col].map(map_dict)

            sns.stripplot(data=grouped_data, x=facet_col, y=plot_column, order = facet_labels, hue='has_interacted', hue_order=["Spon. Mov", "True Stimulus"], ax=ax, palette={"Spon. Mov" : "grey", "True Stimulus" : palette[0]}, alpha=0.5, legend=False, dodge = True)
            sns.pointplot(data=grouped_data, x=facet_col, y=plot_column, order = facet_labels, hue='has_interacted', hue_order=["Spon. Mov", "True Stimulus"], ax=ax, palette={"Spon. Mov" : "grey", "True Stimulus" : palette[0]}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.2) 
            
            # ax.set_xticklabels(facet_labels)

            #Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles = handles, labels=["Spon. Mov", "True Stimulus"])

        else:
            sns.stripplot(data=grouped_data, x='has_interacted', y=plot_column, order=["Spon. Mov", "True Stimulus"], ax=ax, hue ='has_interacted', palette={"Spon. Mov" : "grey", "True Stimulus" : palette[0]}, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='has_interacted', y=plot_column, order=["Spon. Mov", "True Stimulus"], ax=ax, hue ='has_interacted', palette={"Spon. Mov" : "grey", "True Stimulus" : palette[0]}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    # Seaborn Periodograms

    def plot_periodogram_tile(self, labels = None, find_peaks = False, title = '', grids = False, figsize=(0,0)):
        """ Create a tile plot of all the periodograms in a periodogram dataframe"""

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
        # print(data)
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
        fig.suptitle(title, size = 7*root)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axes.flat:
            ax.label_outer()

        return fig

    def plot_periodogram(self, facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False, figsize=(0,0)):
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
                figsize (tuple, optional): The size of the figure to be plotted as (width, height). If set to 
                    (0,0), the size is determined automatically. Default is (0,0).
        
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

            gb_df, t_min, t_max, col, _ = self._generate_overtime_plot(data = data, name = name, col = col, var = power_var, 
                                                                                    avg_win = False, wrap = False, day_len = False, 
                                                                                    light_off= False, t_col = period_var, canvas = 'seaborn')
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
    
    def plot_quantify_periodogram(self, facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False, figsize=(0,0)):
        """
        Creates a boxplot and swarmplot of the peaks in circadian rythymn according to a computed periodogram.
        At its core it is just a wrapper of plot_quantify, with some data augmented before being sent to the method.

        Args:
            facet_col (str, optional): The column name used for faceting. Defaults to None.
            facet_arg (list, optional): List of values used for faceting. Defaults to None.
            facet_labels (list, optional): List of labels used for faceting. Defaults to None.
            title (str, optional): Title of the plot. Defaults to ''.
            grids (bool, optional): If True, add a grid to the plot. Defaults to False.
            figsize (tuple, optional): Tuple specifying the figure size. Default is (0, 0) which auto-adjusts the size.
        Returns:
            fig (matplotlib.figure.Figure): Figure object of the plot.
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
        # call the plot quantify method
        return self.plot_quantify(variable = y_label, facet_col=facet_col, facet_arg=facet_arg, facet_labels=facet_labels, 
                                    fun='max', title=title, grids=grids, figsize=figsize)


    def plot_wavelet(self, mov_variable, sampling_rate = 15, scale = 156, wavelet_type = 'morl', t_col = 't', title = '', grids = False, figsize = (0,0)):
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
            A matplotlib,figure
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

    def plot_hmm_overtime(self, hmm, variable = 'moving', labels = None, colours = None, wrapped = False, t_bin = 60, func = 'max', avg_window = 30, day_length = 24, lights_off = 12, title = '', t_column = 't', grids = False, figsize=(0,0)):
        """
        Creates a plot of the occurance of each state as a percentage at each time point. The method will decode and augment the dataset to be fed into a plot_overtime method.

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red). Default is None.
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn.
                wrapped (bool, optional). If True the plot will be limited to a 24 hour day average. Default is False.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                func (str, optional): When binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
                avg_window (int, optioanl): The window in minutes you want the moving average to be applied to. Default is 30 mins
                day_length (int, optional): The lenght in hours the experimental day is. Default is 24.
                lights_off (int, optional): The time point when the lights are turned off in an experimental day, assuming 0 is lights on. Must be number between 0 and day_lenght. Default is 12.
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            returns a Seaborn figure made by the .plot_overtime() method
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
                                    lights_off=lights_off, title=title, grids=grids, t_column=t_column, col_list = colours, figsize=figsize)

    def plot_hmm_split(self, hmm, variable = 'moving', labels = None, colours= None, facet_col = None, facet_arg = None, facet_labels = None, wrapped = False, t_bin = 60, func = 'max', avg_window = 30, day_length = 24, lights_off = 12, title = '', t_column = 't', grids = False, figsize=(0,0)):
        """ works for any number of states """

        assert isinstance(wrapped, bool)

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        # make the colours a range for plotting 
        # start_colours, end_colours = self._adjust_colours(colours)
        # colour_range_dict = {}
        # colours_dict = {'start' : start_colours, 'end' : end_colours}
        # for q in range(0,len(labels)):
        #     start_color = colours_dict.get('start')[q]
        #     end_color = colours_dict.get('end')[q]
        #     N = len(facet_arg)
        #     colour_range_dict[q] = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]


        df = self.copy(deep=True)
        # decode the whole dataset
        df = self.__class__(self._hmm_decode(self, hmm, bin, variable, func, t_column, return_type='table'), self.meta, check=True)

        states_dict = {k : [] for k in labels}

        # iterate over the faceted column. Decode and augment to be ready to plot
        for arg in facet_arg:

            sub_df = df.xmv(facet_col, arg)
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
            ncols =2
        else:
            nrows =  2
            ncols = round(len(labels) / 2)

        figs = []

        for c, state in enumerate(labels):

            plot_df = pd.concat(states_dict[state])
            plot_m = pd.DataFrame(data = {'id' : list(set(plot_df['facet_col'])), 'facet_col' : list(set(plot_df['facet_col']))})
            plot_df.rename(columns = {'facet_col' : 'id'}, inplace = True)
            plot_bh = self.__class__(plot_df, plot_m, check=True)

            fig = plot_bh.plot_overtime(variable='Probability of state', wrapped=wrapped, facet_col='facet_col', facet_arg=facet_arg, facet_labels=facet_labels,
                                        avg_window=avg_window, day_length=day_length, lights_off=lights_off, title=state, grids=grids, t_column=t_column)#, col_list = colour_range_dict[c])
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

    def plot_hmm_quantify(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, facet_labels = None, t_bin = 60, func = 'max', title = '', t_column = 't', grids = False, figsize=(0,0)):
        """
        Creates a quantification plot of how much a predicted state appears per individual. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red). Default is None.
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                func (str, optional): When binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            returns a Seaborn figure and pandas Dataframe with the means per state per indivdual
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        data = self.copy(deep=True)

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)

        # Decode the whole dataset, count each state and the length of each individuals dataset to find the total percentage
        decoded_data = data.get_hmm_raw(hmm, variable=variable, t_bin=t_bin, func=func, t_column=t_column)
        grouped_data = decoded_data.groupby([decoded_data.index, 'state'], sort=False).agg({'bin' : 'count'})
        grouped_data = grouped_data.join(decoded_data.groupby('id', sort=False).agg({'previous_state':'count'}))
        grouped_data['Fraction of time in each State'] = grouped_data['bin'] / grouped_data['previous_state']
        grouped_data.reset_index(level=1, inplace = True)

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_arg)+2, 4+2)
        
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim([0, 1.01])

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        if facet_col:
            # merge the facet_col column and replace with the labelsBinned
            grouped_data = self.facet_merge(grouped_data, facet_col, facet_arg, facet_labels, hmm_labels = labels)
            sns.stripplot(data=grouped_data, x='state', y='Fraction of time in each State', order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge = True)
            sns.pointplot(data=grouped_data, x='state', y='Fraction of time in each State', order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.4)

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)
            sns.stripplot(data=grouped_data, x='state', y='Fraction of time in each State', order=labels, ax=ax, hue ='state', palette={k: v for k,v in zip(labels, colours)}, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='state', y='Fraction of time in each State', order=labels, ax=ax, hue ='state', palette={k: v for k,v in zip(labels, colours)}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)
        grouped_data.drop(columns=['bin', 'previous_state'], inplace=True)
        return fig, grouped_data

    def plot_hmm_quantify_length(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, facet_labels = None, t_bin = 60, func = 'max', title = '', t_column = 't', grids = False, scale = 'log', figsize=(0,0)):
        """
        Creates a quantification plot of the average length of each state per individual. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red). Default is None.
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                func (str, optional): When binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                scale (str, optional): Sets the yaxis scale to log. If you want it in standard increments change to None. It also takes any str compatable with Seaborn Scale function. Default is 'log'.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            returns a Seaborn figure and pandas Dataframe with the mean length of each state per indivdual
        """
        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        data = self.copy(deep=True)

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)

        # Decode the whole dataset, count each state and the length of each individuals dataset to find the total percentage
        decoded_data = data.get_hmm_raw(hmm, variable=variable, t_bin=t_bin, func=func, t_column=t_column)
        states = decoded_data.groupby(decoded_data.index, sort=False)['state'].apply(list)

        df_lengths = []
        for l, id in zip(states, states.index):
            length = hmm_mean_length(l, delta_t = t_bin) 
            length['id'] = [id] * len(length)
            df_lengths.append(length)

        grouped_data = pd.concat(df_lengths)
        grouped_data.rename(columns={'mean_length' : 'Length of state bout (mins)'}, inplace=True)
        grouped_data.set_index('id', inplace=True)

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_arg)+2, 4+2)
        
        fig, ax = plt.subplots(figsize=figsize)
        if scale is not None:
            plt.yscale(scale)

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        if facet_col:
            # merge the facet_col column and replace with the labelsBinned
            grouped_data = self.facet_merge(grouped_data, facet_col, facet_arg, facet_labels, hmm_labels = labels)
            sns.stripplot(data=grouped_data, x='state', y='Length of state bout (mins)', order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge = True)
            sns.pointplot(data=grouped_data, x='state', y='Length of state bout (mins)', order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.4)

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)
            sns.stripplot(data=grouped_data, x='state', y='Length of state bout (mins)', order=labels, ax=ax, hue ='state', palette={k: v for k,v in zip(labels, colours)}, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='state', y='Length of state bout (mins)', order=labels, ax=ax, hue ='state', palette={k: v for k,v in zip(labels, colours)}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_hmm_quantify_length_min_max(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, facet_labels = None, t_bin = 60, func = 'max', title = '', t_column = 't', grids = False, scale = 'log', figsize=(0,0)):
        """
        Creates a quantification plot of the minimum and maximum lengths of each bout. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red). Default is None.
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                func (str, optional): When binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                scale (str, optional): Sets the yaxis scale to log. If you want it in standard increments change to None. It also takes any str compatable with Seaborn Scale function. Default is 'log'.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            returns a Seaborn figure and pandas Dataframe with the mean length of each state per indivdual

        Notes:
            In processing the first and last bouts of the HMM fed variable are trimmed off to prevent them affecting the result. Any missing data points will also affect the end quantification.
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        # remove the first and last bout to reduce errors and also copy the data
        data = self.remove_first_last_bout(bout_column=variable)

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)

        # Decode the whole dataset, count each state and the length of each individuals dataset to find the total percentage
        decoded_data = data.get_hmm_raw(hmm, variable=variable, t_bin=t_bin, func=func, t_column=t_column)
        states = decoded_data.groupby(decoded_data.index, sort=False)['state'].apply(list)

        df_lengths = []
        for l, id in zip(states, states.index):
            length = hmm_mean_length(l, delta_t = t_bin, raw=True) 
            length['id'] = [id] * len(length)
            df_lengths.append(length)

        grouped_data = pd.concat(df_lengths)
        grouped_data.rename(columns={'length_adjusted' : 'Length of state bout (mins)'}, inplace=True)
        grouped_data.set_index('id', inplace=True)

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_arg)+2, 4+2)
        
        fig, ax = plt.subplots(figsize=figsize)
        if scale is not None:
            plt.yscale(scale)

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        if facet_col:
            # merge the facet_col column and replace with the labelsBinned
            grouped_data = self.facet_merge(grouped_data, facet_col, facet_arg, facet_labels, hmm_labels = labels)
            sns.stripplot(data=grouped_data, x='state', y='Length of state bout (mins)', order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge = True)
            sns.boxplot(data=grouped_data, x='state', y='Length of state bout (mins)', order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict,
                            showcaps=False, showfliers=False, whiskerprops={'linewidth':0}, dodge = 0.4)

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)
            sns.stripplot(data=grouped_data, x='state', y='Length of state bout (mins)', order=labels, ax=ax, hue ='state', palette={k: v for k,v in zip(labels, colours)}, alpha=0.5, legend=False)
            sns.boxplot(data=grouped_data, x='state', y='Length of state bout (mins)', order=labels, ax=ax, hue ='state', palette={k: v for k,v in zip(labels, colours)},
                            showcaps=False, showfliers=False, whiskerprops={'linewidth':0})

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_hmm_quantify_transition(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, facet_labels = None, t_bin = 60, func = 'max', title = '', t_column = 't', grids = False, figsize=(0,0)):
        """
        Creates a quantification plot of the times each state is transitioned into as a percentage of the whole. 

            Args:
                hmm (hmmlearn.hmm.CategoricalHMM): This should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
                variable (str, optional): The column heading of the variable of interest. Default is "moving"
                labels (list[str], optional): The names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'] if a 4 state model. Default is None.
                colours (list[str/RGB], optional): The name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red). Default is None.
                    It accepts a specific colour or an array of numbers that are acceptable to Seaborn.
                facet_col (str, optional): The name of the column to use for faceting, must be from the metadata. Default is None.
                facet_arg (list, optional): The arguments to use for faceting. If None then all distinct groups will be used. Default is None.
                facet_labels (list, optional): The labels to use for faceting, these will be what appear on the plot. If None the labels will be those from the metadata. Default is None.
                t_bin (int, optional): The time in seconds you want to bin the movement data to. Default is 60 or 1 minute
                func (str, optional): When binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
                title (str, optional): The title of the plot. Default is an empty string.
                t_column (str, optional): The name of column containing the timing data (in seconds). Default is 't'
                grids (bool, optional): true/false whether the resulting figure should have grids. Default is False.
                figsize (tuple, optional): The size of the figure in inches. Default is (0, 0) which auto-adjusts the size.

        Returns:
            returns a Seaborn figure and pandas Dataframe with the mean length of each state per indivdual

        Notes:
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        # remove the first and last bout to reduce errors and also copy the data
        data = self.remove_first_last_bout(bout_column=variable)

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)

        # Decode the whole dataset, count each state and the length of each individuals dataset to find the total percentage
        decoded_data = data.get_hmm_raw(hmm, variable=variable, t_bin=t_bin, func=func, t_column=t_column)
        states = decoded_data.groupby(decoded_data.index, sort=False)['state'].apply(list)

        df_list = []
        for l, id in zip(states, states.index):
            length = hmm_pct_transition(l, total_states=list(range(len(labels)))) 
            length['id'] = [id] * len(length)
            df_list.append(length)

        grouped_data = pd.concat(df_list)
        grouped_data = grouped_data.set_index('id').stack().reset_index().set_index('id')
        grouped_data.rename(columns={'level_1' : 'state', 0 : 'Fraction of transitions into each state'}, inplace=True)

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_arg)+2, 4+2)
        
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylim([0, 1.01])

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        if facet_col:
            # merge the facet_col column and replace with the labels 
            grouped_data = self.facet_merge(grouped_data, facet_col, facet_arg, facet_labels, hmm_labels = labels)
            sns.stripplot(data=grouped_data, x='state', y='Fraction of transitions into each state', order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, alpha=0.5, legend=False, dodge = True)
            sns.pointplot(data=grouped_data, x='state', y='Fraction of transitions into each state', order=labels, hue=facet_col, hue_order=facet_labels, ax=ax, palette=palette_dict, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3, dodge = 0.4)

            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)
            sns.stripplot(data=grouped_data, x='state', y='Fraction of transitions into each state', order=labels, ax=ax, hue ='state', palette={k: v for k,v in zip(labels, colours)}, alpha=0.5, legend=False)
            sns.pointplot(data=grouped_data, x='state', y='Fraction of transitions into each state', order=labels, ax=ax, hue ='state', palette={k: v for k,v in zip(labels, colours)}, estimator = 'mean',
                            linestyle='none', errorbar= ("ci", 95), n_boot = 1000, markers="_", markersize=30, markeredgewidth=3)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_hmm_raw(self, hmm, variable = 'moving', colours = None, num_plots = 5, t_bin = 60, stim_df = None, func = 'max', show_movement = False, title = '', t_column = 't', grids = False, figsize=(0,0)):
        """Creates a plot showing the raw output from a hmm decoder.

        Args:
            data: The data to be plotted
            hmm: a trained categorical HMM from hmmlearn with the correct hidden states and emission states for your dataset
            variable: the name (as a string) of the column with the emussion data
            colours: the name of the colours you wish to represent the different states, must be the same length as labels
            tbin: the time in seconds the data should be binned to, Default is 60
            func: the function to apply to the aggregating column, i.e. "max", "mean", ... . Default is "max"

        Returns:
            A seaborn figure
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = None, col = colours)

        colours_index = {c : col for c, col in enumerate(colours)}

        if stim_df is not None:
            assert isinstance(stim_df, self.__class__), 'The stim_df dataframe is not behavpy class'

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

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (12, 4*(num_plots)+2)
        fig, ax = plt.subplots(figsize=figsize)


        states_list, time_list = hmm_decode(data, hmm, tbin, variable, func)
        time_list = list(time_list)
        list_states = list(range(len(hmm.transmat_)))
        if len(list_states) != len(colours):
            raise RuntimeError(
                "The number of colours do not match the number of states in the model"
            )

        rand_ind = random.choice(list(range(0, len(states_list))))

        st = states_list[rand_ind]
        time = time_list[rand_ind]
        time = np.array(time) / 86400

        for c, i in enumerate(colours):
            if c == 0:
                col = np.where(st == c, colours[c], np.NaN)
            else:
                col = np.where(st == c, colours[c], col)

        plt.figure(figsize=(80, 10))

        plt.scatter(time, st, s=50 * 2, marker="o", c=col)
        plt.plot(
            time,
            st,
            marker="o",
            markersize=0,
            mfc="white",
            mec="white",
            c="black",
            lw=1,
            ls="-",
        )

        plt.xlabel("Time (days)")
        plt.ylabel("State")

        return fig

    # def plot_hmm_response(self, mov_df, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, t_bin = 60, facet_labels = None, func = 'max', title = '', grids = False):

    # def plot_response_hmm_bouts(self, mov_df, hmm, variable = 'moving', labels = None, colours = None, x_limit = 30, t_bin = 60, func = 'max', title = '', grids = False, t_column = 't'):

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

    @staticmethod
    # Function to convert figure to image
    def _fig2img(fig, format='png'):
        buf = io.BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = PIL.Image.open(buf)
        return img

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

    def plot_overtime(self, variable:str, wrapped:bool = False, facet_col:None|str = None, facet_arg:None|str = None, facet_labels:None|str = None, facet_tile:None|str = None, avg_window:int = 180, day_length:int = 24, lights_off:int = 12, title:str = '', grids:bool = False, t_column:str = 't', figsize:tuple = (0,0)):
        """
        Plots a line hypnogram using seaborn, displaying rolling averages of a chosen variable over time.
        Optionally, the plot can be wrapped and faceted.

        Args:
            variable (str): The name of the variable in the DataFrame to plot. eg: asleep, moving
            wrapped (bool, optional): If True, time data will be wrapped to daylength. Default is False.
            facet_col (str, optional): The name of the column to use for faceting. Can be main column or from metadata. Default is None.
            facet_arg (list, optional): The arguments to use for faceting. Default is None.
            facet_labels (list, optional): The labels to use for faceting. Default is None.
            avg_window (int, optional): The window size for rolling average calculation. Default is 180.
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

        min_t = []
        max_t = []

        fig, ax = plt.subplots(figsize=figsize)

        for data, name, col in zip(d_list, facet_labels, list(self._get_colours(d_list))):

            gb_df, t_min, t_max, col, _ = self._generate_overtime_plot(data = data, name = name, col = col, var = variable, 
                                                                                    avg_win = avg_window, wrap = wrapped, day_len = day_length, 
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
            This function uses seaborn to create a boxplot and swarmplot. It allows to facet the data by a specific column.
            The function to be applied on the data is specified by the `fun` parameter.
        """

        # If facet_col is provided but facet arg is not, will automatically fill facet_arg and facet_labels with all the possible values
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if not isinstance(variable, list):
            variable = [variable]

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
            
            # apply the specified operation and add the specified columns from metadata
            #data = data.pivot(column = variable, function = fun).merge(data.meta.loc[:,facet_col], left_index=True, right_index=True)
            # we use the following line instead of the builtin pivot method because it allows us to calculate multiple functions
            grouped_data = data.groupby(data.index).agg(**data_summary).merge(data.meta[[facet_col]], left_index=True, right_index=True)

        # this applies in case we want to apply the specified information to ALL the data
        else:
            #no need to merge with metadata if facet_col is not specified
            #data = self.copy(deep=True).pivot(column = variable, function = fun)
            grouped_data = self.copy(deep=True).groupby(self.index).agg(**data_summary)
        
        # BOXPLOT
        fig_rows = len(variable)
        fig_cols = 1

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (2*len(facet_arg), 4*fig_rows)

        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize)
        
        # map the users labels onto he old facet_arg strings
        map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
        grouped_data[facet_col] = grouped_data[facet_col].map(map_dict)

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

            sns.boxplot(data=grouped_data, x=facet_col, y=plot_column, order = facet_arg, ax=ax, palette=self.attrs['sh_pal'], showcaps=False, showfliers=False, whiskerprops={'linewidth':0})
            sns.swarmplot(data=grouped_data, x=facet_col, y=plot_column, order = facet_arg, ax=ax, size=5, hue=facet_col, alpha=0.5, edgecolor='black', linewidth=1, palette=self.attrs['sh_pal'])

            ax.set_xticklabels(facet_labels)

        #Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=facet_labels)

        if grids: plt.grid(axis='y')
        plt.title(title)
        plt.tight_layout()

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_compare_variables(self, variables, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False):

        return self.plot_quantify(variable = variables, facet_col = facet_col, facet_arg = facet_arg, facet_labels = facet_labels, fun = fun, title = title, grids = grids)

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
        
        if facet_col:
            map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
            grouped_data[facet_col] = grouped_data[facet_col].map(map_dict)
            sns.boxplot(data=grouped_data, x=facet_col, y=plot_column, order = facet_labels, hue='has_interacted', hue_order=["Spon. Mov", "True Stimulus"], ax=ax, palette=['grey', 'red'], showcaps=False, showfliers=False, whiskerprops={'linewidth':0}, dodge = True)
            sns.swarmplot(data=grouped_data, x=facet_col, y=plot_column, order = facet_labels, hue='has_interacted', hue_order=["Spon. Mov", "True Stimulus"], ax=ax, size=5, alpha=0.5, edgecolor='black', linewidth=1, palette=['grey', 'red'], dodge = True)
            ax.set_xticklabels(facet_labels)

        else:
            sns.boxplot(data=grouped_data, y=plot_column, x='has_interacted', order=["Spon. Mov", "True Stimulus"], ax=ax, palette=['grey', 'red'], showcaps=False, showfliers=False, whiskerprops={'linewidth':0}, dodge = True)
            sns.swarmplot(data=grouped_data, y=plot_column, x='has_interacted', order=["Spon. Mov", "True Stimulus"], ax=ax, size=5, alpha=0.5, edgecolor='black', linewidth=1, palette=['grey', 'red'], dodge = True)

        #Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=["Spon. Mov", "True Stimulus"])

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data

    def plot_day_night(self, variable, facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, lights_off = 12, title = '', grids = False, figsize=(0,0)):
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
        data.add_day_phase(day_length = day_length, lights_off = lights_off)

        # Do the groupby operation on the entire dataframe and calculate mean and std on the given variable
        # grouped_data = data.groupby([data.index, 'phase']).agg(**data_summary)

        # Reset the index to bring 'phase' back as a column
        # grouped_data = grouped_data.reset_index()

        # Ensure 'phase' column is categorical for efficient memory usage - It already is categorical from the method
        # grouped_data['phase'] = grouped_data['phase'].astype('category')

        # takes subset of data if requested
        if facet_col and facet_arg:
            # Add the specified columns from metadata
            data = data.xmv(facet_col, facet_arg)
            grouped_data = data.groupby([data.index, 'phase']).agg(**data_summary).reset_index(level = 1).merge(data.meta[[facet_col]], left_index=True, right_index=True).reset_index()
        else:
            grouped_data = data.groupby([data.index, 'phase']).agg(**data_summary).reset_index()
        # ^^ for some reason to make the swarm plot work you have to have no personalised index

        # BOXPLOT
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (4*len(facet_arg), 4)

        fig, ax = plt.subplots(figsize=figsize)

        y_range, dtick = self._check_boolean(list(self[variable]))
        if y_range:
            plt.ylim(y_range)

        if facet_col:
            map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
            grouped_data[facet_col] = grouped_data[facet_col].map(map_dict)
            sns.boxplot(data=grouped_data, x="phase", order = ['light', 'dark'], y=plot_column, hue=facet_col, hue_order=facet_labels, ax=ax, palette="Paired", showcaps=False, showfliers=False, whiskerprops={'linewidth':0})
            sns.swarmplot(data=grouped_data, x="phase", order = ['light', 'dark'], y=plot_column, hue=facet_col, hue_order=facet_labels, ax=ax, size=5, alpha=0.5, edgecolor='black', linewidth=1, palette="Paired", dodge = True)
            # Customise legend values
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=facet_labels)
        else:
            sns.boxplot(data=grouped_data, x="phase", y=plot_column, order = ['light', 'dark'], ax=ax, palette=['yellow', 'darkgrey'], showcaps=False, showfliers=False, whiskerprops={'linewidth':0})
            sns.swarmplot(data=grouped_data, x="phase", y=plot_column, order = ['light', 'dark'], ax=ax, size=5, alpha=0.5, edgecolor='black', linewidth=1, palette=['yellow', 'darkgrey'])

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, data

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
        def _plot_single_actogram(dt, figsize, days, title, day_length):

            # (0,0) means automatic size
            if figsize == (0,0):
                figsize = (8, len(days)/3)

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

        # If facet_col is provided but facet arg is not, will automatically fill facet_arg and facet_labels with all the possible values
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        data = data.bin_time(mov_variable, bin_window*60, t_column = t_column)
        data.add_day_phase(time_column = f'{t_column}_bin')
        days = data["day"].unique()

        data = data.merge(data.meta, left_index=True, right_index=True)

        data["hours"] = (data[f'{t_column}_bin'] / (60*60))
        data["hours"] = data["hours"] - (data["day"]*24)

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


        if facet_col:
            figs = []
            for subplot in facet_arg:

                dt = data.loc [data[facet_col] == subplot]
                title = "%s - %s" % (title, subplot)
                fig = _plot_single_actogram(dt, figsize, days, title, day_length)
                plt.close()

                figs.append(fig)

            
            # Create a new figure to combine the figures
            cols, rows = 3, -(-len(facet_arg) // 3)
            c = []

            if figsize == (0,0):
                figsize = (8*rows, len(days))

            combined_fig = plt.figure(figsize = figsize )
            
            for pos, f in enumerate(figs):

                c.append( combined_fig.add_subplot(rows, cols, pos+1))
                c[-1].axis('off')  # Turn off axis
                c[-1].imshow( self._fig2img (f) )

            # Adjust the layout of the subplots in the combined figure
            #combined_fig.tight_layout()

            return combined_fig

        else:

            return _plot_single_actogram(data, figsize, days, title, day_length)

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
        def anticipation_score(d, mov_variable, start, end):
            
            def _ap_score(total, small):
                try:
                    return (small / total) * 100
                except ZeroDivisionError:
                    return 0

            d = d.t_filter(start_time = start[0], end_time = end)
            total = d.pivot(column = mov_variable, function = 'sum')
            
            d = d.t_filter(start_time = start[1], end_time = end)
            small = d.groupby(d.index).agg(**{
                    'moving_small' : (mov_variable, 'sum')
                    })
            d = total.join(small)
            d = d.dropna()
            
            return d[[f'{mov_variable}_sum', 'moving_small']].apply(lambda x: _ap_score(*x), axis = 1)


        # If facet_col is provided but facet arg is not, will automatically fill facet_arg and facet_labels with all the possible values
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg).merge(self.meta, left_index=True, right_index=True)
        else:
            data = self.copy(deep=True)

        data = data.dropna(subset=[mov_variable])
        data.wrap_time(inplace = True)

        # calculate anticipation score for lights_off
        start, end = [lights_off - 6, lights_off - 3], lights_off
        lights_off = pd.DataFrame( anticipation_score(data, mov_variable, start, end).rename("anticipation_score") )
        lights_off["phase"] = "Lights Off"

        # calculate anticipation score for lights_on
        start, end = [day_length - 6, day_length - 3], day_length
        lights_on = pd.DataFrame( anticipation_score(data, mov_variable, start, end).rename("anticipation_score") )
        lights_on["phase"] = "Lights On"

        # reset the index for both dataframes
        lights_off_reset = lights_off.reset_index()
        lights_on_reset = lights_on.reset_index()

        # concatenate along the row axis (i.e., append the dataframes one on top of the other) then set id as index
        anticipation_scores_df = pd.concat([lights_off_reset, lights_on_reset], axis=0).set_index("id")

        # Add the metadata columns and mark them as category
        dataset = anticipation_scores_df.join(self.meta)
        dataset[self.meta.columns] = dataset[self.meta.columns].astype('category')


        # BOXPLOT
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (2*len(facet_arg), 4)

        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(data=dataset, x="phase", y="anticipation_score", hue=facet_col, hue_order=facet_arg, palette=self._palette, ax=ax)
        #sns.swarmplot(data=dataset, x="phase", y="anticipation_score", hue=facet_col, size=8, alpha=0.5, edgecolor='black', linewidth=1, palette=self.palette, ax=ax)

        if grids: plt.grid(axis='y')
        plt.title(title)
        plt.tight_layout()

        # The score is in %
        plt.ylim(0,100)

        # reorder dataframe for stats output
        if facet_col: dataset = dataset.sort_values(facet_col)

        return fig, dataset
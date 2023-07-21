import pandas as pd
import numpy as np 
import warnings

#plotly
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative

#seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from math import floor, ceil, sqrt
from sys import exit
from scipy.stats import zscore
from functools import partial
from colour import Color
import re

from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.misc.bootstrap_CI import bootstrap

from ethoscopy.behavpy_core import behavpy_core

#fig to img
import io
import PIL


class behavpy_draw(behavpy_core):
    """
    Default drawing class containing some general methods that can be used by all children drawing classes
    """

    @staticmethod
    def _is_hex_color(s):
        """Returns True if s is a valid hex color. Otherwise False"""
        if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', s):
            return True
        return False


    @staticmethod
    def _rgb_to_hex(rgb_string):
        """
        Takes a string defining an RGB color and converts it a string of equivalent hex
        Input should be a string containing at least 3 numbers separated by a comma.
        The following input will all work:
        rgb(123,122,100)
        123,122,100
        """

        # Only keep digits and commas
        filtered_string = ''.join(c for c in rgb_string if c.isdigit() or c == ',')

        # Split the filtered string by comma and convert each part to integer
        rgb_values = list(map(int, filtered_string.split(',')))

        # Map the values to integers
        r, g, b = map(int, rgb_values)

        # Convert RGB to hex
        hex_string = '#{:02x}{:02x}{:02x}'.format(r, g, b)

        return hex_string

    @staticmethod
    def _get_colours(plot_list):
        """ returns a colour palette from plotly for plotly"""
        
        # a list of 11 colors from plotly qualitative package. Each color is a string specifying RGBs (e.g. 'rgb(136, 204, 238)')
        if len(plot_list) <= 11:
            return qualitative.Safe

        # a longer list of colors from plotly qualitative. Each color is a string specify HEX (e.g. '#2E91E5')
        elif len(plot_list) < 24:
            return qualitative.Dark24

        else:
            warnings.warn('Too many sub groups to plot with the current colour palette')
            exit()

    @staticmethod
    def _adjust_colours(colour_list):
        """
        Adjust a list of colours to make them lighter
        Each color in colour list can be a versatile input based on RGB or HEX. It will be converted using the Colour clas

        returns two lists: the starting list (converted to list of hex strings) and the lighten list (same format)
        """

        def adjust_color_lighten(r,g,b, factor):
            return [round(255 - (255-r)*(1-factor)), round(255 - (255-g)*(1-factor)), round(255 - (255-b)*(1-factor))]

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


class behavpy_seaborn(behavpy_draw):

    _canvas = 'seaborn'
    palette  = 'deep'
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

        # Set every nth x-tick label, in this example every 12th label
        n = 12
        x_labels = time_list[::n].astype(int)  # Get every nth label
        x_ticks = np.arange(0, len(time_list), n)  # Get every nth location

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (0.5*len(x_ticks), 0.1*len(id))

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, cmap="viridis", ax=ax)


        plt.xticks(ticks=x_ticks, labels=x_labels, rotation=0)
        plt.xlabel("ZT Time (Hours)")

        plt.yticks(ticks=np.arange(0, len(id), 2), labels=id[::-2], rotation=0)

        if title: fig.suptitle(title, fontsize=16)

        return fig

    def plot_overtime(self, variable, wrapped = False, facet_col = None, facet_arg = None, facet_labels = None, avg_window = 180, day_length = 24, lights_off = 12, title = '', grids = False, t_column = 't', figsize = (0,0)):
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

        Raises:
            AssertionError: If wrapped is not a boolean.

        Note:
            For accurate results, the data should be appropriately preprocessed to ensure that 't' values are
            in the correct format (seconds from time 0) and that 'variable' exists in the DataFrame.
        """

        assert isinstance(wrapped, bool)

        # If facet_col is provided but facet arg is not, will automatically fill facet_arg and facet_labels with all the possible values
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg).merge(self.meta, left_index=True, right_index=True)
        else:
            data = self.copy(deep=True)

        # creates the rolling average of the chosen variable over time
        rolling_col = data.groupby(data.index, sort = False)[variable].rolling(avg_window, min_periods = 1).mean().reset_index(level = 0, drop = True)
        data['rolling'] = rolling_col.to_numpy()

        # change t values to wrap data if requested
        if wrapped is True:
            data[t_column] = data[t_column] % (60*60*day_length)
        data[t_column] = data[t_column] / (60*60)

        #calculates the min, max time
        t_min = int(lights_off * floor(data[t_column].min() / lights_off))
        t_max = int(12 * ceil(data[t_column].max() / 12)) 
        x_ticks = np.arange(t_min, t_max, 6)

        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = ( 6 + 1/4 * len(x_ticks), 
                        4 + 1/32 * len(x_ticks) 
                        )


        fig, ax = plt.subplots(figsize=figsize)

        sns.lineplot(data, x='t', y='rolling', errorbar=self.error, hue=facet_col, hue_order=facet_arg, ax=ax, palette=self.palette)

        #Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=facet_labels)

        plt.ylim(0, 1)
        plt.xlim(t_min, t_max)

        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)
        plt.xlabel("ZT Time (Hours)")
        plt.ylabel(variable)

        plt.title(title)

        if grids:
            plt.grid(axis='y')

        thickness = -0.04
        offset = -0.15 #negative means below the figure
        # For every 24 hours, draw a rectangle from 0-12 (daytime) and another from 12-24 (nighttime)
        for i in circadian_bars(t_min, t_max, max_y = 0, day_length = day_length, lights_off = lights_off, canvas = 'seaborn'):
            # Daytime patch
            if i % day_length == 0:
                ax.add_patch(mpatches.Rectangle((i, offset), lights_off, thickness, color='black', alpha=0.4, clip_on=False, fill=None))
            else:
                # Nighttime patch
                ax.add_patch(mpatches.Rectangle((i, offset), day_length-lights_off, thickness, color='black', alpha=0.8, clip_on=False))

        return fig

    def plot_quantify(self, variable, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False, figsize = (0,0)):
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
            data_summary .update( {
                "%s_mean" % var : (var, 'mean'),
                "%s_std" % var : (var, 'std')
                } )

        # takes subset of data if requested
        if facet_col and facet_arg:

            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)
            
            # apply the specified operation and add the specified columns from metadata
            #data = data.pivot(column = variable, function = fun).merge(data.meta.loc[:,facet_col], left_index=True, right_index=True)
            # we use the following line instead of the builtin pivot method because it allows us to calculate multiple functions
            data = self.groupby(self.index).agg(**data_summary).merge(self.meta.loc[:,facet_col], left_index=True, right_index=True)
        
        # this possibility is actually never true because of the self._check_lists output that creates a facet_arg if facet_col exists
        elif facet_col and not facet_arg:
            data = self.groupby(self.index).agg(**data_summary).merge(self.meta.loc[:,facet_col], left_index=True, right_index=True)

        # this applies in case we want to apply the specified information to ALL the data
        else:
            #no need to merge with metadata if facet_col is not specified
            #data = self.copy(deep=True).pivot(column = variable, function = fun)
            data = self.copy(deep=True).groupby(self.index).agg(**data_summary)
        

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

        for ax, var in zip(axes, variable):
        
            plot_column = f'{var}_{fun}'

            y_range, dtick = self._check_boolean(list(self[var]))

            if y_range: 
                ax.set_ylim(y_range)
            #else:
             #   plt.ylim((0,12))

            sns.boxplot(data=data, x=facet_col, y=plot_column, ax=ax, palette=self.palette)
            sns.swarmplot(data=data, x=facet_col, y=plot_column, ax=ax, size=8, hue=facet_col, alpha=0.5, edgecolor='black', linewidth=1, palette=self.palette)

            ax.set_xticklabels(sorted(facet_labels))

        #Customise legend values
        #handles, _ = ax.get_legend_handles_labels()
        #ax.legend(handles=handles, labels=facet_labels)

        if grids: plt.grid(axis='y')
        plt.title(title)
        plt.tight_layout()

        # reorder dataframe for stats output
        if facet_col: data = data.sort_values(facet_col)

        return fig, data

    def plot_compare_variables(variables, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False, figsize = (0,0)):
        """
        Just an alias for plot_quantify that we use for retrocompatibility with the plotly class
        """
        return self.plot_quanitfy(variables, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False, figsize = (0,0))

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
        grouped_data = data.groupby([data.index, 'phase']).agg(**data_summary)

        # Reset the index to bring 'phase' back as a column
        grouped_data = grouped_data.reset_index()

        # Ensure 'phase' column is categorical for efficient memory usage
        grouped_data['phase'] = grouped_data['phase'].astype('category')

        # takes subset of data if requested
        if facet_col and facet_arg:
            # Add the specified columns from metadata
            data = grouped_data.merge(data.meta.loc[:,facet_col], left_on='id', right_index=True)
        else:
            data = grouped_data


        # BOXPLOT
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (2*len(facet_arg), 4)

        fig, ax = plt.subplots(figsize=figsize)

        y_range, dtick = self._check_boolean(list(self[variable]))
        plt.ylim(y_range)

        sns.boxplot(data=data, x="phase", y=plot_column, hue=facet_col, hue_order=facet_arg, ax=ax, palette="Paired", order=["light", "dark"])
        #sns.swarmplot(data=data, x="phase", y=plot_column, hue=facet_col, ax=ax, size=8, alpha=0.5, edgecolor='black', linewidth=1, palette=self.palette)

        #ax.set_xticklabels(sorted(facet_labels))

        #Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=facet_labels)

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: data = data.sort_values(facet_col)

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

        sns.boxplot(data=dataset, x="phase", y="anticipation_score", hue=facet_col, hue_order=facet_arg, palette=self.palette, ax=ax)
        #sns.swarmplot(data=dataset, x="phase", y="anticipation_score", hue=facet_col, size=8, alpha=0.5, edgecolor='black', linewidth=1, palette=self.palette, ax=ax)

        if grids: plt.grid(axis='y')
        plt.title(title)
        plt.tight_layout()

        # The score is in %
        plt.ylim(0,100)

        # reorder dataframe for stats output
        if facet_col: dataset = dataset.sort_values(facet_col)

        return fig, dataset


class behavpy_plotly(behavpy_draw):
    """
    plotly wrapper around behavpy_core
    """

    _canvas = 'plotly'

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


    @staticmethod
    def _generate_overtime_plot(data, name, col, var, avg_win, wrap, day_len, light_off, t_col):

        if len(data) == 0:
            print(f'Group {name} has no values and cannot be plotted')
            return None, None, None, None, None, None

        if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
            col = 'grey'

        if avg_win != False:
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

    def heatmap(self, variable = 'moving', t_column = 't', title = ''):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals using plotly
        
        Params:
        @variable = string, name for the column containing the variable of interest, the default is moving
        
        returns None
        """
        gbm, time_list, id = self.heatmap_dataset(variable, t_column)


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

    def plot_overtime(self, variable, wrapped = False, facet_col = None, facet_arg = None, facet_labels = None, avg_window = 180, day_length = 24, lights_off = 12, title = '', grids = False, t_column = 't'):
        assert isinstance(wrapped, bool)
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

    def plot_overtime_tile(self, variable, facet_tile, wrapped = False, facet_col = None, facet_arg = None, avg_window = 180, day_length = 24, lights_off = 12, title = '', grids = False, t_column = 't'):
        """ """
        assert isinstance(wrapped, bool)

        if facet_tile not in self.meta.columns:
            raise KeyError(f'Column "{facet_tile}" is not a metadata column')

        facet_labels = None

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # find the unique column variables and use to split df into tiled parts
        tile_list = list(set(self.meta[facet_tile].tolist()))

        tile_df = []
        for tile in tile_list:
            tile_df.append(self.xmv(facet_tile, tile))

        # split the tiled dfs into their facet counterparts, save their constituent parts as a nested list
        d_list = []
        name_list = []
        if facet_col is not None:
            for i, n in zip(tile_df, tile_list):
                small_list = []
                small_names = []
                for arg in facet_arg:
                    small_list.append(i.xmv(facet_col, arg))
                    small_names.append(f'{n}-{arg}')
                d_list.append(small_list)
                name_list.append(small_names)
        else:
            d_list = tile_df
            name_list = [str(n) for n in tile_list]

        col_list = self._get_colours(d_list)

        # generate a subplot figure with a single column
        fig = make_subplots(rows=len(tile_list), cols=1, shared_xaxes = True, subplot_titles = tile_list)

        max_var = []
        y_range, dtick = self._check_boolean(list(self[variable]))
        if y_range is not False:
            max_var.append(1)

        min_t = []
        max_t = []

        for c, (plot, tile_name, master_col) in enumerate(zip(d_list, name_list, col_list)):
            c = c+1
            if facet_col is not None:
                for facet_plot, facet_name in zip(plot, tile_name):
                    upper, trace, lower, maxV, t_min, t_max = self._generate_overtime_plot(data = facet_plot, name = facet_name, col = master_col, 
                                                                        var = variable, avg_win = avg_window, wrap = wrapped, 
                                                                        day_len = day_length, light_off = lights_off, t_col = t_column)
                    if upper is None:
                        continue
                    else:
                        fig.append_trace(upper, row = c, col = 1)
                        fig.append_trace(trace, row = c, col = 1)
                        fig.append_trace(lower, row = c, col = 1)
                        
                        min_t.append(t_min)
                        max_t.append(t_max)
                        max_var.append(maxV)

            else:
                upper, trace, lower, maxV, t_min, t_max = self._generate_overtime_plot(data = plot, name = tile_name, col = master_col, 
                                                                    var = variable, avg_win = avg_window, wrap = wrapped, 
                                                                    day_len = day_length, light_off = lights_off, t_col = t_column)
                if upper is None:
                    continue
                else:
                    fig.append_trace(upper, row = c, col = 1)
                    fig.append_trace(trace, row = c, col = 1)
                    fig.append_trace(lower, row = c, col = 1)

                    min_t.append(t_min)
                    max_t.append(t_max)
                    max_var.append(maxV)

        fig.update_xaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            range = [np.nanmin(min_t), np.nanmax(max_t)],
            tick0 = 0,
            dtick = day_length/4,
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            showgrid = False,
            linewidth = 2
        )

        fig.update_yaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            tick0 = 0,
            dtick = dtick,
            ticks = 'outside',
            tickwidth = 2,
            showgrid = grids,
            linewidth = 2
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
                    text = variable,
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
        bar_shapes, min_bar = circadian_bars(np.nanmin(min_t), np.nanmax(max_t), max_y = np.nanmax(max_var), day_length = day_length, lights_off = lights_off, split = len(tile_list))
        fig.update_layout(shapes=list(bar_shapes.values()))

        fig.update_annotations(font_size=18)
        fig['layout']['title'] = title
        fig['layout']['plot_bgcolor'] = 'white'
        if min_bar < 0:
            fig.update_yaxes(range = [min_bar, np.nanmax(max_var)+0.01])
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

        y_range, dtick = self._check_boolean(list(self[variables[0]]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[0], title = title, secondary = False, grid = grids)

        y_range, dtick = self._check_boolean(list(self[variables[-1]]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variables[-1], title = title, secondary = True, xdomain = f'x{axis_counter}', grid = grids)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_anticipation_score(self, mov_variable = 'moving', facet_col = None, facet_arg = None, facet_labels = None, day_length = 24, lights_off = 12, title = '', grids = False):
        """
        Plots the anticipatory activity score and returns a DataFrame containing the data.

        The anticipation score is the ratio of the final 3 hours of activity compared to the total 6 hours of activity prior to lights on/off.

        Parameters
        ----------
        mov_variable : str, default 'moving'
            Column name for the moving activity.
        facet_col : str, optional
            Column name to be used to group the data into separate box plots.
        facet_arg : list, optional
            List of arguments for grouping by facet_col.
        facet_labels : list, optional
            List of labels corresponding to facet_arg. 
        day_length : int, default 24
            The duration of a full day cycle.
        lights_off : int, default 12
            The time when lights are turned off.
        title : str, optional
            The title for the plot.
        grids : bool, default False
            Whether to display grid lines on the plot.

        Returns
        -------
        fig : plotly.graph_objs._figure.Figure
            The figure object of the plot.
        stats_df : pandas.DataFrame
            DataFrame containing the anticipatory phase scores.

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

    def plot_response_overtime(self, response_df, activity = 'inactive', mov_variable = 'moving', facet_col = None, facet_arg = None, facet_labels = None, title = '', t_column = 't', grids = False):
        """ plot function to measure the response rate of flies to a puff of odour from a mAGO or AGO experiment over the consecutive minutes active or inactive

        Params:
        @response_df = behavpy, behapy dataframe intially analysed by the puff_mago loading function
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

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
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
            
            if activity_choice == 'both':
                start_colours, end_colours = self._adjust_colours([col[0] for col in col_list])
                col_list = []
                colours_dict = {'start' : start_colours, 'end' : end_colours}
                for c in range(len(plot_list)):
                    start_color = colours_dict.get('start')[c]
                    end_color = colours_dict.get('end')[c]
                    N = len(facet_arg)
                    col_list.append([x.hex for x in list(Color(start_color).range_to(Color(end_color), N))])
            
            else:
                col_list = [[col] for col in self._get_colours(facet_arg)]
                end_colours, start_colours = self._adjust_colours([col[0] for col in col_list])
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
        """ A augmented version of plot_quanitfy that looks for true and false (spontaneous movement) interactions 
        
        Params:
        @response_col = string, the name of the column in the data with the response per interaction, column data should be in boolean form
        @facet_col = string, the name of the column in the metadata you wish to filter the data by
        @facet_arg = list, if not None then a list of items from the column given in facet_col that you wish to be plotted
        @facet_arg = list, if not None then a list of label names for facet_arg. If not provided then facet_arg items are used

        returns a plotly figure object
        """

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
        y_range, dtick = self._check_boolean(list(self[response_col]))
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = 'Resonse Rate', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = '')

        stats_dict = {}

        for data, name, col in zip(d_list, facet_labels, col_list):
            
            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue
            
            if len(list(set(data.has_interacted))) == 1:
                loop_itr = [1]
            else:
                loop_itr = [2, 1]

            for q in loop_itr:

                filtered = data[data['has_interacted'] == q]
                filtered = filtered.dropna(subset = [response_col])
                gdf = filtered.pivot(column = response_col, function = 'mean')
                median, q3, q1, zlist = self._zscore_bootstrap(gdf[f'{response_col}_mean'].to_numpy())
                stats_dict[f'{name}_{q}'] = zlist

                if q == 1:
                    col = col
                    lab = name
                else:
                    col = 'grey'
                    lab = f'{name} Spon. Mov'

                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [lab], colour =  col, showlegend = False, name = lab, xaxis = 'x'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = col, 
                showlegend = False, name = lab, xaxis = 'x'))

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df
        
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

        tile_df = []
        for tile in tile_list:
            tile_df.append(self.xmv(facet_tile, tile))

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

def behavpy ( data, meta, check = False, index= None, columns=None, dtype=None, copy=True, canvas='plotly' ):

    if canvas == 'plotly':
        return behavpy_plotly(data, meta, check, index, columns, dtype, copy)

    elif canvas == 'seaborn':
        return behavpy_seaborn(data, meta, check, index, columns, dtype, copy)

    elif canvas == None:
        return behavpy_core(data, meta, check, index, columns, dtype, copy)

    else:
        raise ValueError('Invalid canvas specified')
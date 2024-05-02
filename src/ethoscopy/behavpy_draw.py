import pandas as pd
import numpy as np 

import plotly.graph_objs as go 
from plotly.subplots import make_subplots
from plotly.express.colors import qualitative

from math import floor, ceil, sqrt
from scipy.stats import zscore
from functools import partial, update_wrapper
from colour import Color
from ethoscopy.misc.circadian_bars import circadian_bars
from ethoscopy.analyse import max_velocity_detector
from ethoscopy.misc.rle import rle
from ethoscopy.misc.bootstrap_CI import bootstrap

class behavpy_draw(behavpy_core):
    """
    Default drawing class containing some general methods that can be used by all children drawing classes
    """

    _hmm_colours = ['darkblue', 'dodgerblue', 'red', 'darkred']
    _hmm_labels = ['Deep sleep', 'Light sleep', 'Quiet awake', 'Active awake']

    @staticmethod
    def _check_rgb(lst):
        """ checks if the colour list is RGB plotly colours, if it is it changes it to its hex code """
        try:
            return [Color(rgb = tuple(np.array(eval(col[3:])) / 255)) for col in lst]
        except:
            return lst

    @staticmethod
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

    @staticmethod
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

class behavpy_seaborn(behavpy_draw):
    """
    seaborn wrapper around behavpy_core
    """

    _canvas = 'seaborn'
    _palette  = 'deep'
    _long_palette = _palette
    error = 'se'

    @staticmethod
    # Function to convert figure to image
    def _fig2img(fig, format='png'):
        buf = io.BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = PIL.Image.open(buf)
        return img

    def plot_overtime(self, variable, wrapped = False, facet_col = None, facet_arg = None, facet_labels = None, facet_tile = None, avg_window = 180, day_length = 24, lights_off = 12, title = '', grids = False, t_column = 't', figsize = (0,0)):
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
        if facet_col and facet_arg and facet_tile:
            data = self.xmv(facet_col, facet_arg).merge(self.meta[[facet_col, facet_tile]], left_index=True, right_index=True)
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg).merge(self.meta[[facet_col]], left_index=True, right_index=True)
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

        sns.lineplot(data, x='t', y='rolling', errorbar=self.error, hue=facet_col, hue_order=facet_arg, ax=ax, palette=self._palette)
        # sns.relplot(data, x='t', y='rolling', errorbar=self.error, hue=facet_col, hue_order=facet_arg, palette=self.palette, kind = 'line', row = facet_tile)

        #Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=facet_labels)

        yr, dt =  self._check_boolean(data[variable].tolist())
        if yr == [-0.025, 1.01]:
            plt.ylim(0, 1.01)
        plt.xlim(t_min, t_max)

        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)
        plt.xlabel("ZT Time (Hours)")
        plt.ylabel(variable)

        plt.title(title)

        if grids:
            plt.grid(axis='y')

        # For every 24 hours, draw a rectangle from 0-12 (daytime) and another from 12-24 (nighttime)
        bar_range, thickness, offset = circadian_bars(t_min, t_max, max_y = data[variable].max(), day_length = day_length, lights_off = lights_off, canvas = 'seaborn')
        for i in bar_range:
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
            data_summary.update( {
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
            grouped_data = data.groupby(data.index).agg(**data_summary).merge(data.meta[[facet_col]], left_index=True, right_index=True)
        
        # # this possibility is actually never true because of the self._check_lists output that creates a facet_arg if facet_col exists
        # elif facet_col and not facet_arg:
        #     data = self.groupby(self.index).agg(**data_summary).merge(self.meta.loc[:,facet_col], left_index=True, right_index=True)

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
            #else:
            #   plt.ylim((0,12))

            sns.boxplot(data=grouped_data, x=facet_col, y=plot_column, ax=ax, palette=self._palette)
            sns.swarmplot(data=grouped_data, x=facet_col, y=plot_column, ax=ax, size=5, hue=facet_col, alpha=0.5, edgecolor='black', linewidth=1, palette=self._palette)

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

    def plot_compare_variables(variables, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False, figsize = (0,0)):
        """
        Just an alias for plot_quantify that we use for retrocompatibility with the plotly class
        """
        return self.plot_quanitfy(variables, facet_col = None, facet_arg = None, facet_labels = None, fun = 'mean', title = '', grids = False, figsize = (0,0))

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
            grouped_data = self.groupby([self.index, 'has_interacted']).agg(**data_summary).copy(deep=True)

        # BOXPLOT
        # (0,0) means automatic size
        if figsize == (0,0):
            figsize = (3*len(facet_arg), 4)

        fig, ax = plt.subplots(figsize=figsize)

        plt.ylim(0, 1.03)
        map_dict = {1 : 'True Stimulus', 2 : 'Spon. Mov'}
        grouped_data['has_interacted'] = grouped_data['has_interacted'].map(map_dict)

        sns.boxplot(data=grouped_data, x=facet_col, y=plot_column, hue='has_interacted', hue_order=["Spon. Mov", "True Stimulus"], ax=ax, palette=['grey', 'red'])
        sns.swarmplot(data=grouped_data, x=facet_col, y=plot_column, hue='has_interacted', hue_order=["Spon. Mov", "True Stimulus"], ax=ax, size=5, alpha=0.5, edgecolor='black', linewidth=1, palette=['grey', 'red'], dodge = True)

        ax.set_xticklabels(facet_labels)

        #Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=["Spon. Mov", "True Stimulus"])

        plt.title(title)
        if grids: plt.grid(axis='y')

        # reorder dataframe for stats output
        if facet_col: 
            grouped_data = grouped_data.sort_values(facet_col)

        return fig, grouped_data


        return data

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
        plt.ylim(y_range)

        map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
        grouped_data[facet_col] = grouped_data[facet_col].map(map_dict)

        sns.boxplot(data=grouped_data, x="phase", y=plot_column, hue=facet_col, hue_order=facet_labels, ax=ax, palette="Paired")
        sns.swarmplot(data=grouped_data, x="phase", y=plot_column, hue=facet_col, hue_order=facet_labels, ax=ax, size=5, alpha=0.5, edgecolor='black', linewidth=1, palette="Paired", dodge = True)

        # Customise legend values
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=facet_labels)

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

        plt.ylim(0, 105)
        plt.xlim(np.min(x_ticks), np.max(x_ticks))
        plt.xticks(ticks=x_ticks, labels=x_ticks, rotation=0)

        plt.ylabel("Survival (%)")
        plt.xlabel("ZT (Hours)")

        plt.title(title)

        if grids:
            plt.grid(axis='y')

        thickness = 1.8
        offset = -2.5 #negative means below the figure
        # For every 24 hours, draw a rectangle from 0-12 (daytime) and another from 12-24 (nighttime)
        for i in circadian_bars(0, sur_df['hour'].max(), max_y = 0, day_length = day_length, lights_off = lights_off, canvas = 'seaborn'):
            # Daytime patch
            if i % day_length == 0:
                ax.add_patch(mpatches.Rectangle((i, offset), lights_off, thickness, color='black', alpha=0.4, clip_on=False, fill=None))
            else:
                # Nighttime patch
                ax.add_patch(mpatches.Rectangle((i, offset), day_length-lights_off, thickness, color='black', alpha=0.8, clip_on=False))

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

class behavpy_plotly(behavpy_draw):
    """
    plotly wrapper around behavpy_core
    """

    _canvas = 'plotly'
    _palette = 'Safe'
    _long_palette = 'Dark24'

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
        
        if CI:
            trace_box['mean'] = []
            trace_box['median'] = mean

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
            mean, median, q3, q1, zlist = self._zscore_bootstrap(gdf[f'{variable}_{fun}'].to_numpy())
            stats_dict[name] = zlist

            fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
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
                mean, median, q3, q1, zlist = self._zscore_bootstrap(t_gb[f'{variable}_mean'].to_numpy())
                stats_dict[f'{name}_{phase}'] = zlist

                if phase == 'light':
                    col = 'goldenrod'
                else:
                    col = 'black'

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
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
                mean, median, q3, q1, zlist = self._zscore_bootstrap(t_gb[f'{var}_mean'].to_numpy())
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

            mean, median_list, q3_list, q1_list, con_list, label_list = analysis(d_list, phase = phase)

            for c2, label in enumerate(facet_labels):

                if len(facet_arg) == 1:
                    col_index = c
                else:
                    col_index = c2

                stats_dict[f'{label}_{phase}'] = con_list[c2]

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median_list[c2]], q3 = [q3_list[c2]], q1 = [q1_list[c2]], 
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
                mean, median, q3, q1, zlist = self._zscore_bootstrap(gdf[f'{response_col}_mean'].to_numpy())


                stats_dict[lab] = zlist

                fig.add_trace(self._plot_meanbox(mean = [mean], median = [median], q3 = [q3], q1 = [q1], 
                x = [lab], colour =  qcol, showlegend = False, name = lab, xaxis = 'x'))

                fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [lab], colour = qcol, 
                showlegend = False, name = lab, xaxis = 'x'))

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    # def plot_survival()

def behavpy(data, meta, check = False, index= None, columns=None, dtype=None, copy=True, canvas='plotly'):

    if canvas == 'plotly':
        return behavpy_plotly(data, meta, check, index, columns, dtype, copy)

    elif canvas == 'seaborn':
        return behavpy_seaborn(data, meta, check, index, columns, dtype, copy)

    elif canvas == None:
        return behavpy_core(data, meta, check, index, columns, dtype, copy)

    else:
        raise ValueError('Invalid canvas specified')
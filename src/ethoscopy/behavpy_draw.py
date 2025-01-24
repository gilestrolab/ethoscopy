import numpy as np 
import pandas as pd
import re

import seaborn as sns
from plotly.express.colors import qualitative
from colour import Color
from math import sqrt, floor, ceil
from scipy.stats import zscore
from functools import partial

#fig to img
import io
import PIL

from ethoscopy.behavpy_core import behavpy_core
from ethoscopy.misc.general_functions import concat, bootstrap
from ethoscopy.misc.hmm_functions import hmm_pct_transition, hmm_mean_length #, hmm_pct_state

class behavpy_draw(behavpy_core):
    """
    Drawing class that extends @behavpy_core to provide visualization capabilities.

    This class inherits all data manipulation and analysis methods from behavpy_core and adds plotting functionality. 
    The key relationship between these classes is:

    1. Data Processing (behavpy_core):
        - Handles data filtering, grouping, and statistical calculations
        - Manages metadata and core data structures
        - Provides analysis methods like sleep detection and bout analysis

    2. Visualization (behavpy_draw):
        - Uses processed data from behavpy_core methods to create plots
        - Supports multiple plotting backends (Plotly and Seaborn)
        - Provides consistent styling and color schemes across plots
        - Handles faceting and complex multi-plot layouts

    The class is designed to seamlessly integrate data processing and visualization, allowing direct 
    chaining of analysis and plotting methods.

    Attributes:
        _hmm_colours (list[str]): A list of default colors for HMM states if only 4 states are provided.
        _hmm_labels (list[str]): A list of default labels for HMM states if only 4 states are provided.
    """

    _hmm_colours = ['darkblue', 'dodgerblue', 'red', 'darkred']
    _hmm_labels = ['Deep sleep', 'Light sleep', 'Quiet awake', 'Active awake']

    @staticmethod
    def _check_boolean(lst):
        """
        Checks if the input list contains only binary values (0 and 1).

        If the list is binary, returns the y-axis range and tick interval for scaling. 
        Otherwise, returns False for both.

        Args:
            lst (list): A list of numerical values.
        Returns:
            tuple: A tuple containing:
                - y_range (list or bool): The y-axis range for plotting if the list is binary, otherwise False.
                - dtick (float or bool): The tick interval for the y-axis if the list is binary, otherwise False.
        """
        if np.nanmax(lst) == 1 and np.nanmin(lst) == 0:
            y_range = [-0.025, 1.01]
            dtick = 0.2
        else:
            y_range = False
            dtick = False
        return y_range, dtick

    # Internal methods for checking data/arguments before plotting
    def _check_hmm_shape(self, hm, lab, col):
        """
        Validates the lengths of colors and labels for a Hidden Markov Model (HMM) plotting method.

        This method checks if the provided labels and colors match the number of states in the HMM. 
        If either is None, it populates them with default values based on the number of states. 
        The method also ensures that the lengths of labels and colors are equal, raising an error if they are not.

        Args:
            hm (HMM or list): A Hidden Markov Model or a list of HMMs. If a list is provided, 
                              the model with the maximum number of states will be selected.
            lab (list or None): A list of labels for the states. If None, defaults will be used.
            col (list or None): A list of colors for the states. If None, defaults will be used.

        Returns:
            tuple: A tuple containing:
                - _labels (list): The validated list of labels for the states.
                - _colours (list): The validated list of colors for the states.

        Raises:
            ValueError: If the lengths of labels and colors do not match.
        """
        if isinstance(hm, list):  # Select the HMM with the maximum number of states
            len_hmms = [h.transmat_.shape[0] for h in hm]
            hm = hm[len_hmms.index(max(len_hmms))]

        num_states = hm.transmat_.shape[0]

        if lab is not None and col is not None:
            if num_states == len(lab) and num_states == len(col):
                return lab, col
            else:
                raise ValueError('The number of labels and colours does not match the number of states in the HMM')

        elif num_states == 4:
            if lab == None:
                lab = self._hmm_labels
            if col == None:
                col = self._hmm_colours

        elif num_states != 4:
            if lab == None:
                lab = [f'state_{i}' for i in range(0, num_states)]
            if col == None:
                col = self._get_colours(hm.transmat_)[:len(lab)]

        if len(lab) != len(col):
            raise ValueError('Internal check failed: There are more or less states than colours')

        return lab, col

    def _check_lists_hmm(self, f_col, f_arg, f_lab, h, b):
        """
        Validates and prepares the facet arguments, labels, and HMM models for plotting.

        This method checks if the provided facet arguments match the labels and ensures that 
        the number of HMM models corresponds to the number of bin integers. If necessary, it 
        populates the lists with default values or raises errors for mismatches.

        Args:
            f_col (str or None): The name of the column used for faceting. If None, no faceting is applied.
            f_arg (list or None): A list of arguments for faceting. If None, it will be populated based on f_col.
            f_lab (list or None): A list of labels corresponding to f_arg. If None, it will be generated from f_arg.
            h (HMM or list): A Hidden Markov Model or a list of HMMs. If a list is provided, it checks for consistency.
            b (int orlist): A list of integers for binning the time. Must match the length of h if h is a list.

        Returns:
            tuple: A tuple containing:
                - f_arg (list): The validated list of facet arguments.
                - f_lab (list): The validated list of facet labels.
                - h_list (list): The list of HMMs, ensuring consistency with the number of facet arguments.
                - b_list (list): The list of bin integers, ensuring consistency with the number of facet arguments.

        Raises:
            AssertionError: If the lengths of HMMs and bin integers do not match the number of facet arguments.
            ValueError: If the lengths of facet arguments and labels do not match.
            KeyError: If any argument in f_arg is not found in the metadata column specified by f_col.
        """
        # Handle multiple HMMs
        if isinstance(h, list):
            assert isinstance(b, list), (
                "If providing a list of HMMs, also provide a list of ints to bin the time by (t_bin)"
            )
            if len(h) != len(b):
                raise ValueError('The number of HMMs and bin integers do not match')
            # If faceting then the user must provide an equal length list of bin times and facet args
            if f_col is not None:
                assert isinstance(f_arg, list), (
                    "If providing a list of HMMs, also provide a list of groups to filter by via facet_arg"
                )
                if len(h) != len(f_arg):
                    raise ValueError(
                        'There are not enough HMM models or bin integers for the different groups or vice versa'
                    )
                h_list, b_list = h, b
            # If just a list of HMMs but no facet, populate fake lists with None and names to trick the system
            else:
                f_arg = [None] * len(h)
                f_lab = [f'HMM-{i+1}' for i in range(len(h))]
                return f_arg, f_lab, h, b

        else:
            h_list = h
            b_list = b

        if f_col is None: # is no facet column, then return fake lists
            f_arg = [None]
            f_lab = ['']
            return f_arg, f_lab, h_list, b_list 

        if f_arg is not None: # check if all the facet args are in the meta column
            for i in f_arg:
                if i not in self.meta[f_col].tolist():
                    raise KeyError(f'Argument "{i}" is not in the meta column {f_col}')
                
        if f_col is not None and f_arg is not None and f_lab is not None: # is user provides all, just check for length match
            if len(f_arg) != len(f_lab):
                print("The facet labels don't match the length of the variables in the column. Using column variables names instead")
                f_lab = [str(arg) for arg in f_arg]
            return f_arg, f_lab, h_list, b_list 

        if f_col is not None and f_arg is not None and f_lab is None: # if user provides a facet column and args but no labels
            f_lab = [str(arg) for arg in f_arg]
            return f_arg, f_lab, h_list, b_list 

        if f_col is not None and f_arg is None and f_lab is None: # if user provides a facet column but no args or labels
            f_arg = list(set(self.meta[f_col].tolist()))
            f_lab = [str(arg) for arg in f_arg]
            return f_arg, f_lab, h_list, b_list 

    @staticmethod
    def _zscore_bootstrap(array:np.array, z_score:bool = True, second_array:np.array = None, min_max:bool = False):
        """
        Calculates the z-score of a given array, removes values beyond ±3 standard deviations, and performs bootstrapping on the remaining data.

        This method first computes the z-scores of the input array to identify outliers. 
        Values with an absolute z-score greater than 3 are excluded if `z_score` is set to `True`. After outlier removal, 
        it calculates the mean and median of the filtered data. Depending on the `min_max` flag, 
        it either determines the minimum and maximum values or performs bootstrapping to estimate the first and third quartiles. 
        The function is capable of handling a secondary array, which is filtered based on the same z-score criteria as the primary array.

        Args:
            array (np.array): A NumPy array of numerical values to be processed.
            z_score (bool, optional): Determines whether to apply z-score filtering. Defaults to True.
            second_array (np.array, optional): An additional NumPy array to be filtered in tandem with the primary array. Defaults to None.
            min_max (bool, optional): If `True`, returns the minimum and maximum values instead of bootstrapped quartiles. Defaults to False.

        Returns:
            tuple:
                - mean (float): The mean of the filtered array.
                - median (float): The median of the filtered array. If the median is outside the first and third quartiles, it is set to the mean.
                - q3 (float): The third quartile of the filtered array, derived either from bootstrapping or as the maximum value if `min_max` is `True`.
                - q1 (float): The first quartile of the filtered array, derived either from bootstrapping or as the minimum value if `min_max` is `True`.
                - zlist (np.array): The array after z-score filtering.
                - second_array (np.array, optional): The secondary array filtered based on the primary array's z-scores, returned only if provided.

        Raises:
            ZeroDivisionError: If the standard deviation of the array is zero, resulting in undefined z-scores.

        Notes:
            - If the input array has only one unique value, the mean, median, and quartiles are all set to that value.
            - Bootstrapping is performed using the `bootstrap` function from `ethoscopy.misc.bootstrap_CI`.
            - The function ensures robustness by handling cases where the median might fall outside the calculated quartiles.
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

                if min_max == True:
                    q3 = np.max(array)
                    q1 = np.min(array)

                else:
                    boot_array = bootstrap(zlist)
                    q3 = boot_array[1]
                    q1 = boot_array[0]

        except ZeroDivisionError:
            mean = median = q3 = q1 = 0
            zlist = array
        
        if median < q1 or median > q3:
            median = mean

        if second_array is not None:
            return mean, median, q3, q1, zlist, second_array
        else:
            return mean, median, q3, q1, zlist

    @staticmethod
    def _check_rgb(lst):
        """ Convert RGB colors to hex codes """
        try:
            return [Color(rgb = tuple(np.array(eval(col[3:])) / 255)) for col in lst]
        except:
            return lst

    def _get_colours(self, plot_list):
        """
        Retrieve a color palette based on the plotting backend and number of groups.

        Args:
            plot_list (list): List of items to determine the palette size.

        Returns:
            list: A list of color codes from Plotly or Seaborn palettes.

        Raises:
            IndexError: If the number of groups exceeds the maximum supported palette size (48).
        """
        pl_len = len(plot_list)

        if self.canvas == 'plotly':
        
            if pl_len <= len(getattr(qualitative, self.attrs['sh_pal'])):
                return getattr(qualitative, self.attrs['sh_pal'])
            elif pl_len <= len(getattr(qualitative, self.attrs['lg_pal'])):
                return getattr(qualitative, self.attrs['lg_pal'])
            elif pl_len <= 48:
                return qualitative.Dark24 + qualitative.Light24
            else:
                raise IndexError('Too many sub groups to plot with the current colour palette (max is 48)')

        if self.canvas == 'seaborn':
            if pl_len <= len(list(sns.color_palette(self.attrs['sh_pal']))):
                return list(sns.color_palette(self.attrs['sh_pal']))[:pl_len]
            else:
                return sns.color_palette('husl', len(plot_list))

    def _adjust_colours(self, colour_list):
        """
        Adjusts a list of colors by generating lighter versions alongside the original colors.
        This method takes a list of color representations, which can be either color names or hexadecimal codes.
        The color lightening is achieved by increasing the brightness of each RGB component by a specified factor.

        Args:
            colour_list (list[str]): A list of color names or hex codes (e.g., ['red', '#00FF00']).

        Returns:
            tuple:
                - start_colours (list[str]): Hex codes of the lighter versions of the input colors.
                - end_colours (list[str]): Hex codes of the original input colors.

        Raises:
            ValueError: If a color in `colour_list` cannot be converted to a valid color.
        """
        def adjust_color_lighten(r,g,b, factor):
            """
            Lightens a color by a given factor.
            Returns:
                list of int: Lightened RGB components.
            """
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

    # @staticmethod
    # def _is_hex_color(s):
    #     """
    #     Returns True if s is a valid hex color. Otherwise False

    #     Not currently used
    #     """
    #     if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', s):
    #         return True
    #     return False

    # @staticmethod
    # def _rgb_to_hex(rgb_string):
    #     """
    #     Takes a string defining an RGB color and converts it a string of equivalent hex
    #     Input should be a string containing at least 3 numbers separated by a comma.
    #     The following input will work:
    #     rgb(123,122,100)
    #     123,122,100

    #     Not currently used
    #     """

    #     # Only keep digits and commas
    #     filtered_string = ''.join(c for c in rgb_string if c.isdigit() or c == ',')

    #     # Split the filtered string by comma and convert each part to integer
    #     rgb_values = list(map(int, filtered_string.split(',')))

    #     # Map the values to integers
    #     r, g, b = map(int, rgb_values)

    #     # Convert RGB to hex
    #     hex_string = '#{:02x}{:02x}{:02x}'.format(r, g, b)

    #     return hex_string

    @staticmethod
    def _fig2img(fig, format='png'):
        """ Function to convert figure to still image within Seaborn plots """
        buf = io.BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = PIL.Image.open(buf)
        return img

    def save_figure(self, fig, path, width = None, height = None):
        """
        Save the produced plot from either Plotly or Seaborn to a specified path.
        Behaviour varies based in plotting backend (self.canvas)
        - **Plotly**:
            - If `path` ends with `.html`, the figure is saved as an interactive HTML file.
            - If 'path' ends in .jpg or .png or .svg saved as image
            - If `width` and `height` are provided, the figure is saved with the specified dimensions.
                - Only with non-html file types.
                - Otherwise, the figure is saved using Plotly default dimensions.
        
        - **Seaborn/Matplotlib**:
            - The figure is saved to the specified path with tight bounding boxes to minimize padding.

        Args:
            fig (Figure.object): 
                The figure object to save, either from Plotly or Seaborn/Matplotlib.
            
            path (str): 
                The destination file path where the figure will be saved. 
            
            width (int, optional): 
                The width of the exported image in pixels. 
                Applicable only for Plotly figures when saving as an image format. 
                Defaults to `None`.
            
            height (int, optional): 
                The height of the exported image in pixels. 
                Applicable only for Plotly figures when saving as an image format. 
                Defaults to `None`.

        Raises:
            AssertionError:
                If `path` is not a string.
            KeyError:
                If `self.canvas` is neither `'plotly'` nor `'seaborn'`.

        Returns:
            None

        Examples:
            >>> fig = behavpy.plot_overtime()
            >>> df.save_figure(fig, 'line_plot.png', width=800, height=600)
        """
        assert(isinstance(path, str))

        if self.canvas == 'plotly':

            if path.endswith('.html'):
                fig.write_html(path)
            elif width is None and height is None:
                fig.write_image(path)
            else:
                fig.write_image(path, width=width, height=height)
            print(f'Saved to {path}')

        if self.canvas == 'seaborn':

            fig.savefig(path, bbox_inches='tight')
            print(f'Saved to {path}')

    @staticmethod
    def _get_subplots(length):
        """ Get the nearest higher square number """
        square = np.sqrt(length) 
        closest = [floor(square)**2, ceil(square)**2]
        return int(sqrt(closest[1]))

    @staticmethod
    def _check_grey(name, col, response = False):
        """ Checks a string contains control like words and changes palette colour to match """
        if response is False:
            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower() or 'spon. mov' in name.lower():
                col = 'grey'
        else:
            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'black'    
            elif 'spon. mov' in name.lower():
                col = 'grey'
        return name, col

    # GENERAL PLOT HELPERS

    @staticmethod
    def facet_merge(data, meta, facet_col, facet_arg, facet_labels, hmm_labels = None):
        """ An internal method for joining a metadata column to its data for plotting purposes """
        # merge the facet_col column and replace with the labels
        data = data.join(meta[[facet_col]])
        data[facet_col] = data[facet_col].astype('category')
        map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
        data[facet_col] = data[facet_col].map(map_dict)
        if hmm_labels is not None:
            hmm_dict = {k : v for k, v in zip(range(len(hmm_labels)), hmm_labels)}
            data['state'] = data['state'].map(hmm_dict)
        return data

    def _generate_overtime_plot(self, data, name, col, var, avg_win, wrap, day_len, light_off, t_col):
        """
        Prepare and process data for generating an overtime plot.

        This method processes the input data by applying rolling averages, adjusting time indices,
        and calculating statistical measures such as mean and standard error. It then prepares
        the data for plotting using either Seaborn or Plotly based on the selected backend.

        Args:
            data (pd.DataFrame): The dataset to plot, centered around the cursor position.
            name (str): The identifier for the current group or category being plotted.
            col (str): The color designation for the plot elements.
            var (str): The variable/column name in `data` to be plotted over time.
            avg_win (int or bool): The window size for calculating the rolling average. If `False`, no averaging is applied.
            wrap (bool): Determines whether to wrap the time column based on `day_len`.
            day_len (int or bool): The length of the day cycle in seconds. If `False`, time wrapping is not performed.
            light_off (int): The time in seconds indicating when the lights are turned off.
            t_col (str): The name of the time column in `data`.

        Returns:
            tuple: Depending on the plotting backend, returns one of the following:
                - For Seaborn:
                    - gb_df (pd.DataFrame): Aggregated data with mean, standard deviation, and standard error.
                    - t_min (int or None): The minimum time value after adjustment.
                    - t_max (int or None): The maximum time value after adjustment.
                    - col (str): The color used for the plot.
                    - None
                - For Plotly:
                    - upper: Plotly trace for the upper bound of the error.
                    - trace: Plotly trace for the main line.
                    - lower: Plotly trace for the lower bound of the error.
                    - t_min (int or None): The minimum time value after adjustment.
                    - t_max (int or None): The maximum time value after adjustment.

        Raises:
            KeyError: If `self.canvas` is neither `'seaborn'` nor `'plotly'`.
        """

        if len(data) == 0:
            print(f'Group {name} has no values and cannot be plotted')
            return None, None, None, None, None

        name, col = self._check_grey(name, col)

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
            t_max = int(light_off * ceil(data[t_col].max() / light_off)) 
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

        if self.canvas == 'seaborn':
            return gb_df, t_min, t_max, col, None
        elif self.canvas == 'plotly':
            upper, trace, lower = self._plot_line(df = gb_df, x_col = t_col, name = name, marker_col = col)
            return upper, trace, lower, t_min, t_max
        else:
            KeyError(f'Wrong plot type in back end: {self.canvas}')

    def heatmap_dataset(self, variable, t_column):
        """
        Creates an aligned heatmap of the movement data binned into 30-minute intervals.

        Args:
            variable (str): 
                The name of the column containing the variable of interest to be binned and visualized.
            t_column (str): 
                The name of the time column in the DataFrame used for binning the data.

        Returns:
            tuple:
                - gbm (pd.Series): 
                    A grouped and binned series of the specified variable's mean values per ID.
                - time_list (np.ndarray): 
                    An array of time bins in hours, representing the binned intervals.
                - id_list (list): 
                    A list of unique IDs corresponding to each group in the heatmap.
        """

        heatmap_df = self.copy(deep = True)
        # change movement values from boolean to intergers and bin to 30 mins finding the mean
        if variable == 'moving':
            heatmap_df[variable] = np.where(heatmap_df[variable] == True, 1, 0)

        heatmap_df = heatmap_df.bin_time(variable, bin_secs = 1800, t_column = t_column)
        heatmap_df['t_bin'] = heatmap_df['t_bin'] / (60*60)
        # create an array starting with the earliest half hour bin and the last with 0.5 intervals
        start = heatmap_df['t_bin'].min().astype(int)
        end = heatmap_df['t_bin'].max().astype(int)
        time_list = np.array([x / 10 for x in range(start*10, end*10+5, 5)])
        time_map = pd.Series(time_list, 
                    name = 't_bin')

        def align_data(data):
            """
            Merges individual groups with the time map, filling missing points with NaN.

            Args:
                data (pd.DataFrame): 
                    Subset of the DataFrame for a specific group identified by 'id'.

            Returns:
                pd.DataFrame: 
                    Merged DataFrame with aligned time bins and filled NaN values.
                """
            index_name = data.index[0]

            df = data.merge(time_map, how = 'right', on = 't_bin', copy = False).sort_values(by=['t_bin'])

            # read the old id index lost in the merge
            old_index = pd.Index([index_name] * len(df.index), name = 'id')
            df.set_index(old_index, inplace =True)  

            return df                    

        heatmap_df = heatmap_df.groupby('id', group_keys = False).apply(align_data)

        gbm = heatmap_df.groupby(heatmap_df.index)[f'{variable}_mean'].apply(list)
        id_list = heatmap_df.groupby(heatmap_df.index)['t_bin'].mean().index.tolist()

        return gbm, np.array(time_list), id_list

    def _hmm_response(self, mov_df, hmm, variable, response_col, labels, facet_col, facet_arg, t_bin, facet_labels, func, t_column):
        """
        Processes HMM responses by decoding movement data and merging it with response data.

        Args:
            mov_df (pd.DataFrame): 
                DataFrame containing movement data to be decoded and analyzed.
            hmm (HMM or list of HMMs): 
                Hidden Markov Model(s) used for decoding the movement states.
            variable (str): 
                The column name in `mov_df` representing the variable to decode.
            response_col (str): 
                The column name in the response dataset to aggregate.
            labels (list of str): 
                List of labels corresponding to each state in the HMM.
            facet_col (str or None): 
                The metadata column used for faceting the plot. If None, no faceting is applied.
            facet_arg (list or None): 
                List of facet arguments corresponding to `facet_col`. If None, defaults are used.
            t_bin (int or list of int): 
                Time bin size(s) in seconds for aggregating data.
            facet_labels (list of str): 
                Labels for each facet used in the plot.
            func (str): 
                Aggregation function to apply (e.g., 'mean', 'max').
            t_column (str): 
                The column name representing time in the dataset.

        Returns:
            tuple:
                - grouped_data (pd.DataFrame): 
                    Aggregated DataFrame ready for plotting, indexed by specimen ID.
                - palette_dict (dict): 
                    Dictionary mapping plot categories to their corresponding colors.
                - h_order (list of str): 
                    Ordered list of categories for consistent plotting.

        Raises:
            ValueError:
                If the number of labels and colors does not match the number of HMM states.
            KeyError:
                If specified facet arguments are not found in the metadata.

        Notes:
            - Supports both single and multiple HMMs for decoding.
            - Automatically adjusts color palettes based on the number of states and facets.
            - Merges interaction data within the correct time bins to ensure accurate state assignments.
        """        
        data_summary = {
            "%s_mean" % response_col : (response_col, 'mean'),
            "%s_std" % response_col : (response_col, 'std'),
            }

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)
            mdata = mov_df.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)
            mdata = mov_df

        if facet_col is None:  # decode the whole dataset
            mdata = self.__class__(self._hmm_decode(mdata, hmm, t_bin, variable, func, t_column, return_type='table'), mdata.meta, check=True)
        else:
            if isinstance(hmm, list) is False: # if only 1 hmm but is faceted, decode as whole for efficiency
                mdata = self.__class__(self._hmm_decode(mdata, hmm, t_bin, variable, func, t_column, return_type='table'), mdata.meta, check=True)
            else:
                mdata = concat(*[self.__class__(self._hmm_decode(mdata.xmv(facet_col, arg), h, b, variable, func, t_column, return_type='table'), 
                                                mdata.meta, check=True) for arg, h, b in zip(facet_arg, hmm, t_bin)])

        def alter_merge(response, mov, tb):
            """
            Merge the two df's and check if the interaction happened in the right time point
            """
            response['bin'] = response['interaction_t'].map(lambda t:  tb * floor(t / tb))
            response.reset_index(inplace = True)

            merged = pd.merge(mov, response, how = 'inner', on = ['id', 'bin'])
            merged['t_check'] = merged.interaction_t + merged.t_rel
            merged['t_check'] = merged['t_check'].map(lambda t:  tb * floor(t / tb))

            merged['previous_state'] = np.where(merged['t_check'] > merged['bin'], merged['state'], merged['previous_state'])
            return merged

        if isinstance(t_bin, list) is False: # if only 1 bin but is faceted, apply to whole df
            data = self.__class__(alter_merge(data, mdata, t_bin), data.meta, check=True)
        else:
            data = concat(*[self.__class__(alter_merge(data.xmv(facet_col, arg), mdata.xmv(facet_col, arg), b), 
                                           data.meta, check=True) for arg, b in zip(facet_arg, t_bin)])

        grouped_data = data.groupby([data.index, 'previous_state', 'has_interacted']).agg(**data_summary)
        grouped_data = grouped_data.reset_index()
        grouped_data = grouped_data.set_index('id')
        grouped_data['state'] = grouped_data['previous_state']

        map_dict = {1 : 'True Stimulus', 2 : 'Spon. Mov.'}
        grouped_data['has_interacted'] = grouped_data['has_interacted'].map(map_dict)

        if facet_col is not None:
            h_order = [f'{lab} {ty}' for lab in facet_labels for ty in ["Spon. Mov.", "True Stimulus"]]
            palette = self._get_colours(facet_labels)
        else:
            h_order = ['Spon. Mov.', 'True Stimulus']
        palette = self._get_colours(facet_labels)
        palette = [x for xs in [[col, col] for col in palette] for x in xs]
        palette_dict = {name : self._check_grey(name, palette[c], response = True)[1] for c, name in enumerate(h_order)} # change to grey if control
        
        if facet_col is None:
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)
            grouped_data[''] = grouped_data['has_interacted']
        else:
            grouped_data = self.facet_merge(grouped_data, facet_col, facet_arg, facet_labels, hmm_labels = labels) # chnage to have meta given as arg
            grouped_data[facet_col] = grouped_data[facet_col].astype('str')
            grouped_data[facet_col] = grouped_data[facet_col] + " " + grouped_data['has_interacted']

        return grouped_data, palette_dict, h_order
    
    def _bouts_response(self, mov_df, hmm, variable, response_col, labels, colours, x_limit, t_bin, func, t_col):
        """
        Calculates the response rate over consecutive runs of a specified variable.

        Args:
            mov_df (pd.DataFrame): 
                DataFrame containing movement data to analyze.
            hmm (bool or HMM): 
                If provided, uses the HMM to decode states; otherwise, performs direct time binning.
            variable (str): 
                The column name representing the variable to analyze (e.g., 'moving').
            response_col (str): 
                The response column to aggregate and analyze.
            labels (list of str): 
                Labels corresponding to each state or activity type.
            colours (list of str): 
                List of colors assigned to each label for plotting purposes.
            x_limit (int): 
                Maximum allowed value for activity counts to filter the data.
            t_bin (int): 
                Time bin size in seconds for aggregating data.
            func (str): 
                Aggregation function to apply (e.g., 'mean', 'max').
            t_col (str): 
                The column name representing time in the dataset.

        Returns:
            tuple:
                - grouped_data (pd.DataFrame): 
                    Aggregated DataFrame containing mean responses, counts, confidence intervals, and labels.
                - palette_dict (dict): 
                    Dictionary mapping combined state and interaction types to their corresponding colors.
                - h_order (list[str]): 
                    Ordered list of categories for consistent plotting.

        Raises:
            KeyError:
                - If 'response_col' is not present in the DataFrame.
                - If provided 'activity' argument is invalid.
        """        
        data_summary = {
            "mean" : (response_col, 'mean'),
            "count" : (response_col, 'count'),
            "ci" : (response_col, bootstrap),
            }
        try:
            data = self.drop(columns=['moving']).copy(deep=True) # if 'moving' is in the response dataset then it messes up the merge
        except KeyError:
            data = self.copy(deep=True)
        mdata = mov_df.copy(deep=True)

        if hmm is not False:
            # copy and decode the dataset
            mdata = self.__class__(self._hmm_decode(mdata, hmm, t_bin, variable, func, t_col, return_type='table'), mdata.meta, check=True)
            var, newT = 'state', 'bin'
        else:
            mdata = mdata.bin_time(variable, t_bin, function = func, t_column = t_col)
            var, newT = f'{variable}_{func}', f'{t_col}_bin'

        # take the states and time per specimen and find the runs of states
        st_gb = mdata.groupby('id')[var].apply(np.array)
        time_gb = mdata.groupby('id')[newT].apply(np.array)
        all_runs = []
        for m, t, ids in zip(st_gb, time_gb, st_gb.index):
            spec_run = self._find_runs(m, t, ids)
            all_runs.append(spec_run)

        # take the arrays and make a dataframe for merging
        counted_df = pd.concat([pd.DataFrame(specimen) for specimen in all_runs])

        # change the time column to reflect the timing of counted_df
        data[t_col] = data['interaction_t'].map(lambda t:  t_bin * floor(t / t_bin))
        data.reset_index(inplace = True)

        # merge the two dataframes on the id and time column and check the response is in the same time bin or the next
        merged = pd.merge(counted_df, data, how = 'inner', on = ['id', t_col])
        merged['t_check'] = merged.interaction_t + merged.t_rel
        merged['t_check'] = merged['t_check'].map(lambda t:  t_bin * floor(t / t_bin))

        # change both previous if the interaction to stimulus happens in the next time bin
        merged['previous_activity_count'] = np.where(merged['t_check'] > merged[t_col], merged['activity_count'], merged['previous_activity_count'])
        merged['previous_moving'] = np.where(merged['t_check'] > merged[t_col], merged['moving'], merged['previous_moving'])
        merged = merged[merged['previous_activity_count'] <= x_limit]
        merged.dropna(subset = ['previous_moving', 'previous_activity_count'], inplace=True)
        merged['previous_activity_count'] = merged['previous_activity_count'].astype(int)

        # groupby the columns of interest, and find the mean and bootstrapped 95% CIs
        grouped_data = merged.groupby(['previous_moving', 'previous_activity_count', 'has_interacted']).agg(**data_summary)
        grouped_data = grouped_data.reset_index()
        grouped_data[['y_max', 'y_min']] = pd.DataFrame(grouped_data['ci'].tolist(), index =  grouped_data.index)
        grouped_data.drop('ci', axis = 1, inplace = True)
        grouped_data['moving'] = grouped_data['previous_moving']
        map_dict = {1 : 'True Stimulus', 2 : 'Spon. Mov.'}
        grouped_data['has_interacted'] = grouped_data['has_interacted'].map(map_dict)

        if hmm is False:
            grouped_data['facet_col'] = [labels] * len(grouped_data)
            return grouped_data

        hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
        grouped_data['state'] = grouped_data['moving'].map(hmm_dict)
        grouped_data['label_col'] =  grouped_data['state'] + "-" + grouped_data['has_interacted']

        # create the order of plotting and double the colours to assign grey to false stimuli
        h_order = [f'{lab}-{ty}' for lab in labels for ty in ["Spon. Mov.", "True Stimulus"]]
        palette = [x for xs in [[col, col] for col in colours] for x in xs]
        palette_dict = {name : self._check_grey(name, palette[c], response = True)[1] for c, name in enumerate(h_order)} # change to grey if control

        return grouped_data, palette_dict, h_order

    def _internal_bout_activity(self, mov_df, activity, variable, response_col, facet_col, facet_arg, facet_labels, x_limit, t_bin, t_column):
        """
        Analyze and aggregate response data based on activity bouts for visualization.

        Args:
            mov_df (pd.DataFrame): 
                DataFrame containing movement data to analyze.
            activity (str): 
                Type of activity to filter on. Must be one of `'inactive'`, `'active'`, or `'both'`.
            variable (str): 
                The column name in `mov_df` representing the variable to analyze.
            response_col (str): 
                The column name in `mov_df` representing the response to aggregate and analyze.
            facet_col (str or None): 
                Metadata column used for faceting the plot. If `None`, no faceting is applied.
            facet_arg (list or None): 
                List of facet arguments corresponding to `facet_col`. If `None`, defaults are used.
            facet_labels (list or None): 
                List of labels for the facets. If `None`, labels are derived from `facet_arg`.
            x_limit (int): 
                Maximum allowed value for activity counts to filter the data.
            t_bin (int): 
                Time bin size in seconds for aggregating data.
            t_column (str): 
                The column name representing time in the dataset.

        Returns:
            tuple:
                - grouped_data (pd.DataFrame): 
                    Aggregated data ready for plotting, indexed by specimen ID.
                - h_order (list[str)]: 
                    Ordered list of categories for consistent plotting.
                - palette_dict (dict): 
                    Dictionary mapping plot categories to their corresponding colors.

        Raises:
            KeyError:
                - If `activity` is not one of `'inactive'`, `'active'`, or `'both'`.
            ValueError:
                - If the number of facet arguments and labels do not match.
        
        Notes:
            - If `activity` is set to `'both'` and `facet_col` is provided, faceting is disabled 
        """

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        activity_choice = {'inactive' : 0, 'active' : 1, 'both' : (0, 1)}
        if activity not in activity_choice.keys():
            raise KeyError(f'activity argument must be one of {*activity_choice.keys(),}')
        if activity == 'both' and facet_col is not None:
            print('When plotting both inactive and active runs you can not use facet_col. Reverted to None')
            facet_col, facet_arg, facet_labels = None, [None], ['inactive', 'active']

        if facet_col and facet_arg:
            rdata = self.xmv(facet_col, facet_arg)
            # iterate over the filters and call the analysing function
            dfs = [rdata._bouts_response(mov_df=mov_df.xmv(facet_col, arg), hmm = False,
                    variable=variable, response_col=response_col, labels=lab, colours=[], 
                    x_limit=x_limit, t_bin=t_bin, func='max', t_col=t_column) for arg, lab in zip(facet_arg, facet_labels)]
            grouped_data = pd.concat(dfs)
        else:
            grouped_data = self._bouts_response(mov_df=mov_df, hmm = False,
                                                variable=variable, response_col=response_col, labels=[], colours=[], 
                                                x_limit=x_limit, t_bin=t_bin, func='max', t_col=t_column)
            inverse_dict = {v: k for k, v in activity_choice.items()}
            grouped_data['facet_col'] = grouped_data['previous_moving'].map(inverse_dict)

        # Get colours and labels, syncing them together and replacing False Stimuli with a grey colour
        grouped_data['label_col'] =  grouped_data['facet_col'] + " " + grouped_data['has_interacted']
        if facet_col:
            palette = [x for xs in [[col, col] for col in self._get_colours(facet_labels)] for x in xs]
            h_order = [f'{lab} {ty}' for lab in facet_labels for ty in ["Spon. Mov.", "True Stimulus"]]
        else:
            palette = [x for xs in [[col, col] for col in self._get_colours(list(inverse_dict.values()))] for x in xs]
            h_order = [f'{lab} {ty}' for lab in list(inverse_dict.values()) for ty in ["Spon. Mov.", "True Stimulus"]]
        palette_dict = {name : self._check_grey(name, palette[c], response = True)[1] for c, name in enumerate(h_order)} # change to grey if control

        # If not both filter the dataset
        if isinstance(activity_choice[activity], int):
            grouped_data = grouped_data[grouped_data['previous_moving'] == activity_choice[activity]]
        
        return grouped_data, h_order, palette_dict

    def _internal_plot_response_overtime(self, t_bin_hours, response_col, interaction_id_col, facet_col, facet_arg, facet_labels, func, t_column):
        """
        Internal method to curate and analyze data for both Plotly and Seaborn versions of `plot_response_overtime`.

        Args:
            t_bin_hours (int): 
                The number of hours per bin for aggregating the response data.
            response_col (str): 
                The name of the column containing the response data to aggregate.
            interaction_id_col (str): 
                The column name indicating the type of interaction (e.g., stimulus type).
            facet_col (str or None): 
                The name of the metadata column to use for faceting the plot. If `None`, no faceting is applied.
            facet_arg (list or None): 
                A list of arguments used to filter data based on `facet_col`. If `None`, all categories are included.
            facet_labels (list or None): 
                A list of labels corresponding to the `facet_arg` for labeling facets. If `None`, default labels are used.
            func (str): 
                The aggregation function to apply to the binned data (e.g., `'mean'`, `'sum'`).
            t_column (str): 
                The column name representing the time data to bin.

        Returns:
            tuple:
                - df (behavpy_draw): 
                    An instance of `behavpy_draw` containing the grouped and aggregated data ready for plotting.
                - h_order (list[str]): 
                    The order of hue categories for plotting, formatted as `"<facet_label>-<stimulus_type>"`.
                - palette (list[str]): 
                    A list of color codes corresponding to each hue category, adjusted for control groups if applicable.

        Raises:
            KeyError:
                - If `facet_col` or `interaction_id_col` is provided but does not exist in the dataset.
            ValueError:
                - If the number of `facet_arg` does not match the number of `facet_labels`.
                - If unexpected stimulus types are encountered in `interaction_id_col`.
        """
        data_summary = {
            "mean" : (f'{response_col}_{func}', 'mean'),
            "count" : (f'{response_col}_{func}', 'count')
            }

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        if len(set(data[interaction_id_col])) == 1: # if only stimulus type in the dataset
            # get colours
            palette = self._get_colours(facet_labels)
            h_order = [f'{lab}-{ty}' for lab in facet_labels for ty in ["True Stimulus"]]

            # find the average response per hour per specimen
            data = data.bin_time(response_col, (60*60) * t_bin_hours, function = 'mean', t_column = t_column)
            if facet_col and facet_arg:
                data.meta['new_facet'] = data.meta[facet_col] + '-' + 'True Stimulus'
            else:
                data.meta['new_facet'] = '-True Stimulus'

        else:
            # get colours and double them to change to grey later
            palette = [x for xs in [[col, col] for col in self._get_colours(facet_labels)] for x in xs]
            h_order = [f'{lab}-{ty}' for lab in facet_labels for ty in ["Spon. Mov.", "True Stimulus"]]

            # filter into two stimulus and find average per hour per specimen
            data1 = self.__class__(data[data[interaction_id_col]==1].bin_time(response_col, (60*60) * t_bin_hours, function = func, t_column = t_column), data.meta)
            data2 = data[data[interaction_id_col]==2].bin_time(response_col, (60*60) * t_bin_hours, function = func, t_column = t_column)

            # change the id of the false stimuli
            meta2 = data.meta.copy(deep=True)
            meta2['ref_id'] = meta2.index + '_2'
            map_dict = meta2[['ref_id']].to_dict()['ref_id']
            meta2.rename(columns={'ref_id' : 'id'}, inplace=True)
            meta2.index = meta2['id']
            data2.index = data2.index.map(map_dict)

            if facet_col and facet_arg:
                data1.meta['new_facet'] = data1.meta[facet_col] + '-' + 'True Stimulus'
                meta2['new_facet'] = meta2[facet_col] + '-' + 'Spon. Mov.'  
            else:
                data1.meta['new_facet'] = '-True Stimulus'
                meta2['new_facet'] = '-Spon. Mov.'  

            data = concat(data1, self.__class__(data2, meta2))

        palette= [self._check_grey(name, palette[c], response = True)[1] for c, name in enumerate(h_order)] # change to grey if control

        grouped_data = data.groupby([data.index, 't_bin']).agg(**data_summary).reset_index(level=1)
        df = self.__class__(grouped_data, data.meta)
        df.rename(columns={'mean' : 'Response Rate'}, inplace=True)

        return df, h_order, palette

    def _internal_plot_habituation(self, plot_type, t_bin_hours, response_col, interaction_id_col, facet_col, facet_arg, facet_labels, x_limit, t_column): 
        """
        Internal method to curate and analsze data for both Plotly and Seaborn versions of `plot_habituation`.

        Args:
            plot_type (str): 
                Determines the type of plotting. Must be either `'time'` for time-based binning or `'number'` for stimulus number binning.
            t_bin_hours (int): 
                The number of hours per bin for aggregating the response data.
            response_col (str): 
                The name of the column containing the response data to aggregate (e.g., `'has_responded'`).
            interaction_id_col (str): 
                The column name indicating the type of interaction (e.g., `'interaction_id'`).
            facet_col (str or None): 
                The name of the metadata column to use for faceting the plot. If `None`, no faceting is applied.
            facet_arg (list or None): 
                A list of arguments used to filter data based on `facet_col`. If `None`, all categories are included.
            facet_labels (list or None): 
                A list of labels corresponding to the `facet_arg` for labeling facets. If `None`, default labels are used.
            x_limit (int or bool): 
                The maximum allowed value for the plot's x-axis. If `False`, it is set to the maximum value found in the data.
            t_column (str): 
                The column name representing the time data to bin (e.g., `'t_bin'`).

        Returns:
            tuple:
                - grouped_final (pd.DataFrame): 
                    Aggregated DataFrame containing mean responses, counts, confidence intervals, and stimulus counts, ready for plotting.
                - h_order (list[str]): 
                    The order of hue categories for plotting, formatted as `"<facet_label>-<stimulus_type>"`
                - palette_dict (dict): 
                    Dictionary mapping plot categories to their corresponding colors, adjusted for control groups if applicable.
                - x_max (int): 
                    The maximum value for the x-axis after applying `x_limit`.
                - plot_label (str): 
                    The label used for plotting based on `plot_type`, either `'Hours {t_bin_hours} post first stimulus'` or `'Stimulus number post first'`.

        Raises:
            KeyError:
                - If `plot_type` is not one of `'time'` or `'number'`.
            ValueError:
                - If the lengths of `facet_arg` and `facet_labels` do not match when faceting is applied.
        """
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        plot_choice = {'time' : f'Hours {t_bin_hours} post first stimulus', 'number' : 'Stimulus number post first'}

        if plot_type not in plot_choice.keys():
            raise KeyError(f'activity argument must be one of {*plot_choice.keys(),}')

        data_summary = {
            "mean" : (response_col, 'mean'),
            "count" : (response_col, 'count'),
            'ci' : (response_col, bootstrap),
            "stim_count" : ('stim_count', 'sum')
            }
        map_dict = {1 : 'True Stimulus', 2 : 'Spon. Mov.'}

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        def get_response(int_data, ptype, time_window_length, resp_col, t_col):
            # bin the responses per amount of hours given and find the mean per specimen
            if ptype == 'time':
                hour_secs = time_window_length * 60 * 60
                int_data[plot_choice[plot_type]] = int_data[t_col].map(lambda t: hour_secs * floor(t / hour_secs)) 
                min_hour = int_data[plot_choice[plot_type]].min()
                int_data[plot_choice[plot_type]] = (int_data[plot_choice[plot_type]] - min_hour) / hour_secs
                gb = int_data.groupby(plot_choice[plot_type]).agg(**{
                            'has_responded' : (resp_col, 'mean'),
                            'stim_count' : (resp_col, 'count')
                })
                return gb
            # Sort the responses by time, assign int according to place in the list, return as dataframe 
            elif ptype == 'number':
                int_data = int_data.sort_values(t_col)
                int_data['n_stim'] = list(range(1, len(int_data)+1))
                return pd.DataFrame(data = {'has_responded' : int_data['has_responded'].tolist(), plot_choice[plot_type] : int_data['n_stim'].tolist(), 
                                'stim_count' : [1] * len(int_data)}).set_index(plot_choice[plot_type])

        grouped_data = data.groupby([data.index, interaction_id_col]).apply(partial(get_response, ptype=plot_type, time_window_length=t_bin_hours,
                                                        resp_col=response_col, t_col=t_column), include_groups=False)
        grouped_data = self.__class__(grouped_data.reset_index().set_index('id'), data.meta, check=True)

        # reduce dataset to the maximum value of the True stimulus (reduces computation time)
        if x_limit is False:
            x_max = np.nanmax(grouped_data[grouped_data[interaction_id_col] == 1][plot_choice[plot_type]])
        else:
            x_max = x_limit
            
        grouped_data = grouped_data[grouped_data[plot_choice[plot_type]] <= x_max]
        # map stim names and create column to facet by
        grouped_data[interaction_id_col] = grouped_data[interaction_id_col].map(map_dict)
        if facet_col:
            grouped_data = self.facet_merge(grouped_data, self.meta, facet_col, facet_arg, facet_labels)
            grouped_data[facet_col] = grouped_data[facet_col].astype(str) + "-" + grouped_data[interaction_id_col]
        else:
            facet_col = 'stim_type'
            grouped_data[facet_col] = "-" + grouped_data[interaction_id_col]

        grouped_final = grouped_data.groupby([facet_col, plot_choice[plot_type]]).agg(**data_summary).reset_index(level=1)
        grouped_final[['y_max', 'y_min']] = pd.DataFrame(grouped_final['ci'].tolist(), index =  grouped_final.index)
        grouped_final.drop('ci', axis = 1, inplace = True)

        palette = [x for xs in [[col, col] for col in self._get_colours(facet_labels)] for x in xs]
        h_order = [f'{lab}-{ty}' for lab in facet_labels for ty in ["Spon. Mov.", "True Stimulus"]]
        palette_dict = {name : self._check_grey(name, palette[c], response = True)[1] for c, name in enumerate(h_order)} # change to grey if control

        return grouped_final, h_order, palette_dict, x_max, plot_choice[plot_type]

    def _internal_plot_quantify(self, variable, facet_col, facet_arg, facet_labels, fun):
        """
        Internal method for generating quantifying plots by calculating the average and standard deviation of specified variables.
        This method aggregates the data based on the provided variables and applies the specified aggregation function.

        Args:
            variable (str or list[str]):
                The name(s) of the column(s) in the dataset to be quantified.
            facet_col (str or None): 
                The name of the metadata column to use for faceting the plot. If `None`, no faceting is applied.
            facet_arg (list or None): 
                A list of arguments used to filter data based on `facet_col`. If `None`, all categories are included.
            facet_labels (list or None): 
                A list of labels corresponding to the `facet_arg` for labeling facets. If `None`, default labels are used.
            fun (str):
                The aggregation function to apply to the specified variables. Common options include `'mean'`, `'sum'`, etc.

        Returns:
            tuple:
                - grouped_data (pd.DataFrame):
                    A DataFrame containing the aggregated data with calculated statistics.
                - palette_dict (dict):
                    A dictionary mapping facet labels to their corresponding color codes. This palette is used to ensure
                    consistent coloring across different facets in the plots.
                - facet_labels (list of str):
                    The list of facet labels used in the aggregation. This may be modified.
                - variable (list[str]):
                    The list of variables that were aggregated. Ensures consistency even if a single variable was provided as a string.

        Raises:
            KeyError:
                - If `facet_col` is provided but does not exist in the dataset's metadata.
            ValueError:
                - If the `fun` argument does not correspond to a valid aggregation function.     
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
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        # applt the averaging function by index per variable
        grouped_data = data.groupby(data.index).agg(**data_summary)

        if facet_col:
            palette = self._get_colours(facet_labels)
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control
            grouped_data = self.facet_merge(grouped_data, self.meta, facet_col, facet_arg, facet_labels)
        else:
            palette = self._get_colours(variable)
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(variable)} # change to grey if control

        return grouped_data, palette_dict, facet_labels, variable

    def _internal_plot_response_quantify(self, response_col, facet_col, facet_arg, facet_labels):
        """
        Internal method for generating quantifying plots for response datasets by calculating the average 
            and standard deviation of specified variables.
        This method aggregates the data based on the provided response column and finds the mean and standard deviation.

        Args:
            response_col (str):
                The name of the column in the dataset that contains the responses to a stimuli, typically boolean.
            facet_col (str or None): 
                The name of the metadata column to use for faceting the plot. If `None`, no faceting is applied.
            facet_arg (list or None): 
                A list of arguments used to filter data based on `facet_col`. If `None`, all categories are included.
            facet_labels (list or None): 
                A list of labels corresponding to the `facet_arg` for labeling facets. If `None`, default labels are used.

        Returns:
            tuple:
                - grouped_data (pd.DataFrame):
                    A DataFrame containing the aggregated data with calculated statistics.
                - h_order (list[str]): 
                    The order of hue categories for plotting, formatted as `"<facet_label>-<stimulus_type>"`
                - palette_dict (dict):
                    A dictionary mapping facet labels to their corresponding color codes. This palette is used to ensure
                    consistent coloring across different facets in the plots.
        Raises:
            KeyError:
                - If `facet_col` or 'response_col' does not exist in the dataset's metadata.   
        """

        if response_col not in self.columns.tolist():
            raise KeyError(f'The column you gave {response_col}, is not in the data. Check you have analyed the dataset with stimulus_response')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

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

        map_dict = {1 : 'True Stimulus', 2 : 'Spon. Mov.'}
        grouped_data['has_interacted'] = grouped_data['has_interacted'].map(map_dict)
        if facet_col:
            grouped_data['facet_col'] =  grouped_data[facet_col].astype(str) + "-" + grouped_data['has_interacted']
        else:
            grouped_data['facet_col'] =  "-" + grouped_data['has_interacted']

        palette = [x for xs in [[col, col] for col in self._get_colours(facet_labels)] for x in xs]
        h_order = [f'{lab}-{ty}' for lab in facet_labels for ty in ["Spon. Mov.", "True Stimulus"]]       
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(h_order)} # change to grey if control

        return grouped_data, h_order, palette_dict

    def _internal_plot_day_night(self, variable, facet_col, facet_arg, facet_labels, day_length, lights_off, t_column):
        """ internal method to calculate the average variable amounts for the day and night, for use in plot_day_night, plotly and seaborn """
        
        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)
        data_summary = {
            "%s_mean" % variable : (variable, 'mean'),
            "%s_std" % variable : (variable, 'std'),
            }

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        #Add phase information to the data
        data.add_day_phase(day_length = day_length, lights_off = lights_off, t_column = t_column)

        grouped_data = data.groupby([data.index, 'phase'], observed = True).agg(**data_summary).reset_index(1)

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control
        if facet_col: grouped_data = self.facet_merge(grouped_data, self.meta,facet_col, facet_arg, facet_labels)

        return grouped_data, palette_dict, facet_labels
    
    def _internal_plot_anticipation_score(self, variable, facet_col, facet_arg, facet_labels, day_length, lights_off, t_column):
        """ An internal method to applt the preprocessing to a dataset before calling the anticopation_score method """
        if variable not in self.columns.tolist():
            raise KeyError(f'The column you gave {variable}, is not in the data')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        # takes subset of data if requested
        if facet_col and facet_arg:
            data = self.xmv(facet_col, facet_arg)
        else:
            data = self.copy(deep=True)

        grouped_data = data.anticipation_score(variable, day_length, lights_off, t_column)

        palette = self._get_colours(facet_labels)
        palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control

        if facet_col:
            grouped_data = self.facet_merge(grouped_data, self.meta, facet_col, facet_arg, facet_labels)

        return grouped_data, palette_dict, facet_labels

    def _internal_plot_decoder(self, hmm, variable, labels, colours, facet_col, facet_arg, facet_labels,
        t_bin, func, t_column, rm=False):
        """ contains the first part of the internal plotters for HMM quant plots """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, t_bin)

        if rm:
            # remove the first and last bout to reduce errors and also copy the data
            data = self.remove_first_last_bout(variable=variable)
        else:
            data = self.copy(deep=True)

        # takes subset of data if requested
        if facet_col and facet_arg:
            # takes subselection of df that contains the specified facet columns
            data = self.xmv(facet_col, facet_arg)

        def hmm_list_facet(data, meta, facet_label, ind):
            d = data.copy(deep=True)
            m = meta.copy(deep=True)

            d.id = d.id + f'_{ind}'
            m.index = m.index + f'_{ind}'
            m['HMM'] = facet_label
            return self.__class__(d, m, check=True)

        if facet_col is None:  # decode the whole dataset
            if isinstance(hmm, list):
                decoded_data = concat(*[hmm_list_facet(self._hmm_decode(data, h, b, variable, func, t_column, return_type='table'), data.meta, f, c+1) 
                                for c, (h, b, f) in enumerate(zip(h_list, b_list, facet_labels))])  
                facet_arg = facet_labels
                facet_col = 'HMM'
            else:
                decoded_data = self.__class__(self._hmm_decode(data, hmm, t_bin, variable, func, t_column, return_type='table'), data.meta, check=True)  
        else:
            if isinstance(hmm, list): 
                decoded_data = concat(*[self.__class__(self._hmm_decode(data.xmv(facet_col, arg), h, b, variable, func, t_column, return_type='table'), 
                                            data.meta, check=True) for arg, h, b in zip(facet_arg, h_list, b_list)])
            else: # if only 1 hmm but is faceted, decode as whole for efficiency
                decoded_data = self.__class__(self._hmm_decode(data, hmm, t_bin, variable, func, t_column, return_type='table'), data.meta, check=True)


        return decoded_data, labels, colours, facet_col, facet_arg, facet_labels

    def _internal_plot_hmm_quantify(self, hmm, variable, labels, colours, facet_col, facet_arg, facet_labels, 
        t_bin, func, t_column):
        """ internal method to calculate the average amount of each state for use in plot_hmm_quantify, plotly and seaborn """

        decoded_data, labels, colours, facet_col, facet_arg, facet_labels = self._internal_plot_decoder(hmm, variable, labels, colours, 
                                                                                                        facet_col, facet_arg, facet_labels, 
                                                                                                            t_bin, func, t_column)

        # Count each state and find its fraction
        grouped_data = decoded_data.groupby([decoded_data.index, 'state'], sort=False).agg({'bin' : 'count'})
        grouped_data = grouped_data.join(decoded_data.groupby('id', sort=False).agg({'previous_state':'count'}))
        grouped_data['Fraction of time in each State'] = grouped_data['bin'] / grouped_data['previous_state']
        grouped_data.reset_index(level=1, inplace = True)

        if facet_col:
            palette = self._get_colours(facet_labels)
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control
            grouped_data = self.facet_merge(grouped_data, decoded_data.meta, facet_col, facet_arg, facet_labels, hmm_labels = labels) 
        else:
            palette = colours
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(labels)} # change to grey if control            
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)

        return grouped_data, labels, colours, facet_col, facet_labels, palette_dict

    def _internal_plot_hmm_quantify_length(self, hmm, variable, labels, colours, facet_col, facet_arg, facet_labels, 
        t_bin, func, t_column):
        """ internal method to calculate the average length of each state for use in plot_hmm_quantify_length, plotly and seaborn """

        decoded_data, labels, colours, facet_col, facet_arg, facet_labels = self._internal_plot_decoder(hmm, variable, labels, colours, 
                                                                                             facet_col, facet_arg, facet_labels, 
                                                                                                    t_bin, func, t_column)

        # get each specimens states time series to find lengths
        states = decoded_data.groupby(decoded_data.index, sort=False)['state'].apply(list)
        df_lengths = []
        for l, id in zip(states, states.index):
            length = hmm_mean_length(l, delta_t = t_bin) 
            length['id'] = [id] * len(length)
            df_lengths.append(length)

        grouped_data = pd.concat(df_lengths)
        grouped_data.rename(columns={'mean_length' : 'Length of state bout (mins)'}, inplace=True)
        grouped_data.set_index('id', inplace=True)

        if facet_col:
            palette = self._get_colours(facet_labels)
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control
            grouped_data = self.facet_merge(grouped_data, decoded_data.meta, facet_col, facet_arg, facet_labels, hmm_labels = labels) 
        else:
            palette = colours
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(labels)} # change to grey if control            
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)

        return grouped_data, labels, colours, facet_col, facet_labels, palette_dict

    def _internal_plot_hmm_quantify_length_min_max(self, hmm, variable, labels, colours, facet_col, facet_arg, facet_labels, 
        t_bin, func, t_column):
        """ internal method to calculate the average length of each state for use in plot_hmm_quantify_length, plotly and seaborn """

        decoded_data, labels, colours, facet_col, facet_arg, facet_labels = self._internal_plot_decoder(hmm, variable, labels, colours, 
                                                                                                        facet_col, facet_arg, facet_labels, 
                                                                                                            t_bin, func, t_column, rm = True)

        # get each specimens states time series to find lengths
        states = decoded_data.groupby(decoded_data.index, sort=False)['state'].apply(list)
        df_lengths = []
        for l, id in zip(states, states.index):
            length = hmm_mean_length(l, delta_t = t_bin, raw=True) 
            length['id'] = [id] * len(length)
            df_lengths.append(length)

        grouped_data = pd.concat(df_lengths)
        grouped_data.rename(columns={'length_adjusted' : 'Length of state bout (mins)'}, inplace=True)
        grouped_data.set_index('id', inplace=True)

        if facet_col:
            palette = self._get_colours(facet_labels)
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control
            grouped_data = self.facet_merge(grouped_data, decoded_data.meta, facet_col, facet_arg, facet_labels, hmm_labels = labels) 
        else:
            palette = colours
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(labels)} # change to grey if control            
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)

        return grouped_data, labels, colours, facet_col, facet_labels, palette_dict

    def _internal_plot_hmm_quantify_transition(self, hmm, variable, labels, colours, facet_col, facet_arg, facet_labels, 
        t_bin, func, t_column):
        """ An internal method to find the % of transtions into a state occur per state per individual """

        decoded_data, labels, colours, facet_col, facet_arg, facet_labels = self._internal_plot_decoder(hmm, variable, labels, colours, 
                                                                                                        facet_col, facet_arg, facet_labels, 
                                                                                                            t_bin, func, t_column, rm = True)

        # get each specimens states time series to find lengths
        states = decoded_data.groupby(decoded_data.index, sort=False)['state'].apply(list)
        df_list = []
        for l, id in zip(states, states.index):
            length = hmm_pct_transition(l, total_states=list(range(len(labels)))) 
            length['id'] = [id] * len(length)
            df_list.append(length)

        grouped_data = pd.concat(df_list)
        grouped_data = grouped_data.set_index('id').stack().reset_index().set_index('id')
        grouped_data.rename(columns={'level_1' : 'state', 0 : 'Fraction of transitions into each state'}, inplace=True)

        if facet_col:
            palette = self._get_colours(facet_labels)
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(facet_labels)} # change to grey if control
            grouped_data = self.facet_merge(grouped_data, decoded_data.meta, facet_col, facet_arg, facet_labels, hmm_labels = labels) 
        else:
            palette = colours
            palette_dict = {name : self._check_grey(name, palette[c])[1] for c, name in enumerate(labels)} # change to grey if control            
            hmm_dict = {k : v for k, v in zip(range(len(labels)), labels)}
            grouped_data['state'] = grouped_data['state'].map(hmm_dict)

        return grouped_data, labels, colours, facet_col, facet_labels, palette_dict
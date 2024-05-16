import numpy as np 
import pandas as pd
import re

import seaborn as sns
from plotly.express.colors import qualitative
from colour import Color
from math import sqrt, floor, ceil

from ethoscopy.behavpy_core import behavpy_core

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

    def _get_colours(self, plot_list):
        """ returns a colour palette from plotly for plotly """

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

    def save_figure(self, fig, path, width = None, height = None):
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
        """Get the nearest higher square number"""
        square = np.sqrt(length) 
        closest = [floor(square)**2, ceil(square)**2]
        return int(sqrt(closest[1]))

    @staticmethod
    def _check_grey(name, col):
        if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
            col = 'grey'
        return name, col

    # GENERAL PLOT HELPERS

    def _generate_overtime_plot(self, data, name, col, var, avg_win, wrap, day_len, light_off, t_col, canvas):

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

        if canvas == 'seaborn':
            return gb_df, t_min, t_max, col, None
        elif canvas == 'plotly':
            upper, trace, lower = data._plot_line(df = gb_df, x_col = t_col, name = name, marker_col = col)
            return upper, trace, lower, t_min, t_max
        else:
            KeyError(f'Wrong plot type in back end: {plot_type}')

    def heatmap_dataset(self, variable, t_column):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals
        
            Args:
                variable (string): The name for the column containing the variable of interest,
                t_column (str): The name of the time column in the DataFrame.
        
        returns 
            gbm, time_list, id
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
            """merge the individual fly groups time with the time map, filling in missing points with NaN values"""

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

    def anticipation_score(self, data, mov_variable, day_length, lights_off):
        
        def _ap_score(total, small):
            try:
                return (small / total) * 100
            except ZeroDivisionError:
                return 0

        ant_df = pd.DataFrame()

        for phase in ['Lights Off', 'Lights On']:

            if phase == 'Lights Off':
                start = [lights_off - 6, lights_off - 3]
                end = lights_off
            elif phase == 'Lights On':
                start = [day_length - 6, day_length - 3]
                end = day_length

            d = data.t_filter(start_time = start[0], end_time = end)
            total = d.analyse_column(column = mov_variable, function = 'sum')
            
            d = data.t_filter(start_time = start[1], end_time = end)
            small = d.analyse_column(column = mov_variable, function = 'sum')
            d = total.join(small, rsuffix = '_small')
            d = d.dropna()
            d = pd.DataFrame(d[[f'{mov_variable}_sum', f'{mov_variable}_sum_small']].apply(lambda x: _ap_score(*x), axis = 1), columns = ['anticipation_score']).reset_index()
            d['phase'] = phase
            ant_df = pd.concat([ant_df, d])
        
        return ant_df

    def facet_merge(self, data, facet_col, facet_arg, facet_labels):
        # merge the facet_col column and replace with the labels
        data = data.join(self.meta[[facet_col]])
        data[facet_col] = data[facet_col].astype('category')
        map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
        data[facet_col] = data[facet_col].map(map_dict)
        return data
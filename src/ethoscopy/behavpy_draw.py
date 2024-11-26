import numpy as np 
import pandas as pd
import re

import seaborn as sns
from plotly.express.colors import qualitative
from colour import Color
from math import sqrt, floor, ceil
from scipy.stats import zscore

#fig to img
import io
import PIL

from ethoscopy.behavpy_core import behavpy_core
from ethoscopy.misc.bootstrap_CI import bootstrap

class behavpy_draw(behavpy_core):
    """
    Default drawing class containing some general methods that can be used by all children drawing classes
    """

    _hmm_colours = ['darkblue', 'dodgerblue', 'red', 'darkred']
    _hmm_labels = ['Deep sleep', 'Light sleep', 'Quiet awake', 'Active awake']

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

    # Internal methods for checking data/arguments before plotting
    def _check_hmm_shape(self, hm, lab, col):
        """
        Check the colours and labels passed to a plotting method are of equal length. If None then it will be populated with the defaults.
        """
        if isinstance(hm, list):
            hm = hm[0]

        if hm.transmat_.shape[0] == 4:
            if lab == None and col == None:
                _labels = self._hmm_labels
                _colours = self._hmm_colours
            elif lab == None and col != None:
                _labels = self._hmm_labels
                _colours = col
            elif lab != None and col == None:
                _labels = lab
                _colours = self._hmm_colours

        elif hm.transmat_.shape[0] != 4:
            if lab == None and col == None:
                # give generic names and populate with colours from the given palette 
                _labels = [f'state_{i}' for i in range(0, hm.transmat_.shape[0])]
                _colours = self.get_colours(hm.transmat_)
            elif lab != None and col == None:
                _colours = self.get_colours(hm.transmat_)
                _labels = lab
            elif lab == None and col != None:
                _colours = col
                _labels = [f'state_{i}' for i in range(0, hm.transmat_.shape[0])]
            else:
                if len(col) != len(lab):
                    raise RuntimeError('You have more or less states than colours, please rectify so the lists are equal in length')
                _labels = lab
                _colours = col
        else:
            _labels = lab
            _colours = col

        if len(_labels) != len(_colours):
            raise RuntimeError('Internal check fail: You have more or less states than colours, please rectify so they are equal in length')
        
        return _labels, _colours

    def _check_lists_hmm(self, f_col, f_arg, f_lab, h, b):
        """
        Check if the facet arguments match the labels or populate from the column if not.
        Check if there is more than one HMM object for HMM comparison. Populate hmm and bin lists accordingly.
        """
        if isinstance(h, list):
            assert isinstance(b, list), "If providing a list of HMMs, also provide a list of ints to bin the time by (t_bin)"
            if len(h) != len(f_arg) or len(b) != len(f_arg):
                raise RuntimeError('There are not enough hmm models or bin intergers for the different groups or vice versa')
            else:
                h_list = h
                b_list = b

        if f_col is not None:
            if f_arg is None:
                f_arg = list(set(self.meta[f_col].tolist()))
                if f_lab is None:
                    string_args = []
                    for i in f_arg:
                        if i not in self.meta[f_col].tolist():
                            raise KeyError(f'Argument "{i}" is not in the meta column {f_col}')
                        string_args.append(str(i))
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    print("The facet labels don't match the length of the variables in the column. Using column variables instead")
                    f_lab = f_arg
            else:
                if f_lab is None:
                    string_args = []
                    for i in f_arg:
                        string_args.append(str(i))
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    print("The facet labels don't match the entered facet arguments in length. Using column variables instead")
                    f_lab = f_arg
        else:
            f_arg = [None]
            if f_lab is None:
                f_lab = ['']

        if isinstance(h, list) is False:
            h_list = [h]
            b_list = [b]
            if len(h_list) != len(f_arg):
                h_list = [h_list[0]] * len(f_arg)
            if len(b_list) != len(f_arg):
                b_list = [b_list[0]] * len(f_arg)

        return f_arg, f_lab, h_list, b_list

    @staticmethod
    def _zscore_bootstrap(array, z_score = True, second_array = None, min_max = False):
        """ Calculate the z score of a given array, remove any values +- 3 SD and then perform bootstrapping on the remaining
        returns the mean and then several lists with the confidence intervals and z-scored values
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
                boot_array = bootstrap(zlist)
                q3 = boot_array[1]
                q1 = boot_array[0]

        except ZeroDivisionError:
            mean = median = q3 = q1 = 0
            zlist = array

        if min_max == True:
            q3 = np.max(array)
            q1 = np.min(array)
        
        if median < q1 or median > q3:
            median = mean

        if second_array is not None:
            return mean, median, q3, q1, zlist, second_array
        else:
            return mean, median, q3, q1, zlist

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
    # Function to convert figure to image
    def _fig2img(fig, format='png'):
        buf = io.BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = PIL.Image.open(buf)
        return img

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
    def _check_grey(name, col, response = False):
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

    def facet_merge(self, data, facet_col, facet_arg, facet_labels, hmm_labels = None):
        # merge the facet_col column and replace with the labels
        data = data.join(self.meta[[facet_col]])
        data[facet_col] = data[facet_col].astype('category')
        map_dict = {k : v for k, v in zip(facet_arg, facet_labels)}
        data[facet_col] = data[facet_col].map(map_dict)
        if hmm_labels is not None:
            hmm_dict = {k : v for k, v in zip(range(len(hmm_labels)), hmm_labels)}
            data['state'] = data['state'].map(hmm_dict)
        return data

    def _generate_overtime_plot(self, data, name, col, var, avg_win, wrap, day_len, light_off, t_col):

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

    def _hmm_response(self, mov_df, hmm, variable, response_col, labels, colours, facet_col, facet_arg, t_bin, facet_labels, func, t_column):

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
                mdata = concat(*[self.__class__(self._hmm_decode(mdata.xmv(facet_col, arg), h, b, variable, func, t_column, return_type='table'), mdata.meta, check=True) for arg, h, b in zip(facet_arg, h_list, b_list)])

        # merge the two df's and check if the interaction happened in the right time point
        def alter_merge(response, mov, tb):
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
            data = concat(*[self.__class__(alter_merge(data.xmv(facet_col, arg), mdata.xmv(facet_col, arg), b), data.meta, check=True) for arg, b in zip(facet_arg, b_list)])

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
            grouped_data = self.facet_merge(grouped_data, facet_col, facet_arg, facet_labels, hmm_labels = labels)
            grouped_data[facet_col] = grouped_data[facet_col].astype('str')
            grouped_data[facet_col] = grouped_data[facet_col] + " " + grouped_data['has_interacted']

        return grouped_data, palette_dict, h_order
    
    def _bouts_response(self, mov_df, hmm, variable, response_col, labels, colours, x_limit, t_bin, func, t_col):

        data_summary = {
            "mean" : (response_col, 'mean'),
            "count" : (response_col, 'count'),
            "ci" : (response_col, bootstrap),
            }
        data = self.copy(deep=True)
        mdata = mov_df.copy(deep=True)

        if hmm is not False:
            # copy and decode the dataset
            mdata = self.__class__(self._hmm_decode(mdata, hmm, t_bin, variable, func, t_col, return_type='table'), mdata.meta, check=True)
            var, newT, m_var_1, m_var_2 = 'state', 'bin', 'moving', 'previous_moving'
        else:
            mdata = mdata.bin_time(variable, t_bin, function = func, t_column = t_col)
            var, newT, m_var_1, m_var_2 = f'{variable}_{func}', f'{t_col}_bin', 'activity_count', 'previous_activity_count'

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
        grouped_data['state'] = grouped_data['state'].map(hmm_dict)
        grouped_data['label_col'] =  grouped_data['state'] + " " + grouped_data['has_interacted']
        # create the order of plotting and double the colours to assign grey to false stimuli
        h_order = [f'{lab} {ty}' for lab in labels for ty in ["Spon. Mov.", "True Stimulus"]]
        palette = [x for xs in [[col, col] for col in colours] for x in xs]
        palette_dict = {name : self._check_grey(name, palette[c], response = True)[1] for c, name in enumerate(h_order)} # change to grey if control

        return grouped_data, palette_dict, h_order

    def _internal_bout_activity(self, mov_df, activity, variable, response_col, facet_col, facet_arg, facet_labels, x_limit, t_bin, t_column):
        """ The beginning code for plot_response_over_activity for both plotly and seaborn """

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
        palette = [x for xs in [[col, col] for col in self._get_colours(facet_labels)] for x in xs]
        h_order = [f'{lab} {ty}' for lab in facet_labels for ty in ["Spon. Mov.", "True Stimulus"]]
        palette_dict = {name : self._check_grey(name, palette[c], response = True)[1] for c, name in enumerate(h_order)} # change to grey if control

        # If not both filter the dataset
        if isinstance(activity_choice[activity], int):
            grouped_data = grouped_data[grouped_data['previous_moving'] == activity_choice[activity]]
        
        return grouped_data, h_order, palette_dict, activity_choice[activity]
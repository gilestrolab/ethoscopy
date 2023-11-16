import pandas as pd
import numpy as np 
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
import pickle

from tabulate import tabulate
from math import floor, ceil
from sys import exit
from colour import Color
from functools import partial

from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from astropy.timeseries import LombScargle

from ethoscopy.behavpy_class import behavpy
from ethoscopy.misc.periodogram_functions import chi_squared, lomb_scargle, fourier, welch, wavelet

class behavpy_periodogram(behavpy):
    """
    The behavpy_circadian class is a subclassed version of behavpy, itself a subclass of pandas. See behavpy doc strings for more information on behavpy.

    Behavpy_circadian is for use with behavioural datasets to perform circadian analysis

    """

    def __init__(self, data, meta,
                colour = 'Safe', long_colour = 'Dark24',
                check = False, 
                index= None, columns=None, dtype=None, copy=True):

        super(behavpy, self).__init__(data=data,
                                        index=index,
                                        columns=columns,
                                        dtype=dtype,
                                        copy=copy)

        self.meta = meta 

        if check is True:
            self._check_conform(self)
        self.attrs = {'short_col' : colour, 'long_col' : long_colour}

    def _validate(self):
        """ Validator to check further periodogram methods if the data is produced from the periodogram method """
        if  any([i not in self.columns.tolist() for i in ['period', 'power']]):
            raise AttributeError('This method is for the computed periodogram data only, please run the periodogram method on your data first')
        

    def _check_periodogram_input(self, v, per, per_range, t_col, wavelet_type = False):
        """ Method to check the input to periodogram methods"""

        periodogram_list = ['chi_squared', 'lomb_scargle', 'fourier', 'welch']

        if v not in self.columns.tolist():
            raise AttributeError(f"Variable column {v} is not a column title in your given dataset")

        if t_col not in self.columns.tolist():
            raise AttributeError(f"Time column {t_col} is not a column title in your given dataset")

        if wavelet_type is not False:
            fun = eval(per)
            return fun

        if per in periodogram_list:
            fun = eval(per)
        else:
            raise AttributeError(f"Unknown periodogram type, please use one of {*periodogram_list,}")

        if isinstance(per_range, list) is False and isinstance(per_range, np.array) is False:
            raise TypeError(f"per_range should be a list or nummpy array, please change")

        if isinstance(per_range, list) or isinstance(per_range, np.array):

            if len(per_range) != 2:
                warnings.warn("The period range can only be a tuple/array of length 2, please amend")
                exit()

            if per_range[0] < 0 or per_range[1] < 0:
                raise ValueError(f"One or both of the values of the period_range given are negative, please amend")

        return fun

    def periodogram(self, mov_variable, periodogram, period_range = [10, 32], sampling_rate = 15, alpha = 0.01, t_col = 't'):
        """ A method to apply a periodogram function to given behavioural data. Call this method first to create an analysed dataset that can access 
        the other methods of this class 
        params:
        """

        fun = self._check_periodogram_input(mov_variable, periodogram, period_range, t_col)

        sampling_rate = 1 / (sampling_rate * 60)

        data = self.copy(deep = True)
        sampled_data = data.interpolate(variable = mov_variable, step_size = 1 / sampling_rate)
        sampled_data = sampled_data.reset_index()
        return  type(self)(sampled_data.groupby('id', group_keys = False)[[t_col, mov_variable]].apply(partial(fun, var = mov_variable, t_col = t_col, period_range = period_range, freq = sampling_rate, alpha = alpha)), data.meta, colour = self.attrs['short_col'], long_colour = self.attrs['long_col'], check = True)

    @staticmethod
    def wavelet_types():
        wave_types = ['morl', 'cmor', 'mexh', 'shan', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8']
        return wave_types

    def wavelet(self, mov_variable, sampling_rate = 15, scale = 156, wavelet_type = 'morl', t_col = 't', title = '', grids = False):
        """ A method to apply a wavelet function using the python package pywt. Head to https://pywavelets.readthedocs.io/en/latest/ for information about the pacakage and the other wavelet types
        This method will produce a single wavelet transformation plot, averging the the data from across all specimens. It is therefore recommended you filter your dataset accordingly before applying 
        this method, i.e. by different experimental groups or a singl specimen.
        params:
        @variable:  """
        
        fun = self._check_periodogram_input(v = mov_variable, per = 'wavelet', per_range = None, t_col = t_col, wavelet_type = wavelet_type)
        sampling_rate = 1 / (sampling_rate * 60)

        data = self.copy(deep = True)
        sampled_data = data.interpolate(variable = mov_variable, step_size = 1 / sampling_rate)
        avg_data = sampled_data.groupby(t_col).agg(**{
                        mov_variable : (mov_variable, 'mean')
        })
        avg_data = avg_data.reset_index()

        fig = go.Figure()
        yticks = [1,2,4,6,12,24,36]
        self._plot_ylayout(fig, yrange = np.log2([2,38]), tickvals = np.log2(yticks), ticktext = yticks, title = title, ylabel = 'Period Frequency (Hours)', grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = 12, xlabel = 'ZT (Hours)')

        t, per, pow = fun(avg_data, t_col = t_col, var = mov_variable, scale = scale, wavelet_type = wavelet_type)

        trace = go.Contour(
                z=pow,
                x=t, 
                y=per,
                contours=dict(
                    start= -3,
                    end= 3,
                    size= 1,
                    # type = 'constraint'
                ),
                colorscale='Jet',
                colorbar=dict(nticks=7, ticks='outside',
                        ticklen=5, tickwidth=1,
                        showticklabels=True,
                        tickangle=0, tickfont_size=12)
            )
        fig.add_trace(trace)
        # config = {'staticPlot': True} 
        
        return fig

    @staticmethod
    def _wrapped_find_peaks(data, num, height = None):

        if height is True:
            peak_ind, _ = find_peaks(x = data['power'].to_numpy(), height = data['sig_threshold'].to_numpy())
        else:
            peak_ind, _ = find_peaks(x = data['power'].to_numpy())

        peaks = data['period'].to_numpy()[peak_ind]

        peak_power = data['power'].to_numpy()[peak_ind]
        order = peak_power.argsort()[::-1]
        ranks = order.argsort() + 1

        rank_dict = {k : int(v) for k,v in zip(peaks, ranks)}
        data['peak'] = data['period'].map(rank_dict).fillna(False)
        data['peak'] =  np.where(data['peak'] > num, False, data['peak'])

        return data
    
    def find_peaks(self, num_peaks):
        """ Find the peaks in a computed periodogram, a wrapper for the scipy find_peaks function"""
        self._validate()
        data = self.copy(deep=True)
        data = data.reset_index()
        if 'sig_threshold' in data.columns.tolist():
            return  type(self)(data.groupby('id', group_keys = False).apply(partial(self._wrapped_find_peaks, num = num_peaks, height = True)), data.meta, colour = self.attrs['short_col'], long_colour = self.attrs['long_col'], check = True)
        else:
            return  type(self)(data.groupby('id', group_keys = False).apply(partial(self._wrapped_find_peaks, num = num_peaks)), data.meta, colour = self.attrs['short_col'], long_colour = self.attrs['long_col'], check = True)

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

        root =  self._get_subplots(len(data.meta))

        # make a square subplot domain
        fig = make_subplots(rows=root, cols=root, shared_xaxes = False, subplot_titles = title_list)
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

            if 'peak' in d.columns.tolist():
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
        """ Plots the averaged periodograms of different experimental groups """
        self._validate()

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

            upper, trace, lower, _, _, _ = self._generate_overtime_plot(data = data, name = name, col = col, var = 'power', avg_win = False, wrap = False, day_len = False, light_off = False, t_col = 'period')
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

        return fig
    
    def quantify_periodogram(self, facet_col = None, facet_arg = None, facet_labels = None, title = '', grids = False):
        """ Plots a box plot of means and 95% confidence intervals of the highest ranked peak in a series of periodograms"""

        self._validate()

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        col_list = self._get_colours(d_list)
        variable = 'period'

        fig = go.Figure() 
        self._plot_ylayout(fig, yrange = [min(self['period']), max(self['period'])], t0 = False, dtick = False, ylabel = 'Period (Hours)', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = '')

        stats_dict = {}

        for data, name, col in zip(d_list, facet_labels, col_list):
            
            if len(data) == 0:
                print(f'Group {name} has no values and cannot be plotted')
                continue

            if 'peak' not in data.columns.tolist():
                data = data.find_peaks(num_peaks = 1)
            data = data[data['peak'] == 1]

            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'grey'

            median, q3, q1, zlist, z_second = self._zscore_bootstrap(data[f'{variable}'].to_numpy(), second_array = data['power'].to_numpy())
            stats_dict[name] = zlist

            fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
            x = [name], colour =  col, showlegend = False, name = name, xaxis = 'x'))

            fig.add_trace(self._plot_boxpoints(y = zlist, x = len(zlist) * [name], colour = col, 
            showlegend = False, name = name, xaxis = 'x'))

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df
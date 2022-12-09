import pandas as pd
import numpy as np 
import warnings
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

    def _validate(self):
        """ Validator to check further periodogram methods if the data is produced from the periodogram method """
        if  any([i not in self.columns.tolist() for i in ['period', 'power']]):
            raise AttributeError('This method is for the computed periodogram data only, please run the periodogram method on your data first')
        

    def _check_periodogram_input(self, v, per, per_range, t_col, wavelet = False):
        """ Method to check the input to periodogram methods"""

        periodogram_list = ['chi_squared', 'lomb_scargle', 'fourier', 'welch', 'wavelet']

        if per in periodogram_list:
            fun = eval(per)
        else:
            raise AttributeError(f"Unknown periodogram type, please use one of {*periodogram_list,}")

        if v not in self.columns.tolist():
            raise AttributeError(f"Variable column {v} is not a column title in your given dataset")

        if t_col not in self.columns.tolist():
            raise AttributeError(f"Time column {t_col} is not a column title in your given dataset")

        if wavelet is True:
            return fun

        if isinstance(per_range, list) is False and isinstance(per_range, np.array) is False:
            raise TypeError(f"per_range should be a list or nummpy array, please change")

        if isinstance(per_range, list) or isinstance(per_range, np.array):

            if len(per_range) != 2:
                warnings.warn("The period range can only be a tuple/array of length 2, please amend")
                exit()

            if per_range[0] < 0 or per_range[1] < 0:
                raise ValueError(f"One or both of the values of the period_range given are negative, please amend")

        return fun

    def periodogram(self, variable, periodogram, period_range = [10, 36], sampling_rate = 15, alpha = 0.01, t_col = 't'):
        """ A method to apply a periodogram function to given behavioural data. Call this method first to create an analysed dataset that can access 
        the other methods of this class 
        params:
        """

        fun = self._check_periodogram_input(variable, periodogram, period_range, t_col)

        sampling_rate = 1 / (sampling_rate * 60)

        data = self.copy(deep = True)
        sampled_data = data.interpolate(variable = variable, step_size = 1 / sampling_rate)
        sampled_data = sampled_data.reset_index()
        return  behavpy_periodogram(sampled_data.groupby('id', group_keys = False)[[t_col, variable]].apply(partial(fun, var = variable, t_col = t_col, period_range = period_range, freq = sampling_rate, alpha = alpha)), data.meta, check = True)

    def wavelet(self, variable, id, facet_col = None, facet_arg = None, facet_labels = None, sampling_rate = 15, scale = 156, wavelet = 'morl', t_col = 't', plotter = 'seaborn'):
        """ A method to apply a wavelet function using the python package pywt. Head to https://pywavelets.readthedocs.io/en/latest/ for information about the pacakage and the other wavelet types
        The method will return a figure using either seaborn or plotly. Due to how many data points there are to plot its recommended to use seaborn instead of plotly.
        params:
        @variable:  """

        if facet_col is None:
            if id not in self.meta.index.tolist():
                raise AttributeError(f'{id} is not an id in the metadata')

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        if facet_col != None:
            root = self._get_subplots(len(facet_arg))
            title_list = facet_labels
        else:
            facet_arg = self.meta.index.tolist()
            root =  self._get_subplots(len(facet_arg))
            if id is not None:
                title_list = self.meta[id].tolist()
            else:
                title_list = facet_arg


        fun = self._check_periodogram_input(v = variable, per = 'wavelet', per_range = None, t_col = t_col, wavelet = True)
        sampling_rate = 1 / (sampling_rate * 60)

        data = self.copy(deep = True)
        sampled_data = data.interpolate(variable = variable, step_size = 1 / sampling_rate)
        sampled_data = sampled_data.reset_index()
        t, period, power = sampled_data.groupby('id', group_keys = False)[[t_col, variable]].apply(partial(fun, t_co = t_col, var = variable, scale = scale, wavelet = wavelet))

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
        """ Find the peaks in a computed periodogram"""
        self._validate()
        data = self.copy(deep=True)
        data = data.reset_index()
        if 'sig_threshold' in data.columns.tolist():
            return  behavpy_periodogram(data.groupby('id', group_keys = False).apply(partial(self._wrapped_find_peaks, num = num_peaks, height = True)), data.meta, check = True)
        else:
            return  behavpy_periodogram(data.groupby('id', group_keys = False).apply(partial(self._wrapped_find_peaks, num = num_peaks)), data.meta, check = True)

    def plot_periodogram_tile(self, labels = None, find_peaks = False, title = '', grids = False, save = False):
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
                    marker_size = 15
                ), row = row, col = col)

        tick_6 = np.arange(0,60,6)

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
            tickfont = dict(
                size = 14
            ),
            showgrid = False
        )

        fig.update_yaxes(
            zeroline = False,
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            range = [int(min(data['power'])), int(max(data['power'])) + 20],
            ticks = 'outside',
            showgrid = grids,
        )
        
        fig.update_annotations(font_size=8)
        fig['layout']['title'] = title
        fig['layout']['plot_bgcolor'] = 'white'
        
        fig.add_annotation(
                    font = {'size': 18, 'color' : 'black'},
                    showarrow = False,
                    text = 'Period Time (Hours)',
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
        if isinstance(save, str):
            if save.endswith('.html'):
                fig.write_html(save)
            else:
                fig.write_image(save, width=1500, height=650)
            print(f'Saved to {save}')
            fig.show()
        else:
            fig.show()
        
        return fig

    
    def quantify_periodogram(self):
        return

    

if __name__ == '__main__':
    df_orig = pd.read_csv(r'C:\Users\lab\Documents\Projects\dummy_ethoscopy\periodogram_csv.csv')
    df_orig = df_orig.set_index('id')
    df_meta = pd.read_pickle(r'C:\Users\lab\Documents\Projects\dummy_ethoscopy\sd_periodogram_meta.pkl')

    df = behavpy_periodogram(df_orig, df_meta, check = True)

    rand_flies = np.random.permutation(list(set(df.meta.index)))[:9]
    rand_flies = list(rand_flies)

    tdf = df.xmv('id', rand_flies)
    # tdf = df.xmv('id', '2016-04-04_17-38-06_019aee|04')
    print('start periodogram')
    perio = tdf.periodogram(variable = 'moving', periodogram = 'chi_squared')
    print('finsihed periodogram')
    perio = perio.find_peaks(num_peaks = 2)
    # print(perio[perio['peak'] != False])
    f = perio.plot_periodogram_tile()
import pandas as pd

import numpy as np
import plotly.graph_objs as go 
from plotly.express.colors import qualitative

import seaborn as sns
import matplotlib.pyplot as plt

from ethoscopy.behavpy_class import behavpy
from ethoscopy.misc.circadian_bars import circadian_bars


class behavpy_base2(behavpy):

    @staticmethod  
    def _plot_line(df, column, name, marker_col, t_col = 't'):

        def pop_std(array):
            return np.std(array, ddof = 0)

        gb_df = df.groupby(t_col).agg(**{
                    'mean' : (column, 'mean'), 
                    'SD' : (column, pop_std),
                    'count' : (column, 'count')
                })

        max_var = max(gb_df['mean'])

        gb_df['SE'] = (1.96*gb_df['SD']) / np.sqrt(gb_df['count'])
        gb_df['y_max'] = gb_df['mean'] + gb_df['SE']
        gb_df['y_min'] = gb_df['mean'] - gb_df['SE']

        upper_bound = go.Scatter(
        showlegend = False,
        legendgroup = name,
        x = gb_df.index.values,
        y = gb_df['y_max'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0,
                shape = 'spline'
                ),
        )
        trace = go.Scatter(
            legendgroup = name,
            x = gb_df.index.values,
            y = gb_df['mean'],
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
            x = gb_df.index.values,
            y = gb_df['y_min'],
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
    def _get_colours(plot_list):
        if len(plot_list) <= 11:
            return qualitative.Safe
        elif len(plot_list) < 24:
            return qualitative.Dark24
        else:
            warnings.warn('Too many sub groups to plot with the current colour palette')
            exit()
            
    # set meta as permenant attribute
    _metadata = ['meta']
        
    _colours_small = qualitative.Safe
    _colours_large = qualitative.Dark24

    def _heatmap(self, variable = 'moving'):
        '''
        '''
        heatmap_df = self.copy(deep = True)
        # change movement values from boolean to intergers and bin to 30 mins finding the mean
        if variable == 'moving':
            heatmap_df[variable] = np.where(heatmap_df[variable] == True, 1, 0)

        heatmap_df = heatmap_df.bin_time(column = variable, bin_secs = 1800)
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

        return time_list, heatmap_df.groupby('id', group_keys = False).apply(align_data)


    def _plot_overtime(self, variable, wrapped = False, facet_col = None, facet_arg = None, facet_labels = None, avg_window = 30, day_length = 24, lights_off = 12, title = '', grids = False, save = False):

        facet_arg, facet_labels = self._check_lists(facet_col, facet_arg, facet_labels)

        d_list = []
        if facet_col is not None:
            for arg in facet_arg:
                d_list.append(self.xmv(facet_col, arg))
        else:
            d_list = [self.copy(deep = True)]
            facet_labels = ['']

        if len(d_list) < 11:
            col_list = self._colours_small
        elif len(d_list) < 24:
            col_list = self._colours_large
        else:
            warnings.warn('Too many sub groups to plot with the current colour palette')
            exit()

        max_var = []
        y_range, dtick = self._check_boolean(list(self[variable].dropna()))
        if y_range is False:
            max_var.append(1)
        
        fig = go.Figure() 
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = dtick, ylabel = variable, title = title, grid = grids)
        self._plot_xlayout(fig, xrange = False, t0 = 0, dtick = day_length/4, xlabel = 'ZT (Hours)')

        min_t = []
        max_t = []

        for data, name, col in zip(d_list, facet_labels, col_list):

            if 'baseline' in name.lower() or 'control' in name.lower() or 'ctrl' in name.lower():
                col = 'grey'

            rolling_col = data.groupby(data.index, sort = False)[variable].rolling(avg_window).mean().reset_index(level = 0, drop = True)
            data['rolling'] = rolling_col.to_numpy()
            data = data.dropna(subset = ['rolling'])

            if wrapped is True:
                data['t'] = data['t'].map(lambda t: t % (60*60*day_length))
            data['t'] = data['t'].map(lambda t: t / (60*60))

            t_min = int(lights_off * np.floor(data.t.min() / lights_off))
            min_t.append(t_min)
            t_max = int(12 * np.ceil(data.t.max() / 12)) 
            max_t.append(t_max)

            upper, trace, lower, maxV = self._plot_line(df = data, column = 'rolling', name = name, marker_col = col)
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

            max_var.append(maxV)

        # Light-Dark annotaion bars
        bar_shapes = circadian_bars(t_min, t_max, max_y = max(max_var), day_length = day_length, lights_off = lights_off)
        fig.update_layout(shapes=list(bar_shapes.values()))
    
        fig['layout']['xaxis']['range'] = [t_min, t_max]

        if isinstance(save, str):
            if save.endswith('.html'):
                fig.write_html(save)
            else:
                fig.write_image(save, width=1500, height=650)
            print(f'Saved to {save}')
            fig.show()
        else:
            fig.show()

    @staticmethod
    def _plot_ylayout(fig, yrange, t0, dtick, ylabel, title, secondary = False, xdomain = False, ytype = "-", grid = False):
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
                                x = 0.85,
                                y = 0.99
                            )
                        )
        fig['layout'][axis].update(
                        linecolor = 'black',
                        type = ytype,
                        tick0 = t0,
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
                        linewidth = 4
                    )
        if yrange is not False:
            fig['layout'][axis]['range'] = yrange
        if dtick is not False:
            fig['layout'][axis]['dtick'] = dtick
        if secondary is not False:
            fig['layout'][axis]['side'] = 'right'
            fig['layout'][axis]['overlaying'] = 'y'
            fig['layout'][axis]['anchor'] = xdomain
        if grid is False:
            fig['layout'][axis]['showgrid'] = False
        else:
            fig['layout'][axis]['showgrid'] = True
            fig['layout'][axis]['gridcolor'] = 'black'

    @staticmethod
    def _plot_xlayout(fig, xrange, t0, dtick, xlabel, domains = False, axis = None, type = "-"):
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
                        linewidth = 4
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
    def _plot_boxpoints(y, x, colour, showlegend, name, xaxis):
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
        return trace_box


    
class behavpy_plotly(behavpy_base2):
    """
    A wrapper around the behavpy object
    Handles plotting of figures using the preferred canvas
    Current choice is between plotly and seaborn
    """

    def __init__(self, *args, **kwargs):
        super(behavpy_plotly, self).__init__(*args, **kwargs)


    def heatmap(self, variable = 'moving', title = '', save = None):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals using plotly
        
        :param str variable: Name of the column containing the variable of interest, the default is moving
        :param str title: The title of the plot
        :param save: The filename of the figure we want to save
        :type save: str or None
        
        :return: The figure instance
        """
        
        time_list, heatmap_df = self._heatmap(variable = variable)
        
        gbm = heatmap_df.groupby(heatmap_df.index)[f'{variable}_mean'].apply(list)
        id = heatmap_df.groupby(heatmap_df.index)['t_bin'].mean().index.tolist()

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
                tickfont = dict(
                    size = 16
                ),
                linewidth = 2)
                )
        return fig



class behavpy_sns(behavpy_base2):
    def __init__(self, *args, **kwargs):
        super(behavpy_sns, self).__init__(*args, **kwargs)

    def heatmap(self, variable = 'moving', title = '', save = None, reverse=True):
        """
        Creates an aligned heatmap of the movement data binned to 30 minute intervals using seaborn
        
        :param str variable: Name of the column containing the variable of interest, the default is moving
        :param str title: The title of the plot
        :param save: The filename of the figure we want to save
        :type save: str or None
        
        :return: The figure instance
        """
        
        _, hmdf = self._heatmap(variable = variable)
        heatmap_df = pd.pivot( hmdf , columns='t_bin')[::-1 if reverse else 1]
        fig, ax = plt.subplots(figsize=(16,3)) 
        sns.heatmap(heatmap_df, ax=ax, cmap="viridis")
        return ax
        

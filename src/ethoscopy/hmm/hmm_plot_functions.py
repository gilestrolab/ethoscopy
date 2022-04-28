import pandas as pd  
import numpy as np 
import warnings
from pandas.core.reshape import merge

import plotly.graph_objs as go 
from plotly.subplots import make_subplots

from hmmlearn import hmm
from math import floor, ceil
from colour import Color
from scipy.stats import zscore

from behavpy_class import behavpy
from rle import rle
from bootstrap_CI import bootstrap
from hmm_functions import hmm_pct_state, hmm_pct_transition, hmm_mean_length
from format_warnings import format_Warning

pd.options.mode.chained_assignment = None
warnings.formatwarning = format_Warning

def hmm_overview(df, hmm, labels, colours, bin = 60, wrapped = True, title = '', curate = None, save = False, location = ''):
    """
    Creates a plot of all states overlayed with y-axis shows the liklihood of being in a sleep state and the x-axis showing time in hours.
    The plot is generated through the plotly package

    Params:
    @df = behavpy,
    @hmm = hmmlearn.hmm.MultinomialHMM,
    @labels = list[string], the names of the different states present in the hidden markov model
    @colours = list[string], the name of the colours you wish to represent the different states, must be the same length as labels.
    It accepts a specific colour or an array of numbers that are mapped to the color scale relative that are acceptable to plotly
    @bin = int, the time in seconds you want to bin the movement data to, default is 60 or 1 minute
    @wrapped = bool, if True the plot will be limited to a 24 hour day average
    @title = string, the title of the plot when generated
    @curate = bool, a check that each fly data provided has more than 24 hours worth of time points
    @save = bool, if true the plot will be saved to local
    @location = string, only needed if save is True, provide the location and file type of the plot
    
    returns None
    """

    assert isinstance(df, behavpy)

    list_states = list(range(len(labels)))

    # change the movement column of choice to intergers, 1 == active, 0 == inactive
    df['moving'] = np.where(df['moving'] == True, 1, 0)

    # bin the data to 60 second intervals with a selected column and function on that column
    bin_df = df.bin_time('moving', bin, function= 'max')

    gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list)
    gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

    if curate is not None:
        curated_gb = []
        curated_gb2 = []

        for i, q in zip(gb, gb2):
            if len(i) and len(q) >= curate * 60:
                curated_gb.append(i)
                curated_gb2.append(q)

        gb = curated_gb
        gb2 = curated_gb2

    def decode_array(nested_list):

        logprob_list = []
        states_list = []

        for i in nested_list:
            seq = np.array(i)
            seq = seq.reshape(-1, 1)
            logprob, states = hmm.decode(seq)

            logprob_list.append(logprob)
            states_list.append(states)
            
        return logprob_list, states_list

    _, states = decode_array(gb)

    df_list = pd.DataFrame()

    for l, t in zip(states, gb2):
        df = hmm_pct_state(l, t, list_states, avg_window = int(1800/bin))
        df_list = df_list.append(df, ignore_index= True)

    if wrapped is True:
        df_list['t'] = df_list['t'].map(lambda t: t % 86400)

    df_list['t'] = df_list['t'] / (60*60)
    t_min = int(12 * floor(df_list.t.min() / 12))
    t_max = int(12 * ceil(df_list.t.max() / 12))    
    t_range = [t_min, t_max]  

    def pop_std(array):
        return np.std(array, ddof = 0)

    stats_dict = {}

    for state in list_states:
        stats_dict['df' + str(state)] = df_list.groupby('t').agg(**{
                    'mean' : (f'state_{state}', 'mean'), 
                    'SD' : (f'state_{state}', pop_std),
                    'count' : (f'state_{state}', 'count')
                })

    layout = go.Layout(
        title = title,
        yaxis = dict(
            color = 'black',
            linecolor = 'black',
            title = dict(
                text = 'Probability of being in state',
                font = dict(
                    size = 24,
                )
            ),
            range = [-0.025, 1], 
            tick0 = 0,
            dtick = 0.2,
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            )
        ),
        xaxis = dict(
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            title = dict(
                text = 'ZT (Hours)',
                font = dict(
                    size = 24,
                    color = 'black'
                )
            ),
            range = t_range,
            tick0 = 0,
            dtick = 6,
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            )
        ),
        plot_bgcolor = 'white',
        yaxis_showgrid=False,
        xaxis_showgrid = False,
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

    fig = go.Figure(layout = layout)

    for i, c, n in zip(list_states, colours, labels):

        loop_df = stats_dict[f'df{i}']

        loop_df['SE'] = (1.96*loop_df['SD']) / np.sqrt(loop_df['count'])
        loop_df['y_max'] = loop_df['mean'] + loop_df['SE']
        loop_df['y_min'] = loop_df['mean'] - loop_df['SE']

        y = loop_df['mean']
        y_upper = loop_df['y_max']
        y_lower = loop_df['y_min']
        x = loop_df.index.values

        upper_bound = go.Scatter(
        showlegend = False,
        x = x,
        y = y_upper,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0,
                shape = 'spline'
                ),
        )
        fig.add_trace(upper_bound)

        trace = go.Scatter(
        x = x,
        y = y,
        mode = 'lines',
        name = n,
        line = dict(
            shape = 'spline',
            color = c
            ),
        fill = 'tonexty'
        )
        fig.add_trace(trace)

        lower_bound = go.Scatter(
        showlegend = False,
        x = x,
        y = y_lower,
        mode='lines',
        marker=dict(
            color = c
            ),
        line=dict(width = 0,
                shape = 'spline'
                ),
        fill = 'tonexty'
        )  
        fig.add_trace(lower_bound)

    # Light-Dark annotaion bars
    bar_shapes = {}

    for i, bars in enumerate(range(t_min, t_max, 12)):
        if bars % 24 == 0:
            bar_col = 'white'
        else:
            bar_col = 'black'

        bar_shapes[f'shape_{i}'] = go.layout.Shape(type="rect", 
                                                        x0=bars, 
                                                        y0=-0.025, 
                                                        x1=bars+12, 
                                                        y1=0, 
                                                        line=dict(
                                                            color="black", 
                                                            width=1) ,
                                                        fillcolor=bar_col
                                                    )

    fig.update_layout(shapes=list(bar_shapes.values()))

    if save is True:
        fig.write_image(location, width=1000, height=650)
        print(f'Saved to {location}')
    else:
        fig.show()

def hmm_prior_run(df, hmm, run_length, ref_state, bin = 60, title = '', save = False, location = ''):
    """
    function to create a box plot of probability of prior minutes movement in relation to a start state"""

    assert isinstance(df, behavpy)

    # change the movement column of choice to intergers, 1 == active, 0 == inactive
    df['moving'] = np.where(df['moving'] == True, 1, 0)
    # bin the data to 60 second intervals with a selected column and function on that column
    bin_df = df.bin_time('moving', bin, function= 'max')

    gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list)

    def decode_array(nested_list):

        states_list = []

        for i in range(len(nested_list)):
            seq = np.array(nested_list[i])
            seq = seq.reshape(-1, 1)

            _, states = hmm.decode(seq)
            states_list.append(states)
            
        return states_list

    states = decode_array(gb)

    for counter, (i, j) in enumerate(zip(states, gb)):
        value, start, _ = rle(i)
        con = np.stack([value, start], axis = 1)
        test = con[con[:,0] == ref_state]
        idx = test[:,1]
        c = 0
        for id in idx:
            st = id - run_length
            end = id
            filt = np.array(j, dtype = np.int64)[st:end]
            filt = filt.flatten()
            if len(filt) != run_length:
                continue
            if c == 0:
                first_run = filt
                c += 1
            elif c == 1:
                final = np.vstack((first_run, filt))
                c += 1
            else:
                final = np.vstack((final, filt))
                c += 1
        if counter == 0:
            first_f = final.mean(axis=0)
        elif counter == 1:
            final_final = np.vstack((first_f, final.mean(axis=0)))
        else:
            final_final = np.vstack((final_final, final.mean(axis=0)))

    layout = go.Layout(
        title = title,
        yaxis = dict(
            color = 'black',
            linecolor = 'black',
            title = dict(
                text = 'Probability of Movement',
                font = dict(
                    size = 24,
                )
            ),
            range = [0, 1], 
            tick0 = 0,
            dtick = 0.25,
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            )
        ),
        xaxis = dict(
            color = 'black',
            linecolor = 'black',
            gridcolor = 'black',
            title = dict(
                text = 'Minutes prior',
                font = dict(
                    size = 24,
                    color = 'black'
                )
            ),
        ),
        plot_bgcolor = 'white',
        yaxis_showgrid=False,
        xaxis_showgrid = False,
        showlegend = False
    )

    fig = go.Figure(layout = layout)

    positions = list(range(run_length))
    r_positions = np.array(positions[::-1]) + 1

    for p in positions:
        name = f'{r_positions[p]}'

        fig.add_trace(go.Box(
                y = final_final[:, p],
                name = name,
                notched = True,
                boxmean = True,
                boxpoints = 'all',
                jitter = 0.75, 
                pointpos = 0
            )
        )

    if save is True:
        fig.write_image(location, width=800, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_state_runs(df, states, hmm_list, mago_df = None, hmm_compare = False, num_plots = 5, bin_list = [60], title = '', save = False, location = ''):
    """ plots the raw dedoded hmm model per fly (total = num_plots) """

    assert isinstance(df, behavpy)

    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    if mago_df is not None:
        assert isinstance(mago_df, behavpy)

    if len(hmm_list) > 1:
        num_plots = len(hmm_list)

    fig = make_subplots(
    rows= num_plots, 
    cols=1,
    shared_xaxes=True, 
    shared_yaxes=True, 
    vertical_spacing=0.02,
    horizontal_spacing=0.02
    )

    def decode(data, bin, hmm):
        # change the movement column to int and bin to time and group into list for decoding
        data['moving'] = np.where(data['moving'] == True, 1, 0)
        bin_df = df.bin_time('moving', bin, function= 'max')
        gb_m = bin_df.groupby(bin_df.index)['moving_max'].apply(list).reset_index(name='moving')
        gb_t = bin_df.groupby(bin_df.index)['t_bin'].apply(list).reset_index(name='time')
        gb_merge = pd.merge(gb_m, gb_t, on = 'id')

        mov = gb_merge['moving']
        time = gb_merge['time']
        id = gb_merge['id']

        temp_df = pd.DataFrame()

        for i in range(len(mov)):
            t = np.array(time[i])
            seq_orig = np.array(mov[i])
            seq = seq_orig.reshape(-1, 1)
            _, states = hmm.decode(seq)
            label = [id[i]] * len(t)
            previous_state = np.array(states[:-1], dtype = float)
            previous_state = np.insert(previous_state, 0, np.nan)
            previous_moving = np.array(seq_orig[:-1], dtype = float)
            previous_moving = np.insert(previous_moving, 0, np.nan)
            all = zip(label, t, states, previous_state, seq_orig, previous_moving)
            all = pd.DataFrame(data = all)
            temp_df = pd.concat([temp_df, all], ignore_index = False)

        temp_df.columns = ['id', 'bin', 'state', 'previous_state', 'moving', 'previous_moving']
        temp_df.set_index('id', inplace = True)

        return temp_df

    if hmm_compare is True:
        decoded_dict = {f'df{c}' : decode(df, bin = b, hmm = h) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
        rand_flies = np.random.permutation(list(set(decoded_dict['df0'].index)))[0]
        ite_len = len(hmm_list)
    else:
        decoded_dict = {'df0' : decode(df, bin = bin_list[0], hmm = hmm_list[0])}
        rand_flies = np.random.permutation(list(set(decoded_dict['df0'].index)))[:num_plots]
        ite_len = len(rand_flies)

    if states == 2:
        colorsIdx = {0: 'blue', 1: 'red'}
    elif states == 3:
        colorsIdx = {0: 'blue', 1: 'green', 2: 'red'}
    elif states == 4:
        colorsIdx = {0: 'darkblue', 1: 'dodgerblue', 2: 'red', 3: 'darkred'}
    elif states == 5:
        colorsIdx = {0: 'darkblue', 1: 'dodgerblue', 2: 'hotpink', 3: 'red', 4: 'darkred'}

    for i in range(ite_len):
        
        if hmm_compare is True:
            df = decoded_dict[f'df{i}'].filter(like = rand_flies, axis = 0)
            print(rand_flies)
        else:
            df = decoded_dict['df0'].filter(like = rand_flies[i], axis = 0)
            print(rand_flies[i])

        df['col'] = df['previous_state'].map(colorsIdx)

        if mago_df is not None:
            if hmm_compare is True:
                df2 = mago_df.filter(like = rand_flies, axis = 0)
            else:
                df2 = mago_df.filter(like = rand_flies[i], axis = 0)
            df2 = df2[df2['has_interacted'] == 1]
            df2['bin'] = df2['interaction_t'].map(lambda t:  bin_list[0] * floor(t / bin_list[0]))
            df2.reset_index(inplace = True)
            df = pd.merge(df, df2, how = 'outer', on = ['id', 'bin'])
            df['col'] = np.where(df['has_responded'] == True, 'purple', df['col'])
            df['col'] = np.where(df['has_responded'] == False, 'lime', df['col'])
            df['bin'] = df['bin'].map(lambda t: t / (60*60))
        
        else:
            df['bin'] = df['bin'].map(lambda t: t / (60*60))
            
        mov = df['previous_moving']
        
        trace1 = go.Scatter(
            showlegend = False,
            y = df['previous_state'],
            x = df['bin'],
            mode = 'markers+lines', 
            marker = dict(
                color = df['col'].tolist(),
                ),
            line = dict(
                color = 'black',
                width = 0.5
            )
            )
        fig.add_trace(trace1, row = i+1, col= 1)

        trace2 = go.Scatter(
            showlegend = False,
            y = mov,
            x = df['bin'],
            mode = 'lines',
            line = dict(
                color = 'black',
                width = 2
            )
            )
        fig.add_trace(trace2, row = i+1, col= 1)

    fig.update_layout(
        title = title,
        plot_bgcolor = 'white',
        legend = dict(
            bgcolor = 'rgba(201, 201, 201, 1)',
            bordercolor = 'grey',
            font = dict(
                size = 22
            ),
            x = 0.5,
            y = 0.90
        )
    )
    y_range = [-0.2, states-0.8]

    fig.update_yaxes(
        showgrid = False,
        linecolor = 'black',
        zeroline = False,
        ticks = 'outside',
        range = y_range, 
        tick0 = 0, 
        dtick = 1,
        tickwidth = 2,
        tickfont = dict(
            size = 18
        ),
        linewidth = 4
    )
    fig.update_xaxes(
        showgrid = False,
        color = 'black',
        linecolor = 'black',
        ticks = 'outside',
        tickwidth = 2,
        tickfont = dict(
            size = 18
        ),
        linewidth = 4
    )

    fig.update_yaxes(
        title = dict(
            text = 'Predicted State',
            font = dict(
                size = 20,
                color = 'black'
            )
        ),
        row = ceil(num_plots/2),
        col = 1
    )

    fig.update_xaxes(
        title = dict(
            text = 'ZT Hours',
            font = dict(
                size = 20,
                color = 'black'
            )
        ),
        row = num_plots,
        col = 1
    )

    if save is True:
        fig.write_image(location, width=1000, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_non_response_length(df, mago_df, states, hmm, column, variables, bin = 60, title = ''):
    """doesn't work - see individual script"""

    assert isinstance(df, behavpy)
    assert isinstance(mago_df, behavpy)

    #treatment = list(set(df.meta[variable]))
    all = pd.DataFrame()

    for treat in variables:
        print(treat)
        fil_df = df.xmv(column, treat)

        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        fil_df['moving'] = np.where(fil_df['moving'] == True, 1, 0)
        # bin the data to 60 second intervals with a selected column and function on that column
        bin_df = df.bin_time('moving', bin, function= 'max')

        gb_m = bin_df.groupby(bin_df.index)['moving_max'].apply(list).reset_index(name='moving')
        gb_t = bin_df.groupby(bin_df.index)['t_bin'].apply(list).reset_index(name='time')

        gb_merge = pd.merge(gb_m, gb_t, on = 'id')

        def decode_array(dataframe):
            mov = dataframe['moving']
            time = dataframe['time']
            id = dataframe['id']

            temp_df = pd.DataFrame()

            for i in range(len(mov)):
                t = np.array(time[i])
                seq = np.array(mov[i])
                seq = seq.reshape(-1, 1)
                logprob, states = hmm.decode(seq)
                label = [id[i]] * len(t)
                previous_state = np.array(states[:-1], dtype = float)
                previous_state = np.insert(previous_state, 0, np.nan)
                all = zip(label, t, states, previous_state, seq)
                all = pd.DataFrame(data = all)
                temp_df = temp_df.append(all, ignore_index = False)

            temp_df.columns = ['id', 'bin', 'state', 'previous_state', 'moving']
            temp_df.set_index('id', inplace = True)
                
            return temp_df

        s = decode_array(gb_merge)
        s['change'] = s['state'] - s['previous_state']
        s['change'] = np.where(s['change'] > 0, True, False)
        s.dropna(inplace = True)

        if states == 2:
            colorsIdx = {0: 'blue', 1: 'red'}
        elif states == 3:
            colorsIdx = {0: 'blue', 1: 'green', 2: 'red'}
        elif states == 4:
            colorsIdx = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}

        s['col'] = s['state'].map(colorsIdx)


        df2 = mago_df.xmv(column, treat)
        df2 = df2[df2['t_rel'] == 0]
        df2['bin'] = df2['interaction_t'].map(lambda t:  bin * floor(t / bin))
        df2.reset_index(inplace = True)
        df_merge = pd.merge(s, df2, how = 'outer', on = ['id', 'bin'])
        df_merge['col'] = np.where((df_merge['t_rel'] == 0) & (df_merge['change'] == True), 'purple', df_merge['col'])
        df_merge['col'] = np.where((df_merge['t_rel'] == 0) & (df_merge['change'] == False), 'orangered', df_merge['col'])
        df_merge['bin'] = df_merge['bin'].map(lambda t: t / (60*60))
        df_merge = df_merge[['id','state', 'col']]

        all_df = pd.DataFrame()

        for _, fly in df_merge.groupby('id'):
            v, s, l = rle(fly['state'])

            list_runs = np.array([], dtype = np.int)
            for counter, run in enumerate(l):
                x = [counter] * run 
                list_runs = np.append(list_runs, x)

            fly['run'] = list_runs

            counts = np.array([])
            for i in fly.groupby('run')['col'].apply(list):
                count = 0
                for q in i:
                    if q == 'orangered':
                        count += 1
                counts = np.append(counts, count)

            counts_df = pd.DataFrame(data = {'state' : v, 'o_count' : counts})
            counts_df = counts_df[counts_df['o_count'] >= 1]
            agg_df = counts_df.groupby('state').agg(**{
                                        'mean_count' : ('o_count', 'mean')
            })
            if agg_df.empty:
                continue
            else:
                all_df = all_df.append(agg_df)

        def pop_std(array):
            return np.std(array, ddof = 0)

        big_agg_gb = all_df.groupby('state').agg(**{
                                'avg_run' : ('mean_count', 'mean'),
                                'SD' : ('mean_count', pop_std),
                                'count' : ('mean_count', 'count'),
        })

        big_agg_gb['se'] = (1.96*big_agg_gb['SD']) / np.sqrt(big_agg_gb['count'])

        all = all.append(big_agg_gb)

    print(all)

    trace_1 = go.Bar(
        showlegend = False,
        y = [all['avg_run'].iloc[0], all['avg_run'].iloc[2]], 
        x = variables,
        name = 'Deep Sleep',
        marker = dict(
            line = dict(
                color = 'blue',
                width = 4
            ),
        color = 'rgba(0, 0, 255, 0.5)'
        ),
        error_y = dict(
            type = 'data',
            symmetric = True,
            array = [all['se'].iloc[0], all['se'].iloc[2]], 
            color = 'black',
            thickness = 1,
            width = 20
        )
    )

    trace_2 = go.Bar(
        showlegend = False,
        y = [all['avg_run'].iloc[1], all['avg_run'].iloc[3]], 
        name = 'Light Sleep',
        xaxis = 'x2',
        marker = dict(
            line = dict(
                color = 'green',
                width = 4
            ),
        color = 'rgba(0, 128, 0, 0.2)'
        ),
        error_y = dict(
            type = 'data',
            symmetric = True,
            array = [all['se'].iloc[1], all['se'].iloc[3]], 
            color = 'black',
            thickness = 1,
            width = 20
        )
    )

    layout = go.Layout(
    title = '4 states response rate',
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Liklihood to change state',
                font = dict(
                    size = 20,
                    color = 'black'
                )
            ),
            range = [0, 4],
            tick0 = 0,
            dtick = 0.5,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 4
        ), 
        xaxis = dict(
            domain=[0, 0.5],
            showgrid = False,
            color = 'black',
            linecolor = 'black',
            title = dict(
                text = 'Deep Sleep',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 4
        ),
        xaxis2 = dict(
            domain=[0.5, 1],
            showgrid = False,
            color = 'black',
            linecolor = 'black',
            title = dict(
                text = 'Light Sleep',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 4
        ),
        boxmode = 'group', 
        boxgroupgap = 1,
        plot_bgcolor = 'white',
        legend = dict(
            bgcolor = 'rgba(201, 201, 201, 1)',
            bordercolor = 'grey',
            font = dict(
                size = 22
            ),
            x = 0.5,
            y = 0.90
        )
    )

    fig = go.Figure(data = [trace_1, trace_2], layout = layout)
    fig.show()

def hmm_state_length(df_list, hmm_list, labels, colours, bin_list = [60], title = '', curate = None, group_labels = None, hmm_compare = False, save = False, location = ''):     

    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    if group_labels is not None:
        if len(group_labels) != len(df_list) and len(group_labels) != len(hmm_list):
            warnings.warn('There are not enough labels for the different groups or vice versa')
            exit()
    else:
        group_labels = ['']

    for d in df_list:
        assert isinstance(d, behavpy)

    list_states = list(range(len(labels)))

    def decode(data, bin, hmm, curate):
        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        data['moving'] = np.where(data['moving'] == True, 1, 0)
        # bin the data to 60 second intervals with a selected column and function on that column
        bin_df = data.bin_time('moving', bin, function= 'max')

        gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list)
        gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

        if curate is not None:
            curated_gb = []
            curated_gb2 = []

            for i, q in zip(gb, gb2):
                if len(i) and len(q) >= (curate*60):
                    curated_gb.append(i)
                    curated_gb2.append(q)

            gb = curated_gb
            gb2 = curated_gb2

        states_list = []

        for i in range(len(gb)):
            seq = np.array(gb[i])
            seq = seq.reshape(-1, 1)

            _, states = hmm.decode(seq)

            states_list.append(states)

        return states_list, gb2

    if hmm_compare is True:
        if len(df_list) == 1:
            decoded_dict = {f'df{c}' : decode(df_list[0], bin = b, hmm = h, curate = curate) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
        else:
            decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h, curate = curate) for c, (d, h, b) in enumerate(zip(df_list, hmm_list, bin_list))}
        ite_len = len(hmm_list)
    else:
        decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0], curate = curate) for c, d in enumerate(df_list)}
        ite_len = len(df_list)

    def analysis(states, t_diff):

        df_lengths = pd.DataFrame()

        for l in states:
            length = hmm_mean_length(l, delta_t = t_diff) 
            df_lengths = df_lengths.append(length, ignore_index= True)
        return df_lengths

    if hmm_compare is True:
        analysed_dict = {f'df{c}' : analysis(v[0], t_diff = b) for c, (v, b) in enumerate(zip(decoded_dict.values(), bin_list))}
    else:
        analysed_dict = {f'df{c}' : analysis(v[0], t_diff = bin_list[0]) for c, v in enumerate(decoded_dict.values())}

    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            type = 'log',
            dtick = 0.69897000433,
            title = dict(
                text = 'Length of state bout (mins)',
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
        ),
        plot_bgcolor = 'white',
    )

    fig = go.Figure(layout = layout)

    for c, v in enumerate(analysed_dict.values()):
        v['labels'] = group_labels[c]

    gb_dict = {f'gb{c}' : v.groupby('state') for c, v in enumerate(analysed_dict.values())}

    for state, col, lab in zip(list_states, colours, labels):

        median_list = np.array([])
        q3_list = np.array([])
        q1_list = np.array([])

        for i in range(ite_len):
            try:
                array = gb_dict[f'gb{i}'].get_group(state)
                if len(array) == 1:
                    median_list = np.append(median_list, array['mean_length'].mean())
                    q3_list = np.append(q3_list, bootstrap(array['mean_length'])[1])
                    q1_list = np.append(q1_list, bootstrap(array['mean_length'])[0])

                else:
                    median_list = np.append(median_list, array['mean_length'][np.abs(zscore(array['mean_length'])) < 3].mean())
                    q3_list = np.append(q3_list, bootstrap(array['mean_length'][np.abs(zscore(array['mean_length'])) < 3])[1])
                    q1_list = np.append(q1_list, bootstrap(array['mean_length'][np.abs(zscore(array['mean_length'])) < 3])[0])


            except:
                median_list = np.append(median_list, 0)
                q3_list = np.append(q3_list, 0)
                q1_list = np.append(q1_list, 0)

        for c, i in enumerate(group_labels):

            if 'baseline' in i or 'control' in i:
                if 'rebound' in i:
                    marker_col = 'black'
                else:
                    marker_col = 'grey'
            else:
                if 'rebound' in i:
                    marker_col = f'dark{col}'
                else:
                    marker_col = col

            trace_box = go.Box(
                showlegend = False,
                median = [median_list[c]],
                q3 = [q3_list[c]],
                q1 = [q1_list[c]],
                x = [group_labels[c]],
                xaxis = f'x{state+1}',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = False,
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9,
            )
            fig.add_trace(trace_box)

            try:
                array = gb_dict[f'gb{c}'].get_group(state)
                if len(array) == 1:
                    con_list = array['mean_length']
                    label_list = array['labels']
                else:
                    con_list = array['mean_length'][np.abs(zscore(array['mean_length'])) < 3]
                    label_list = array['labels'][np.abs(zscore(array['mean_length'])) < 3]

            except:
                con_list = np.array([0])
                label_list = np.array([group_labels[c]])

            trace_box2 = go.Box(
                showlegend = False,
                y = con_list, 
                x = label_list,
                xaxis = f'x{state+1}',
                line = dict(
                    color = 'rgba(0,0,0,0)'
                ),
                fillcolor = 'rgba(0,0,0,0)',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = 'all',
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9
            )
            fig.add_trace(trace_box2)

        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        axis = f'xaxis{state+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[state:state+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = lab,
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickmode = 'linear',
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_state_pct(df_list, hmm_list, labels, colours, facet_col = None, bin_list = [60], title = '', curate = None, group_labels = None, hmm_compare = False, save = False, location = ''):

    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    if facet_col is not None:
        if len(df_list) != 1:
            warnings.warn('If faceting, there can only be one behavpy object')
            exit()
        d_list = []
        arg_list = list(set(df_list[0].meta[facet_col].tolist()))
        for arg in arg_list:
            temp_df = df_list[0].xmv(facet_col, arg)
            d_list.append(temp_df)
        if group_labels is None:
            group_labels = arg_list
        elif len(d_list) != len(group_labels):
            warnings.warn('The number of group labels do not match the number of arguments in the facet column')
            group_labels = arg_list
    else:
        d_list = df_list

    if group_labels is not None:
        if len(group_labels) != len(d_list) and len(group_labels) != len(hmm_list):
            warnings.warn('There are not enough labels for the different groups or vice versa')
            exit()
    else:
        group_labels = ['']

    for d in d_list:
        assert isinstance(d, behavpy)

    list_states = list(range(len(labels)))

    def decode(data, bin, hmm, curate):
        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        data['moving'] = np.where(data['moving'] == True, 1, 0)
        # bin the data to 60 second intervals with a selected column and function on that column
        bin_df = data.bin_time('moving', bin, function= 'max')

        gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list)
        gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

        if curate is not None:
            curated_gb = []
            curated_gb2 = []

            for i, q in zip(gb, gb2):
                if len(i) and len(q) >= curate:
                    curated_gb.append(i)
                    curated_gb2.append(q)

            gb = curated_gb
            gb2 = curated_gb2

        states_list = []

        for i in range(len(gb)):
            seq = np.array(gb[i])
            seq = seq.reshape(-1, 1)

            _, states = hmm.decode(seq)

            states_list.append(states)

        return states_list, gb2

    if hmm_compare is True:
        if len(d_list) == 1:
            decoded_dict = {f'df{c}' : decode(d_list[0], bin = b, hmm = h, curate = curate) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
        else:
            decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h, curate = curate) for c, (d, h, b) in enumerate(zip(d_list, hmm_list, bin_list))}
        ite_len = len(hmm_list)
    else:
        decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0], curate = curate) for c, d in enumerate(d_list)}
        ite_len = len(d_list)

    def analysis(array_states):

        counts_all = pd.DataFrame()

        for c, i in enumerate(array_states):
            if len(labels) == 3:
                count_df = pd.DataFrame(data = {0 : (i == 0).sum(), 1 : (i == 1).sum(), 2 : (i == 2).sum()}, index = [c])
            if len(labels) == 4:
                count_df = pd.DataFrame(data = {0 : (i == 0).sum(), 1 : (i == 1).sum(), 2 : (i == 2).sum(), 3 : (i == 3).sum()}, index = [c])
            if len(labels) == 5:
                count_df = pd.DataFrame(data = {0 : (i == 0).sum(), 1 : (i == 1).sum(), 2 : (i == 2).sum(), 3 : (i == 3).sum(), 4 : (i == 4).sum()}, index = [c])
            counts_all = counts_all.append(count_df)
            
        counts_all['sum'] = counts_all.sum(axis=1)
        counts_all = counts_all.iloc[:, list_states[0]: list_states[-1]+1].div(counts_all['sum'], axis=0)
        
        return counts_all

    analysed_dict = {f'df{c}' : analysis(v[0]) for c, v in enumerate(decoded_dict.values())}
    
    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Fraction of time in each state',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            range = [0, 1.01],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18,
                color = 'black'
            ),
            linewidth = 4
        ),
        plot_bgcolor = 'white',
    )

    fig = go.Figure(layout = layout)

    for c, v in enumerate(analysed_dict.values()):
        v['labels'] = group_labels[c]

    for state, col, lab in zip(list_states, colours, labels):

        median_list = [analysed_dict[f'df{i}'][state][np.abs(zscore(analysed_dict[f'df{i}'][state])) < 3].mean() for i in range(ite_len)]
        q3_list = [bootstrap(analysed_dict[f'df{i}'][state][np.abs(zscore(analysed_dict[f'df{i}'][state])) < 3])[1] for i in range(ite_len)]
        q1_list = [bootstrap(analysed_dict[f'df{i}'][state][np.abs(zscore(analysed_dict[f'df{i}'][state])) < 3])[0] for i in range(ite_len)]

        for c, i in enumerate(group_labels):

            if 'baseline' in i.lower() or 'control' in i.lower() or 'ctrl' in i.lower():
                if 'rebound' in i.lower():
                    marker_col = 'black'
                else:
                    marker_col = 'grey'
            else:
                if 'rebound' in i.lower():
                    marker_col = f'dark{col}'
                else:
                    marker_col = col

            trace_box = go.Box(
                showlegend = False,
                median = [median_list[c]],
                q3 = [q3_list[c]],
                q1 = [q1_list[c]],
                x = [group_labels[c]],
                xaxis = f'x{state+1}',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = False,
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9,
            )
            fig.add_trace(trace_box)

            con_list = pd.Series(dtype = 'float64')
            label_list = pd.Series(dtype = 'str')
            con_list = pd.concat([con_list, analysed_dict[f'df{c}'][state][np.abs(zscore(analysed_dict[f'df{c}'][state])) < 3]])
            label_list = pd.concat([label_list, analysed_dict[f'df{c}']['labels'][np.abs(zscore(analysed_dict[f'df{c}'][state])) < 3]])

            trace_box2 = go.Box(
                showlegend = False,
                y = con_list, 
                x = label_list,
                xaxis = f'x{state+1}',
                line = dict(
                    color = 'rgba(0,0,0,0)'
                ),
                fillcolor = 'rgba(0,0,0,0)',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = 'all',
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9
            )
            fig.add_trace(trace_box2)

        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        axis = f'xaxis{state+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[state:state+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = lab,
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickmode = 'linear',
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_state_transition(df_list, hmm_list, labels, colours, bin_list = [60], title = '', curate = None, group_labels = None, hmm_compare = False, save = False, location = ''):
    
    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    if group_labels is not None:
        if len(group_labels) != len(df_list) and len(group_labels) != len(hmm_list):
            warnings.warn('There are not enough labels for the different groups or vice versa')
            exit()
    else:
        group_labels = ['']

    for d in df_list:
        assert isinstance(d, behavpy)

    list_states = list(range(len(labels)))

    def decode(data, bin, hmm, curate):
        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        data['moving'] = np.where(data['moving'] == True, 1, 0)
        # bin the data to 60 second intervals with a selected column and function on that column
        bin_df = data.bin_time('moving', bin, function= 'max')

        gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list)
        gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

        if curate is not None:
            curated_gb = []
            curated_gb2 = []

            for i, q in zip(gb, gb2):
                if len(i) and len(q) >= curate:
                    curated_gb.append(i)
                    curated_gb2.append(q)

            gb = curated_gb
            gb2 = curated_gb2

        logprob_list = []
        states_list = []

        for i in range(len(gb)):
            seq = np.array(gb[i])
            seq = seq.reshape(-1, 1)

            logprob, states = hmm.decode(seq)

            logprob_list.append(logprob)
            states_list.append(states)

        return states_list, gb2

    if hmm_compare is True:
        if len(df_list) == 1:
            decoded_dict = {f'df{c}' : decode(df_list[0], bin = b, hmm = h, curate = curate) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
        else:
            decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h, curate = curate) for c, (d, h, b) in enumerate(zip(df_list, hmm_list, bin_list))}
        ite_len = len(hmm_list)
    else:
        decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0], curate = curate) for c, d in enumerate(df_list)}
        ite_len = len(df_list)

    def analysis(states, time):

        df_trans = pd.DataFrame()

        for l, t in zip(states, time):
            trans = hmm_pct_transition(l, list_states) 
            df_trans = df_trans.append(trans, ignore_index= True)

        return df_trans

    analysed_dict = {f'df{c}' : analysis(v[0], v[1]) for c, v in enumerate(decoded_dict.values())}

    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Fraction of runs of each state',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            range = [0, 1.05],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18,
                color = 'black'
            ),
            linewidth = 4
        ),
        plot_bgcolor = 'white',
        legend = dict(
        bgcolor = 'rgba(201, 201, 201, 1)',
        bordercolor = 'grey',
        font = dict(
            size = 14
            ),
        )
    )

    fig = go.Figure(layout = layout)

    for c, v in enumerate(analysed_dict.values()):
        v['labels'] = group_labels[c]

    for state, col, lab in zip(list_states, colours, labels):
        median_list = [analysed_dict[f'df{i}'][str(state)][np.abs(zscore(analysed_dict[f'df{i}'][str(state)])) < 3].mean() for i in range(ite_len)]
        q3_list = [bootstrap(analysed_dict[f'df{i}'][str(state)][np.abs(zscore(analysed_dict[f'df{i}'][str(state)])) < 3])[1] for i in range(ite_len)]
        q1_list = [bootstrap(analysed_dict[f'df{i}'][str(state)][np.abs(zscore(analysed_dict[f'df{i}'][str(state)])) < 3])[0] for i in range(ite_len)]

        for c, i in enumerate(group_labels):

            if 'baseline' in i.lower() or 'control' in i.lower():
                if 'rebound' in i.lower():
                    marker_col = 'black'
                else:
                    marker_col = 'grey'
            else:
                if 'rebound' in i.lower():
                    marker_col = f'dark{col}'
                else:
                    marker_col = col

            trace_box = go.Box(
                showlegend = False,
                median = [median_list[c]],
                q3 = [q3_list[c]],
                q1 = [q1_list[c]],
                x = [group_labels[c]],
                xaxis = f'x{state+1}',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = False,
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9,
            )
            fig.add_trace(trace_box)

            con_list = pd.Series(dtype = 'float64')
            label_list = pd.Series(dtype = 'str')
            con_list = pd.concat([con_list, analysed_dict[f'df{c}'][str(state)][np.abs(zscore(analysed_dict[f'df{c}'][str(state)])) < 3]])
            label_list = pd.concat([label_list, analysed_dict[f'df{c}']['labels'][np.abs(zscore(analysed_dict[f'df{c}'][str(state)])) < 3]])

            trace_box2 = go.Box(
                showlegend = False,
                y = con_list, 
                x = label_list,
                xaxis = f'x{state+1}',
                line = dict(
                    color = 'rgba(0,0,0,0)'
                ),
                fillcolor = 'rgba(0,0,0,0)',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = 'all',
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9
            )
            fig.add_trace(trace_box2)

        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        axis = f'xaxis{state+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[state:state+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = lab,
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickmode = 'linear',
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_split_plot(df_list, hmm_list, labels = None, facet_col = None, hmm_compare = False, wrapped = False, curate = False, bin_list = [60], save = False, location = '', title = ''):
    """ Only works for 4 state models """
    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    if facet_col is not None:
        if len(df_list) != 1:
            warnings.warn('If faceting, there can only be one behavpy object')
            exit()
        d_list = []
        arg_list = list(set(df_list[0].meta[facet_col].tolist()))
        for arg in arg_list:
            temp_df = df_list[0].xmv(facet_col, arg)
            d_list.append(temp_df)
        if labels is None:
            labels = arg_list
        elif len(d_list) != len(labels):
            labels = arg_list
    else:
        d_list = df_list

    for d in d_list:
        assert isinstance(d, behavpy)

    if hmm_compare is True:
        if len(hmm_list) != len(d_list) or len(bin_list) != len(d_list):
            warnings.warn('There are not enough hmm models or bin ints for the different groups or vice versa')
            exit()

    fig = make_subplots(
        rows=2, 
        cols=2,
        shared_xaxes=True, 
        shared_yaxes=True, 
        vertical_spacing=0.02,
        horizontal_spacing=0.02
        )

    if hmm_compare is False:
        if len(hmm_list) != len(d_list):
            hmm_list = [hmm_list[0]] * len(d_list)
        if len(bin_list) != len(d_list):
            bin_list = [bin_list[0]] * len(d_list)

    colour_range_dict = {}
    for q in range(0,4):
        colours_dict = {'start' : ['#b2d8ff', '#8df086', '#eda866', '#ed776d'], 'end' : ['#00264c', '#086901', '#8a4300', '#700900']}
        start_color = colours_dict.get('start')[q]
        end_color = colours_dict.get('end')[q]
        N = len(d_list)
        colour_range_dict[q] = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]

    for c, (d, n, h, b) in enumerate(zip(d_list, labels, hmm_list, bin_list)):   
        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        d['moving'] = np.where(d['moving'] == True, 1, 0)
        # bin the data to 60 second intervals with a selected column and function on that column
        bin_gb = d.bin_time('moving', b, function= 'max')
        gb = bin_gb.groupby(bin_gb.index)['moving_max'].apply(list)
        gb2 = bin_gb.groupby(bin_gb.index)['t_bin'].apply(list)

        if curate is True:
            curated_gb = []
            curated_gb2 = []

            for i, q in zip(gb, gb2):
                if len(i) and len(q) >= 86400/b:
                    curated_gb.append(i)
                    curated_gb2.append(q)

            gb = curated_gb
            gb2 = curated_gb2

        def decode_array(nested_list):

            logprob_list = []
            states_list = []

            for i in nested_list:
                seq = np.array(i)
                seq = seq.reshape(-1, 1)
                
                logprob, states = h.decode(seq)

                logprob_list.append(logprob)
                states_list.append(states)
                
            return logprob_list, states_list

        _, states = decode_array(gb) 
        analsyed_df = pd.DataFrame()

        for l, t in zip(states, gb2):
            temp_df = hmm_pct_state(l, t, [0, 1, 2, 3], avg_window = int(1800/b))
            analsyed_df = analsyed_df.append(temp_df, ignore_index= True)

        if wrapped is True:
            analsyed_df['t'] = analsyed_df['t'].map(lambda t: t % 86400)
        analsyed_df['t'] = analsyed_df['t'] / (60*60)

        if n.lower() == 'control' or n.lower() == 'baseline':
            black_list = ['black'] * len(d_list)
            black_range_dict = {0 : black_list, 1: black_list, 2 : black_list, 3 : black_list}
            marker_col = black_range_dict
        else:
            marker_col = colour_range_dict

        t_min = int(12 * floor(analsyed_df.t.min() / 12))
        t_max = int(12 * ceil(analsyed_df.t.max() / 12))    
        t_range = [t_min, t_max]  
        
        def pop_std(array):
            return np.std(array, ddof = 0)

        for i, row, col in zip(range(4), [1,1,2,2], [1,2,1,2]):

            loop_df = analsyed_df.groupby('t').agg(**{
                        'mean' : (f'state_{i}', 'mean'), 
                        'SD' : (f'state_{i}', pop_std),
                        'count' : (f'state_{i}', 'count')
                    })

            loop_df['SE'] = (1.96*loop_df['SD']) / np.sqrt(loop_df['count'])
            loop_df['y_max'] = loop_df['mean'] + loop_df['SE']
            loop_df['y_min'] = loop_df['mean'] - loop_df['SE']

            y = loop_df['mean']
            y_upper = loop_df['y_max']
            y_lower = loop_df['y_min']
            x = loop_df.index.values

            upper_bound = go.Scatter(
            showlegend = False,
            x = x,
            y = y_upper,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0,
                    shape = 'spline'
                    ),
            )
            fig.add_trace(upper_bound,row=row, col=col)

            trace = go.Scatter(
            x = x,
            y = y,
            mode = 'lines',
            name = n,
            line = dict(
                shape = 'spline',
                color = marker_col.get(i)[c]
                ),
            fill = 'tonexty'
            )
            fig.add_trace(trace, row=row, col=col)

            lower_bound = go.Scatter(
            showlegend = False,
            x = x,
            y = y_lower,
            mode='lines',
            marker=dict(
                color = marker_col.get(i)[c]
                ),
            line=dict(width = 0,
                    shape = 'spline'
                    ),
            fill = 'tonexty'
            )  
            fig.add_trace(lower_bound, row=row, col=col)

    fig.update_xaxes(
        zeroline = False,
        color = 'black',
        linecolor = 'black',
        gridcolor = 'black',
        range = t_range,
        tick0 = 0,
        dtick = 6,
        ticks = 'outside',
        tickwidth = 2,
        tickfont = dict(
            size = 18
        ),
        linewidth = 2,
        showgrid = False
    )
    fig.update_yaxes(
        zeroline = False, 
        color = 'black',
        linecolor = 'black',
        range = [-0.05, 1], 
        tick0 = 0,
        dtick = 0.2,
        ticks = 'outside',
        tickwidth = 2,
        tickfont = dict(
            size = 18
        ),
        linewidth = 4,
        showgrid = True
    )

    fig.update_layout(
        title = title,
        plot_bgcolor = 'white',
        legend = dict(
            bgcolor = 'rgba(201, 201, 201, 1)',
            bordercolor = 'grey',
            font = dict(
                size = 14
            ),
            x = 1.005,
            y = 0.5
        )
    )

    fig.update_layout(
        annotations=[
            go.layout.Annotation({
                'font': {'size': 22, 'color' : 'black'},
                'showarrow': False,
                'text': 'ZT Time (Hours)',
                'x': 0.5,
                'xanchor': 'center',
                'xref': 'paper',
                'y': 0,
                'yanchor': 'top',
                'yref': 'paper',
                'yshift': -30
            }),
            go.layout.Annotation({
                'font': {'size': 22, 'color' : 'black'},
                'showarrow': False,
                'text': 'Likelihood to be in sleep state',
                'x': 0,
                'xanchor': 'left',
                'xref': 'paper',
                'y': 0.5,
                'yanchor': 'middle',
                'yref': 'paper',
                'xshift': -85,
                'textangle' : -90
            })
        ]
    )
    # Light-Dark annotaion bars
    bar_shapes = {}

    for i, bars in enumerate(range(t_min, t_max, 12)):
        if bars % 24 == 0:
            bar_col = 'white'
        else:
            bar_col = 'black'
        for c in range(4):
            bar_shapes['shape_' + f'{i}-{c}'] = go.layout.Shape(type="rect", 
                                                        x0=bars, 
                                                        y0=-0.05, 
                                                        x1=bars+12, 
                                                        y1=-0.02, 
                                                        xref=f'x{c+1}', 
                                                        yref=f'y{c+1}',
                                                        line=dict(
                                                            color="black", 
                                                            width=1) ,
                                                        fillcolor=bar_col
                                                    )

    fig.update_layout(shapes=list(bar_shapes.values()))

    if save is True:
        fig.write_image(location, width=1150, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_day_night(df_list, hmm_list, labels, colours, facet_col = None, group_labels = None, bin_list = [60], title = '', hmm_compare = False, save = False, location = ''):

    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    for d in df_list:
        assert isinstance(d, behavpy)
        d.add_day_phase()

    if facet_col is not None:
        if len(df_list) != 1:
            warnings.warn('If faceting, there can only be one behavpy object')
            exit()
        d_list = []
        arg_list = list(set(df_list[0].meta[facet_col].tolist()))
        for arg in arg_list:
            temp_df = df_list[0].xmv(facet_col, arg)
            d_list.append(temp_df)
        if group_labels is None:
            group_labels = arg_list
        elif len(d_list) != len(group_labels):
            warnings.warn('The number of group labels do not match the number of arguments in the facet column')
            group_labels = arg_list
    else:
        d_list = df_list

    if hmm_compare is True:
        if len(hmm_list) != len(d_list) or len(bin_list) != len(d_list):
            warnings.warn('There are not enough hmm models or bin ints for the different groups or vice versa')
            exit()

    list_states = list(range(len(labels)))

    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Fraction of time asleep',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            range = [0, 1.01],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18,
                color = 'black'
            ),
            linewidth = 4
        ),
        plot_bgcolor = 'white',
    )

    fig = go.Figure(layout = layout)

    domain_count = 0

    for phase in ['light', 'dark']:

        phase_list = [d[d['phase'] == phase] for d in d_list]
        phase_list = [d.drop(['phase', 'day'], axis = 1) for d in phase_list]

        def decode(data, bin, hmm):
            # change the movement column of choice to intergers, 1 == active, 0 == inactive
            data['moving'] = np.where(data['moving'] == True, 1, 0)
            # bin the data to 60 second intervals with a selected column and function on that column
            bin_df = data.bin_time('moving', bin, function= 'max')

            gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list)
            gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

            logprob_list = []
            states_list = []

            for i in range(len(gb)):
                seq = np.array(gb[i])
                seq = seq.reshape(-1, 1)

                logprob, states = hmm.decode(seq)

                logprob_list.append(logprob)
                states_list.append(states)

            return states_list, gb2

        if hmm_compare is True:
            if len(d_list) == 1:
                decoded_dict = {f'df{c}' : decode(phase_list[0], bin = b, hmm = h) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
            else:
                decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h) for c, (d, h, b) in enumerate(zip(phase_list, hmm_list, bin_list))}
            ite_len = len(hmm_list)
        else:
            decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0]) for c, d in enumerate(phase_list)}
            ite_len = len(phase_list)

        def analysis(array_states):

            counts_all = pd.DataFrame()

            for c, i in enumerate(array_states):
                if len(labels) == 3:
                    count_df = pd.DataFrame(data = {0 : (i == 0).sum(), 1 : (i == 1).sum(), 2 : (i == 2).sum()}, index = [c])
                if len(labels) == 4:
                    count_df = pd.DataFrame(data = {0 : (i == 0).sum(), 1 : (i == 1).sum(), 2 : (i == 2).sum(), 3 : (i == 3).sum()}, index = [c])
                if len(labels) == 5:
                    count_df = pd.DataFrame(data = {0 : (i == 0).sum(), 1 : (i == 1).sum(), 2 : (i == 2).sum(), 3 : (i == 3).sum(), 4 : (i == 4).sum()}, index = [c])
                counts_all = counts_all.append(count_df)
                
            counts_all['sum'] = counts_all.sum(axis=1)
            counts_all = counts_all.iloc[:, list_states[0]: list_states[-1]+1].div(counts_all['sum'], axis=0)

            return counts_all

        analysed_dict = {f'df{c}' : analysis(v[0]) for c, v in enumerate(decoded_dict.values())}
        
        for index, v in enumerate(analysed_dict.values()):
            v['labels'] = group_labels[index]

        for state, col, lab in zip(list_states, colours, labels):

            median_list = [analysed_dict[f'df{i}'][state][np.abs(zscore(analysed_dict[f'df{i}'][state])) < 3].mean() for i in range(ite_len)]
            q3_list = [bootstrap(analysed_dict[f'df{i}'][state][np.abs(zscore(analysed_dict[f'df{i}'][state])) < 3])[1] for i in range(ite_len)]
            q1_list = [bootstrap(analysed_dict[f'df{i}'][state][np.abs(zscore(analysed_dict[f'df{i}'][state])) < 3])[0] for i in range(ite_len)]

            for c, i in enumerate(group_labels):
                try:
                    if 'baseline' in i.lower() or 'control' in i.lower():
                        if 'rebound' in i.lower():
                            marker_col = 'black'
                        else:
                            marker_col = 'grey'
                    else:
                        if 'rebound' in i.lower():
                            marker_col = f'dark{col}'
                        else:
                            marker_col = col
                except:
                    marker_col = col

                trace_box = go.Box(
                    showlegend = False,
                    median = [median_list[c]],
                    q3 = [q3_list[c]],
                    q1 = [q1_list[c]],
                    x = [group_labels[c]],
                    xaxis = f'x{domain_count+1}',
                    marker = dict(
                        color = marker_col,
                        opacity = 0.5,
                        size = 4
                    ),
                    boxpoints = False,
                    jitter = 0.75, 
                    pointpos = 0, 
                    width = 0.9,
                )
                fig.add_trace(trace_box)

                con_list = pd.Series(dtype = 'float64')
                label_list = pd.Series(dtype = 'str')
                con_list = pd.concat([con_list, analysed_dict[f'df{c}'][state][np.abs(zscore(analysed_dict[f'df{c}'][state])) < 3]])
                label_list = pd.concat([label_list, analysed_dict[f'df{c}']['labels'][np.abs(zscore(analysed_dict[f'df{c}'][state])) < 3]])

                trace_box2 = go.Box(
                    showlegend = False,
                    y = con_list, 
                    x = label_list,
                    xaxis = f'x{domain_count+1}',
                    line = dict(
                        color = 'rgba(0,0,0,0)'
                    ),
                    fillcolor = 'rgba(0,0,0,0)',
                    marker = dict(
                        color = marker_col,
                        opacity = 0.5,
                        size = 4
                    ),
                    boxpoints = 'all',
                    jitter = 0.75, 
                    pointpos = 0, 
                    width = 0.9
                )
                fig.add_trace(trace_box2)

            width = len(labels) * len(d_list)
            domains = np.arange(0, 1+(1/width), 1/width)

            axis = f'xaxis{domain_count+1}'
            fig['layout'][axis] = {}

            fig['layout'][axis].update(
                    domain = domains[domain_count:domain_count+2],
                    showgrid = False,
                    color = 'black',
                    linecolor = 'black',
                    title = dict(
                        text = lab,
                        font = dict(
                            size = 18,
                            color = 'black'
                        )
                    ),
                    ticks = 'outside',
                    tickwidth = 2,
                    tickmode = 'linear',
                    tickfont = dict(
                        size = 18
                    ),
                    linewidth = 4
                )
            domain_count += 1

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_total_sleep_ctrl_comp(df_list, states, hmm_list, labels, colours, facet_col = None, bin_list = [60], title = '', curate = False, hmm_compare = False, save = False, location = ''):

    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    if facet_col is not None:
        if len(df_list) != 1:
            warnings.warn('If faceting, there can only be one behavpy object')
            exit()
        d_list = []
        arg_list = list(set(df_list[0].meta[facet_col].tolist()))
        for arg in arg_list:
            temp_df = df_list[0].xmv(facet_col, arg)
            d_list.append(temp_df)
        if labels is None:
            labels = arg_list
        elif len(d_list) != len(labels):
            labels = arg_list
    else:
        d_list = df_list

    for d in d_list:
        assert isinstance(d, behavpy)

    if hmm_compare is True:
        if len(hmm_list) != len(d_list) or len(bin_list) != len(d_list):
            warnings.warn('There are not enough hmm models or bin integers for the different groups or vice versa')
            exit()
        if facet_col is not None:
            warnings.warn("You can't facet if comparing hmm models")
            exit()

    list_states = list(range(states))

    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Fraction of time asleep',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            range = [0, 1.01],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18,
                color = 'black'
            ),
            linewidth = 4
        ),
        plot_bgcolor = 'white',
    )

    fig = go.Figure(layout = layout)

    for c_orig, (d, col, lab) in enumerate(zip(d_list, colours, labels)):

        d_ctrl = d.xmv(xmv_args[0], xmv_args[1])
        d_test = d.xmv(xmv_args[0], lab)
        d_list = [d_ctrl, d_test]
            
        def decode(data, bin, hmm, curate):
            # change the movement column of choice to intergers, 1 == active, 0 == inactive
            data['moving'] = np.where(data['moving'] == True, 1, 0)
            # bin the data to 60 second intervals with a selected column and function on that column
            bin_df = data.bin_time('moving', bin, function= 'max')

            gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list)
            gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

            if curate is True:
                curated_gb = []
                curated_gb2 = []

                for i, q in zip(gb, gb2):
                    if len(i) and len(q) >= 86400/bin:
                        curated_gb.append(i)
                        curated_gb2.append(q)

                gb = curated_gb
                gb2 = curated_gb2

            logprob_list = []
            states_list = []

            for i in range(len(gb)):
                seq = np.array(gb[i])
                seq = seq.reshape(-1, 1)

                logprob, states = hmm.decode(seq)

                logprob_list.append(logprob)
                states_list.append(states)

            return states_list, gb2

        if hmm_compare is True:
            if len(df_list) == 1:
                decoded_dict = {f'df{c}' : decode(d_list[0], bin = b, hmm = h, curate = curate) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
            else:
                decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h, curate = curate) for c, (d, h, b) in enumerate(zip(d_list, hmm_list, bin_list))}
            ite_len = len(hmm_list)
        else:
            decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0], curate = curate) for c, d in enumerate(d_list)}
            ite_len = len(df_list)

        def analysis(array_states):

            counts_all = pd.DataFrame()

            for c, i in enumerate(array_states):
                count_df = pd.DataFrame(data = {0 : (i == 0).sum(), 1 : (i == 1).sum(), 2 : (i == 2).sum(), 3 : (i == 3).sum()}, index = [c])
                counts_all = counts_all.append(count_df)
                
            counts_all['sum'] = counts_all.sum(axis=1)
            counts_all = counts_all.iloc[:, list_states[0]: list_states[-1]+1].div(counts_all['sum'], axis=0)
            counts_all['total_sleep'] = counts_all[0] + counts_all[1]
            counts_all['total_awake'] = counts_all[2] + counts_all[3]
            counts_all.drop([0, 1, 2, 3], axis = 1, inplace = True)

            return counts_all

        analysed_dict = {f'df{c}' : analysis(v[0]) for c, v in enumerate(decoded_dict.values())}
        
        for c, v in enumerate(analysed_dict.values()):
            v['labels'] = xmv_labels[c]

        median_list = [analysed_dict[f'df{i}']['total_sleep'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3].mean() for i in range(len(xmv_labels))]
        q3_list = [bootstrap(analysed_dict[f'df{i}']['total_sleep'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3])[1] for i in range(len(xmv_labels))]
        q1_list = [bootstrap(analysed_dict[f'df{i}']['total_sleep'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3])[0] for i in range(len(xmv_labels))]

        for i in range(len(xmv_labels)):
            if i == 0:
                marker_col = ctrl_col
            else:
                marker_col = col
            trace_box = go.Box(
                showlegend = False,
                median = [median_list[i]], 
                q3 = [q3_list[i]], 
                q1 = [q1_list[i]],
                x = [xmv_labels[i]],
                xaxis = f'x{c_orig+1}',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = False,
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9,
            )
            fig.add_trace(trace_box)

            trace_box2 = go.Box(
                showlegend = False,
                y = analysed_dict[f'df{i}']['total_sleep'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3], 
                x = analysed_dict[f'df{i}']['labels'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3],
                xaxis = f'x{c_orig+1}',
                line = dict(
                    color = 'rgba(0,0,0,0)'
                ),
                fillcolor = 'rgba(0,0,0,0)',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = 'all',
                jitter = 0.75, 
                pointpos = 0, 
                width = 0.9
            )
            fig.add_trace(trace_box2)

        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        axis = f'xaxis{c_orig+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[c_orig:c_orig+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = lab,
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_total_sleep_day_night(df_list, hmm_list, labels, facet_col = None, bin_list = [60], title = '', hmm_compare = False, save = False, location = ''):

    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    for d in df_list:
        assert isinstance(d, behavpy)
        d.add_day_phase()

    if facet_col is not None:
        if len(df_list) != 1:
            warnings.warn('If faceting, there can only be one behavpy object')
            exit()
        d_list = []
        arg_list = list(set(df_list[0].meta[facet_col].tolist()))
        for arg in arg_list:
            temp_df = df_list[0].xmv(facet_col, arg)
            d_list.append(temp_df)
        if labels is None:
            labels = arg_list
        elif len(d_list) != len(labels):
            labels = arg_list
    else:
        d_list = df_list

    if hmm_compare is True:
        if len(hmm_list) != len(d_list) or len(bin_list) != len(d_list):
            warnings.warn('There are not enough hmm models or bin ints for the different groups or vice versa')
            exit()

    list_states = list(range(4))

    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Fraction of time asleep',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            range = [0, 1.01],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18,
                color = 'black'
            ),
            linewidth = 4
        ),
        plot_bgcolor = 'white',
    )

    fig = go.Figure(layout = layout)

    for c, phase in enumerate(['light', 'dark']):

        phase_list = [d[d['phase'] == phase] for d in d_list]
        phase_list = [d.drop(['phase', 'day'], axis = 1) for d in phase_list]

        def decode(data, bin, hmm):
            # change the movement column of choice to intergers, 1 == active, 0 == inactive
            data['moving'] = np.where(data['moving'] == True, 1, 0)
            # bin the data to 60 second intervals with a selected column and function on that column
            bin_df = data.bin_time('moving', bin, function= 'max')

            gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list)
            gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

            logprob_list = []
            states_list = []

            for i in range(len(gb)):
                seq = np.array(gb[i])
                seq = seq.reshape(-1, 1)

                logprob, states = hmm.decode(seq)

                logprob_list.append(logprob)
                states_list.append(states)

            return states_list, gb2

        if hmm_compare is True:
            if len(d_list) == 1:
                decoded_dict = {f'df{c}' : decode(phase_list[0], bin = b, hmm = h) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
            else:
                decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h) for c, (d, h, b) in enumerate(zip(phase_list, hmm_list, bin_list))}
            ite_len = len(hmm_list)
        else:
            decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0]) for c, d in enumerate(phase_list)}
            ite_len = len(phase_list)

        def analysis(array_states):

            counts_all = pd.DataFrame()

            for c, i in enumerate(array_states):
                count_df = pd.DataFrame(data = {0 : (i == 0).sum(), 1 : (i == 1).sum(), 2 : (i == 2).sum(), 3 : (i == 3).sum()}, index = [c])
                counts_all = counts_all.append(count_df)
                
            counts_all['sum'] = counts_all.sum(axis=1)
            counts_all = counts_all.iloc[:, list_states[0]: list_states[-1]+1].div(counts_all['sum'], axis=0)
            counts_all['total_sleep'] = counts_all[0] + counts_all[1]
            counts_all['total_awake'] = counts_all[2] + counts_all[3]
            counts_all.drop([0, 1, 2, 3], axis = 1, inplace = True)

            return counts_all

        analysed_dict = {f'df{c}' : analysis(v[0]) for c, v in enumerate(decoded_dict.values())}
        
        for index, v in enumerate(analysed_dict.values()):
            v['labels'] = labels[index]

        median_list = [analysed_dict[f'df{i}']['total_sleep'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3].mean() for i in range(len(labels))]
        q3_list = [bootstrap(analysed_dict[f'df{i}']['total_sleep'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3])[1] for i in range(len(labels))]
        q1_list = [bootstrap(analysed_dict[f'df{i}']['total_sleep'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3])[0] for i in range(len(labels))]

        if phase == 'light':
            col = 'goldenrod'
        else:
            col = 'black'

        trace_box = go.Box(
            showlegend = False,
            median = median_list, 
            q3 = q3_list, 
            q1 = q1_list,
            x = labels,
            xaxis = f'x{c+1}',
            marker = dict(
                color = col,
            ),
            boxpoints = False,
            jitter = 0.75, 
            pointpos = 0, 
            width = 0.9,
        )
        fig.add_trace(trace_box)

        con_list = pd.Series(dtype = 'float64')
        label_list = pd.Series(dtype = 'str')
        for i in range(ite_len):
            con_list = pd.concat([con_list, analysed_dict[f'df{i}']['total_sleep'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3]])
            label_list = pd.concat([label_list, analysed_dict[f'df{i}']['labels'][np.abs(zscore(analysed_dict[f'df{i}']['total_sleep'])) < 3]])

        trace_box2 = go.Box(
            showlegend = False,
            y = con_list, 
            x = label_list,
            xaxis = f'x{c+1}',
            line = dict(
                color = 'rgba(0,0,0,0)'
            ),
            fillcolor = 'rgba(0,0,0,0)',
            marker = dict(
                color = col,
                opacity = 0.5,
                size = 4
            ),
            boxpoints = 'all',
            jitter = 0.75, 
            pointpos = 0, 
            width = 0.9
        )
        fig.add_trace(trace_box2)

        domains = np.arange(0, 2, 1/2)

        axis = f'xaxis{c+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[c:c+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = phase,
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
    fig.show()

def ratio_plot(df_list, hmm_list, labels, colours, bin_list = [60], wrapped = False, asleep = True, hmm_compare = False, save = False, location = '', title = ''):
    
    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    for d in df_list:
        assert isinstance(d, behavpy)
    
    if hmm_compare is True:
        if len(hmm_list) != len(df_list) or len(bin_list) != len(df_list):
            warnings.warn('There are not enough hmm models or bin ints for the different groups or vice versa')
            exit()

    def decode(data, bin, hmm):
        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        data['moving'] = np.where(data['moving'] == True, 1, 0)
        # bin the data to 60 second intervals with a selected column and function on that column
        bin_df = data.bin_time('moving', bin, function= 'max')

        gb = bin_df.groupby(bin_df.index)['moving_max'].apply(list).reset_index(name='moving')
        gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list).reset_index(name='time')
        
        gb_merge = pd.merge(gb, gb2, on = 'id')

        mov = gb_merge['moving']
        time = gb_merge['time']
        id = gb_merge['id']

        temp_df = pd.DataFrame()

        for i in range(len(mov)):
            t = np.array(time[i])
            seq = np.array(mov[i])
            seq = seq.reshape(-1, 1)
            _, states = hmm.decode(seq)
            label = [id[i]] * len(t)
            all = zip(label, t, states)
            all = pd.DataFrame(data = all)
            temp_df = temp_df.append(all, ignore_index = False)

        temp_df.columns = ['id', 't', 'state']
        temp_df.set_index('id', inplace = True)
            
        return temp_df
    
    if hmm_compare is True:
        if len(df_list) == 1:
            decoded_dict = {f'df{c}' : decode(df_list[0], bin = b, hmm = h) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
        else:
            decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h) for c, (d, h, b) in enumerate(zip(df_list, hmm_list, bin_list))}
    else:
        decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0]) for c, d in enumerate(df_list)}

    y_min_global = []
    y_max_global = []

    def analysis(decoded_df, asleep = True):

        decoded_df['bin'] = decoded_df['t'].map(lambda t: 1800 * floor(t / 1800))
        if asleep is True:
            decoded_df['0'] = np.where(decoded_df['state'] == 0, 1, 0)
            decoded_df['1'] = np.where(decoded_df['state'] == 1, 1, 0)
        else:
            decoded_df['0'] = np.where(decoded_df['state'] == 3, 1, 0)
            decoded_df['1'] = np.where(decoded_df['state'] == 2, 1, 0)

        all = pd.DataFrame()
        for _, fly in decoded_df.groupby('id'):
            bin_s = fly.groupby('bin').agg(**{
                'sum_0' : ('0', 'sum'),
                'sum_1' : ('1', 'sum')   
            })
            bin_s['ratio_0'] = bin_s['sum_0'] / bin_s['sum_1'].replace(0, 1)
            bin_s['ratio_1'] = bin_s['sum_1'] / bin_s['sum_0'].replace(0, 1)
            
            bin_s['r_norm'] = np.where(bin_s['ratio_0'] >= 1, bin_s['ratio_0'], -bin_s['ratio_1'])

            bin_s.drop(['sum_0', 'sum_1', 'ratio_0', 'ratio_1'], axis = 1, inplace = True)
            all = all.append(bin_s)
        
        if wrapped is True:
            all.index = all.index.map(lambda t: t % 86400)

        all.index = all.index.map(lambda t: t / (60*60))

        def pop_std(array):
            return np.std(array, ddof = 0)

        all_gb = all.groupby(all.index).agg(**{
                        'mean' : ('r_norm', 'mean'), 
                        'SD' : ('r_norm', pop_std),
                        'count' : ('r_norm', 'count')
                    })
        all_gb['SE'] = (1.96*all_gb['SD']) / np.sqrt(all_gb['count'])
        all_gb['error+'] = all_gb['mean'] + all_gb['SE']
        all_gb['error-'] = all_gb['mean'] - all_gb['SE']

        global t_min
        global t_max 

        t_min = int(all_gb.index.min())
        t_max = int(all_gb.index.max() + 0.5)

        y_max_global.append(int(ceil(all_gb['error+'].max())))
        y_min_global.append(int(floor(all_gb['error-'].min())))

        return all_gb

    analysed_dict = {f'df{c}' : analysis(v, asleep = asleep) for c, v in enumerate(decoded_dict.values())}

    if asleep is True:
        y_title = 'Fold change of Deep Sleep to Light Asleep'
    else:
        y_title = 'Fold change of Active Awake to Quite Awake'

    layout = go.Layout(
        title = title,
            yaxis= dict(
                showgrid = False,
                linecolor = 'black',
                title = dict(
                    text = y_title,
                    font = dict(
                        size = 20,
                        color = 'black'
                    )
                ),
                range = [np.array(y_min_global).min() - 1, np.array(y_max_global).max()],
                dtick = 5,
                zeroline = False,
                        ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4,
            ), 
            xaxis = dict(
                showgrid = False,
                zeroline = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = 'ZT Hours',
                    font = dict(
                        size = 20,
                        color = 'black'
                    )
                ),
                range = [t_min - 0.1, t_max],
                tick0 = 0,
                dtick = 6,
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                ),
                linewidth = 3
            ),
        plot_bgcolor = 'white',
        )

    fig = go.Figure(layout = layout)
    for c, (col, lab) in enumerate(zip(colours, labels)):
        print(c)
        
        plot_df = analysed_dict[f'df{c}']
        
        trace_1 = go.Scatter(
            showlegend = True,
            name = lab,
            y = plot_df['mean'],
            x = plot_df.index,
            mode = 'lines+markers',
            marker = dict(
                color = col,
                line = dict(
                    color = col,
                    width = 4
                )
            ),
            error_y = dict(
                type = 'data',
                symmetric = True,
                array = plot_df['SE'],
                color = 'black',
                thickness = 1,
                width = 10
            )
        )

        fig.add_trace(trace_1)

    # Light-Dark annotaion bars
    bar_shapes = {}

    for i, bars in enumerate(range(t_min, t_max, 12)):
        if bars % 24 == 0:
            bar_col = 'white'
        else:
            bar_col = 'black'

        bar_shapes['shape_' + str(i)] = go.layout.Shape(type='rect', 
                                                        x0=bars, 
                                                        y0=np.array(y_min_global).min() - 1, 
                                                        x1=bars+12, 
                                                        y1=np.array(y_min_global).min() - 0.5, 
                                                        line=dict(
                                                            color='black', 
                                                            width=1),
                                                        fillcolor=bar_col
                                                    )
    fig.update_layout(shapes=list(bar_shapes.values()))
    
    fig.add_shape(type='line',
                    x0= t_min - 0.1,
                    y0= 0,
                    x1= t_max,
                    y1= 0,
                    line=dict(color='black',),
                    xref='x',
                    yref='y'
    )

    if save is True:
        fig.write_image(location, width=1000, height=650)
        print(f'Saved to {location}')
    else:
        fig.show()

def hmm_state_movement_comp(df_list, hmm_list, bin_list, labels, state, curate = False, hmm_compare = False, save = False, location = '', title = ''):
    """ compares the distribution of movement types when given a specific state from a decoded HMM model - 4 states only """
    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    for d in df_list:
        assert isinstance(d, behavpy)

    def decode(data, bin, hmm, curate):

        data['moving'] = np.where(data['moving'] == True, 1, 0)
        data['inactive'] = np.where(data['moving'] == True, 0, 1)
        data['micro'] = np.where(data['micro'] == True, 1, 0)
        data['walk'] = np.where(data['walk'] == True, 1, 0)
        
        data['t'] = data['t'].map(lambda t: bin * floor(t / bin))
        bin_df = data.groupby([data.index, 't']).agg(**{
            'moving' : ('moving', 'max'),
            'inactive' : ('inactive', 'max'), 
            'micro' : ('micro', 'max'), 
            'walk' : ('walk', 'max'),     
        })
        bin_df.reset_index(level = 1, inplace = True)

        gb_mov = bin_df.groupby(bin_df.index)['moving'].apply(list)
        gb_inactive = bin_df.groupby(bin_df.index)['inactive'].apply(list)
        gb_micro = bin_df.groupby(bin_df.index)['micro'].apply(list)
        gb_walk = bin_df.groupby(bin_df.index)['walk'].apply(list)

        if curate is True:
            curated_gb_mov = []
            curated_gb_inactive = []
            curated_gb_micro = []
            curated_gb_walk = []
            for i, q, w, e in zip(gb_mov, gb_inactive, gb_micro, gb_walk):
                if len(i) >= 86400/bin and len(q) >= 86400/bin and len(w) >= 86400/bin and len(e) >= 86400/bin:
                    curated_gb_mov.append(i)
                    curated_gb_inactive.append(q)
                    curated_gb_micro.append(w)
                    curated_gb_walk.append(e)

            gb_mov = curated_gb_mov
            gb_inactive = curated_gb_inactive
            gb_micro = curated_gb_micro
            gb_walk = curated_gb_walk

        logprob_list = []
        states_list = []

        for i in range(len(gb_mov)):
            seq = np.array(gb_mov[i])
            seq = seq.reshape(-1, 1)

            logprob, states = hmm.decode(seq)

            logprob_list.append(logprob)
            states_list.append(states)

        return states_list, gb_inactive, gb_micro, gb_walk

    if hmm_compare is True:
        if len(df_list) == 1:
            decoded_dict = {f'df{c}' : decode(df_list[0], bin = b, hmm = h, curate = curate) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
        else:
            decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h, curate = curate) for c, (d, h, b) in enumerate(zip(df_list, hmm_list, bin_list))}
        ite_len = len(hmm_list)
    else:
        decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0], curate = curate) for c, d in enumerate(df_list)}
        ite_len = len(df_list)

    def analysis(movement_arrays):

        df_all = pd.DataFrame()

        for c, (i) in enumerate(zip(movement_arrays[0], movement_arrays[1], movement_arrays[2], movement_arrays[3])):
            df_small = pd.DataFrame(data = {'state' : i[0], 'Inactive' : i[1], 'Micro-Movement' : i[2], 'Walking' : i[3]}, index = [c] * len(i[0]))
            df_small = df_small[df_small['state'] ==  state]
            sum_df = pd.DataFrame(df_small.sum(axis = 0)).T
            sum_df.drop(['state'], axis = 1, inplace = True)
            sum_df['sum'] = sum_df.sum(axis=1)
            sum_df = sum_df.iloc[:, 0:3].div(sum_df['sum'], axis=0)
            sum_df.fillna(0, inplace = True)
            df_all = df_all.append(sum_df)

        return df_all

    analysed_dict = {f'df{c}' : analysis(v) for c, v in enumerate(decoded_dict.values())}

    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Fraction of time in movement group',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            range = [0, 1.01],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18,
                color = 'black'
            ),
            linewidth = 4
        ),
        plot_bgcolor = 'white',
    )

    fig = go.Figure(layout = layout)

    treatment = analysed_dict['df0'].columns.tolist()

    for c, v in enumerate(analysed_dict.values()):
        v['labels'] = labels[c]

    if state == 0:
        col = 'blue'
    if state == 1:
        col = 'green'
    if state == 2:
        col = 'orange'
    if state == 3:
        col = 'red'

    for c, treat in enumerate(treatment):

        median_list = [analysed_dict[f'df{i}'][treat][np.abs(zscore(analysed_dict[f'df{i}'][treat])) < 3].mean() for i in range(ite_len)]
        q3_list = [bootstrap(analysed_dict[f'df{i}'][treat][np.abs(zscore(analysed_dict[f'df{i}'][treat])) < 3])[1] for i in range(ite_len)]
        q1_list = [bootstrap(analysed_dict[f'df{i}'][treat][np.abs(zscore(analysed_dict[f'df{i}'][treat])) < 3])[0] for i in range(ite_len)]

        trace_box = go.Box(
            showlegend = False,
            median = median_list, 
            q3 = q3_list, 
            q1 = q1_list,
            x = labels,
            xaxis = f'x{c+1}',
            marker = dict(
                color = col,
                opacity = 0.5,
                size = 4
            ),
            boxpoints = False,
            jitter = 0.75, 
            pointpos = 0, 
            width = 0.9,
        )
        fig.add_trace(trace_box)

        con_list = pd.Series(dtype = 'float64')
        label_list = pd.Series(dtype = 'str')
        for i in range(ite_len):
            con_list = pd.concat([con_list, analysed_dict[f'df{i}'][treat][np.abs(zscore(analysed_dict[f'df{i}'][treat])) < 3]])
            label_list = pd.concat([label_list, analysed_dict[f'df{i}']['labels'][np.abs(zscore(analysed_dict[f'df{i}'][treat])) < 3]])

        trace_box2 = go.Box(
            showlegend = False,
            y = con_list, 
            x = label_list,
            xaxis = f'x{c+1}',
            line = dict(
                color = 'rgba(0,0,0,0)'
            ),
            fillcolor = 'rgba(0,0,0,0)',
            marker = dict(
                color = col,
                opacity = 0.5,
                size = 4
            ),
            boxpoints = 'all',
            jitter = 0.75, 
            pointpos = 0, 
            width = 0.9
        )
        fig.add_trace(trace_box2)

        domains = np.arange(0, 1+(1/len(treatment)), 1/len(treatment))

        axis = f'xaxis{c+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[c:c+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = treat,
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def mago_response(df, mago_df, hmm, colours, labels, bin = 60, xmv_column = None, xmv_args = None, save = False, location = '', title = ''):
    """ plot function to measure the response rate of flies to a puff of odour from a mAGO or AGO experiment
        df = behavpy dataframe with fly movement for decoding by hmm
        mago_df = behapy dataframe intially analysed by the puff_mago loading function
        xmv_column = the column you want to compare data by 
        xmv_args= any filterings of the dataframe to compare - args will search metadata and return filtered dataframe"""

    assert isinstance(df, behavpy)
    assert isinstance(mago_df, behavpy)
    assert isinstance(labels, list)
    assert isinstance(colours, list)

    if any([xmv_column, xmv_args]):
        if all([xmv_column, xmv_args]) is False:
            warnings.warn('One of xmv_args/column have arguments but not both, please have both or none')
            exit()
        else:
            assert isinstance(xmv_args, list)

    def decode(col = None, arg = None, df = df, mago_df = mago_df):

        if all([col, arg]):
            d = df.xmv(col, arg)
            data = mago_df.xmv(col, arg)
        else:
            d = df
            data = mago_df

        d['t'] = d['t'].map(lambda t: bin * floor(t / bin))
        d['moving'] = np.where(d['moving'] == True, 1, 0)

        bin_gb = d.groupby(['id','t']).agg(**{
                    'moving' : ('moving', 'max')
            })
        bin_gb.reset_index(level = 1, inplace = True)

        gb_m = bin_gb.groupby(bin_gb.index)['moving'].apply(list).reset_index(name='moving')
        gb_t = bin_gb.groupby(bin_gb.index)['t'].apply(list).reset_index(name='time')

        gb_merge = pd.merge(gb_m, gb_t, on = 'id')

        def decode_array(dataframe):
            mov = dataframe['moving']
            time = dataframe['time']
            id = dataframe['id']

            temp_df = pd.DataFrame()

            for i in range(len(mov)):
                t = np.array(time[i])
                seq_orig = np.array(mov[i])
                seq = seq_orig.reshape(-1, 1)
                _, states = hmm.decode(seq)
                label = [id[i]] * len(t)
                previous_state = np.array(states[:-1], dtype = float)
                previous_state = np.insert(previous_state, 0, np.nan)
                all = zip(label, t, states, previous_state, seq_orig)
                all = pd.DataFrame(data = all)
                temp_df = pd.concat([temp_df, all], ignore_index = False)

            temp_df.columns = ['id', 'bin', 'state', 'previous_state', 'moving']
            temp_df.set_index('id', inplace = True)

            return temp_df

        s = decode_array(gb_merge)  
        data['bin'] = data['interaction_t'].map(lambda t:  60 * floor(t / 60))
        data.reset_index(inplace = True)

        merged = pd.merge(s, data, how = 'inner', on = ['id', 'bin'])
        merged['t_check'] = merged.interaction_t + merged.t_rel
        merged['t_check'] = merged['t_check'].map(lambda t:  60 * floor(t / 60))

        merged['previous_state'] = np.where(merged['t_check'] > merged['bin'], merged['state'], merged['previous_state'])

        interaction_dict = {}
        for i in list(set(merged.has_interacted)):
            filt_merged = merged[merged['has_interacted'] == i]
            big_gb = filt_merged.groupby(['id', 'previous_state']).agg(**{
                        'prop_respond' : ('has_responded', 'mean')
                })
            interaction_dict[f'int_{int(i)}'] = big_gb.groupby('previous_state')['prop_respond'].apply(np.array)

        return interaction_dict

    if all([xmv_column, xmv_args]) is True:
        response_dict = {f'list_{i}' : decode(col = xmv_column, arg = i) for i in xmv_args}
    else:
        response_dict = {'list_' : decode()}

    if xmv_column is None:
        xmv_args = ['']

    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Response rate',
                font = dict(
                    size = 20,
                    color = 'black'
                )
            ),
            range = [0, 1.05],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 4
        ),
        plot_bgcolor = 'white'    
    )
    ind = [1, 2] * len(xmv_args)
    xmv_args_temp = []
    xmv = []
    for i in xmv_args:
        xmv_args_temp.append(i)
        xmv.append(i)
        xmv.append(i)
        xmv_args_temp.append(f'{i} Spon. Mov.')
    xmv_args = xmv_args_temp

    fig = go.Figure(layout = layout)

    for state, col, lab in zip(range(0,4), colours, labels):

        median_list = [np.mean(np.array(response_dict[f'list_{i}'][f'int_{q}'][state])[np.abs(zscore(response_dict[f'list_{i}'][f'int_{q}'][state])) < 3]) for i, q in zip(xmv , ind)]
        q3_list = [bootstrap(np.array(response_dict[f'list_{i}'][f'int_{q}'][state])[np.abs(zscore(response_dict[f'list_{i}'][f'int_{q}'][state])) < 3])[1] for i, q in zip(xmv , ind)]
        q1_list = [bootstrap(np.array(response_dict[f'list_{i}'][f'int_{q}'][state])[np.abs(zscore(response_dict[f'list_{i}'][f'int_{q}'][state])) < 3])[0] for i, q in zip(xmv , ind)]

        for c, i in enumerate(xmv_args):

            if 'baseline' in i or 'control' in i or 'Spon. Mov.' in i:
                marker_col = 'grey'
            else:
                marker_col = col

            trace_box = go.Box(
                showlegend = False,
                median = [median_list[c]],
                q3 = [q3_list[c]],
                q1 = [q1_list[c]],
                x = [xmv_args[c]],
                xaxis = f'x{state+1}',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = False
            )
            fig.add_trace(trace_box)

            if (c+1) % 2 == 0:
                x = 2
            else:
                x = 1

            con_list = np.array(response_dict[f'list_{xmv[c]}'][f'int_{x}'][state])[np.abs(zscore(response_dict[f'list_{xmv[c]}'][f'int_{x}'][state])) < 3]
            label_list = len(np.array(response_dict[f'list_{xmv[c]}'][f'int_{x}'][state])[np.abs(zscore(response_dict[f'list_{xmv[c]}'][f'int_{x}'][state])) < 3]) * [i]

            trace_box2 = go.Box(
                showlegend = False,
                y = con_list,
                x = label_list,
                xaxis = f'x{state+1}',
                line = dict(
                    color = 'rgba(0,0,0,0)'
                ),
                fillcolor = 'rgba(0,0,0,0)',
                marker = dict(
                    color = marker_col,
                    opacity = 0.5,
                    size = 4
                ),
                boxpoints = 'all',
                jitter = 0.75,
                pointpos = 0,
                width = 0.9
            )
            fig.add_trace(trace_box2)

        domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

        axis = f'xaxis{state+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[state:state+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = lab,
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()

def hmm_state_change(df_list, hmm_list, bin_list, labels, state, curate = False, hmm_compare = False, save = False, location = '', title = ''):
    """ finds the percentage of each movement group that turns into the next state - 4 states only
    dataframes must include columns 'micro' and 'walk' as boolean entities 
    labels are the flanking states around state of interest either side of 'No change', e.g. ['Light sleep', 'No change', 'Active Awake']
    state is the number corresponding to your state of interest"""
    assert isinstance(df_list, list)
    assert isinstance(bin_list, list)
    assert isinstance(hmm_list, list)

    for d in df_list:
        assert isinstance(d, behavpy)

    def decode(data, bin, hmm, curate):

        data['moving'] = np.where(data['moving'] == True, 1, 0)
        data['inactive'] = np.where(data['moving'] == True, 0, 1)
        data['micro'] = np.where(data['micro'] == True, 1, 0)
        data['walk'] = np.where(data['walk'] == True, 1, 0)
        
        data['t'] = data['t'].map(lambda t: bin * floor(t / bin))
        bin_df = data.groupby([data.index, 't']).agg(**{
            'moving' : ('moving', 'max'),
            'inactive' : ('inactive', 'max'), 
            'micro' : ('micro', 'max'), 
            'walk' : ('walk', 'max'),     
        })
        bin_df.reset_index(level = 1, inplace = True)

        gb_mov = bin_df.groupby(bin_df.index)['moving'].apply(list)
        gb_inactive = bin_df.groupby(bin_df.index)['inactive'].apply(list)
        gb_micro = bin_df.groupby(bin_df.index)['micro'].apply(list)
        gb_walk = bin_df.groupby(bin_df.index)['walk'].apply(list)

        if curate is True:
            curated_gb_mov = []
            curated_gb_inactive = []
            curated_gb_micro = []
            curated_gb_walk = []
            for i, q, w, e in zip(gb_mov, gb_inactive, gb_micro, gb_walk):
                if len(i) >= 86400/bin and len(q) >= 86400/bin and len(w) >= 86400/bin and len(e) >= 86400/bin:
                    curated_gb_mov.append(i)
                    curated_gb_inactive.append(q)
                    curated_gb_micro.append(w)
                    curated_gb_walk.append(e)

            gb_mov = curated_gb_mov
            gb_inactive = curated_gb_inactive
            gb_micro = curated_gb_micro
            gb_walk = curated_gb_walk

        logprob_list = []
        states_list = []

        for i in range(len(gb_mov)):
            seq = np.array(gb_mov[i])
            seq = seq.reshape(-1, 1)

            logprob, states = hmm.decode(seq)

            logprob_list.append(logprob)
            states_list.append(states)

        return states_list, gb_inactive, gb_micro, gb_walk

    if hmm_compare is True:
        if len(df_list) == 1:
            decoded_dict = {f'df{c}' : decode(df_list[0], bin = b, hmm = h, curate = curate) for c, (h, b) in enumerate(zip(hmm_list, bin_list))}
        else:
            decoded_dict = {f'df{c}' : decode(data = d, bin = b, hmm = h, curate = curate) for c, (d, h, b) in enumerate(zip(df_list, hmm_list, bin_list))}
    else:
        decoded_dict = {f'df{c}' : decode(d, bin = bin_list[0], hmm = hmm_list[0], curate = curate) for c, d in enumerate(df_list)}

    def analysis(movement_arrays):
        small_dict = {}
        for mov in ['Inactive', 'Micro-Movement', 'Walking']:
            df_all = pd.DataFrame()
            for c, (i) in enumerate(zip(movement_arrays[0], movement_arrays[1], movement_arrays[2], movement_arrays[3])):
                df_small = pd.DataFrame(data = {'state' : i[0], 'Inactive' : i[1], 'Micro-Movement' : i[2], 'Walking' : i[3]}, index = [c] * len(i[0]))
                next_state = np.array(i[0][1:], dtype = int)
                next_state = np.append(next_state, np.nan)
                df_small['next_state'] = next_state
                df_small = df_small[df_small['state'] ==  state]   
                df_slice = df_small[[mov, 'next_state']]
                df_slice = df_slice[df_slice[mov] == 1]
                if len(df_slice) < 1:
                    continue
                else:
                    sum_df = df_slice['next_state'].value_counts().to_frame().T
                    sum_df['sum'] = sum_df.sum(axis=1)
                    sum_df = sum_df.iloc[:, 0:3].div(sum_df['sum'], axis=0)
                    sum_df.fillna(0, inplace = True)
                    df_all = df_all.append(sum_df)
            if len(df_all) < 1:
                continue
            else:
                if 'sum' in df_all.columns.tolist():
                    df_all.drop('sum', axis = 1, inplace = True)
                df_all['index'] = mov
                df_all.set_index('index', inplace = True)
                small_col  = np.sort(df_all.columns.to_numpy())
                df_all = df_all[small_col]
                small_dict[mov] = df_all
        return small_dict


    analysed = analysis(decoded_dict['df0'])
    
    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Chance of changing to state',
                font = dict(
                    size = 18,
                    color = 'black'
                )
            ),
            range = [0, 1.01],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18,
                color = 'black'
            ),
            linewidth = 4
        ),
        plot_bgcolor = 'white',
    )

    fig = go.Figure(layout = layout)

    treatment = analysed.keys()

    if state == 0:
        col = 'blue'
    if state == 1:
        col = 'green'
    if state == 2:
        col = 'orange'
    if state == 3:
        col = 'red'

    for c, treat in enumerate(treatment):
        median_list = [analysed[treat][i].mean() for i in analysed[treat].columns.tolist()]
        q3_list = [bootstrap(analysed[treat][i])[1] for i in analysed[treat].columns.tolist()]
        q1_list = [bootstrap(analysed[treat][i])[0] for i in analysed[treat].columns.tolist()]

        trace_box = go.Box(
            showlegend = False,
            median = median_list, 
            q3 = q3_list, 
            q1 = q1_list,
            x = labels,
            xaxis = f'x{c+1}',
            marker = dict(
                color = col,
                opacity = 0.5,
                size = 4
            ),
            boxpoints = False,
            jitter = 0.75, 
            pointpos = 0, 
            width = 0.9,
        )
        fig.add_trace(trace_box)

        con_list = pd.Series(dtype = 'float64')
        label_list = []
        for i in analysed[treat].columns.tolist():
            con_list = pd.concat([con_list, analysed[treat][i]])
        for i in range(len(labels)):
            label_list = label_list + len(analysed[treat].index) * [labels[i]]

        trace_box2 = go.Box(
            showlegend = False,
            y = con_list, 
            x = label_list,
            xaxis = f'x{c+1}',
            line = dict(
                color = 'rgba(0,0,0,0)'
            ),
            fillcolor = 'rgba(0,0,0,0)',
            marker = dict(
                color = col,
                opacity = 0.5,
                size = 4
            ),
            boxpoints = 'all',
            jitter = 0.75, 
            pointpos = 0, 
            width = 0.9
        )
        fig.add_trace(trace_box2)

        domains = np.arange(0, 1+(1/len(treatment)), 1/len(treatment))

        axis = f'xaxis{c+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[c:c+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = treat,
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
    else:
        fig.show()

def mago_response_per_min(df, mago_df, hmm, state, bin = 60, save = False, location = '', title = ''):
    """ 
    Plots the response rate per minute for individual states
    """

    assert isinstance(df, behavpy)
    assert isinstance(mago_df, behavpy)

    def decode(state, df = df, mago_df = mago_df):

        d = df
        data = mago_df

        d['t'] = d['t'].map(lambda t: bin * floor(t / bin))
        d['moving'] = np.where(d['moving'] == True, 1, 0)

        bin_gb = d.groupby(['id','t']).agg(**{
                    'moving' : ('moving', 'max')
            })
        bin_gb.reset_index(level = 1, inplace = True)

        gb_m = bin_gb.groupby(bin_gb.index)['moving'].apply(list).reset_index(name='moving')
        gb_t = bin_gb.groupby(bin_gb.index)['t'].apply(list).reset_index(name='time')

        gb_merge = pd.merge(gb_m, gb_t, on = 'id')

        def find_index(data):
            index_list = []
            for c, q in enumerate(data):
                if c == 0:
                    temp_list = []
                    temp_list.append(q)
                else:
                    if q == temp_list[-1]:
                        temp_list.append(q)
                    else:
                        temp_index = list(range(1, len(temp_list) + 1))
                        index_list.extend(temp_index)
                        temp_list = []
                        temp_list.append(q)
            return index_list


        def decode_array(dataframe):
            mov = dataframe['moving']
            time = dataframe['time']
            id = dataframe['id']

            temp_df = pd.DataFrame()

            for i in range(len(mov)):
                t = np.array(time[i])
                seq = np.array(mov[i])
                seq = seq.reshape(-1, 1)
                _, states = hmm.decode(seq)
                label = [id[i]] * len(t)
                previous_state = np.array(states[:-1], dtype = float)
                previous_state = np.insert(previous_state, 0, np.nan)
                position = find_index(states)
                previous_position = np.array(position[:-1], dtype = float)
                previous_position = np.insert(previous_position, 0, np.nan)
                all = zip(label, t, states, previous_state, position, previous_position)
                all = pd.DataFrame(data = all)
                temp_df = temp_df.append(all, ignore_index = False)    

            temp_df.columns = ['id', 'bin', 'state', 'previous_state', 'position', 'previous_position']
            temp_df.set_index('id', inplace = True)

            return temp_df

        s = decode_array(gb_merge)  

        data['bin'] = data['interaction_t'].map(lambda t:  60 * floor(t / 60))
        data.reset_index(inplace = True)

        merged = pd.merge(s, data, how = 'inner', on = ['id', 'bin'])
        merged['t_check'] = merged.interaction_t + merged.t_rel
        merged['t_check'] = merged['t_check'].map(lambda t:  60 * floor(t / 60))
        #merged['position'] = np.where(merged['t_check'] > merged['bin'], merged['position'], merged['previous_position'])
        merged = merged[merged['position'] <= 60]
        
        merged_filtered = merged[merged['previous_state'] == state]

        q_mins = list(set(merged_filtered['position'].astype(int)))

        big_gb = merged_filtered.groupby(['id', 'position']).agg(**{
                                'prop_respond' : ('has_responded', 'mean')
        })

        return np.array(big_gb.groupby('position')['prop_respond'].apply(list)), q_mins


    response_dict = {'list_' : decode(state = state)}
    
    queried_mins = response_dict['list_'][-1]

    layout = go.Layout(
        title = title,
        yaxis= dict(
            showgrid = False,
            linecolor = 'black',
            title = dict(
                text = 'Response rate',
                font = dict(
                    size = 20,
                    color = 'black'
                )
            ),
            range = [0, 1.05],
            tick0 = 0,
            dtick = 0.2,
            zeroline = False,
                    ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            ),
            linewidth = 4
        ),    
        plot_bgcolor = 'white',
    )
    fig = go.Figure(layout = layout)

    # changing shades of colour from dark to light 0 = blue, 1 = green, = 2 = orange, 3 = red
    start_col = {0 : '#00264c', 1 : '#086901', 2 : '#8a4300', 3 : '#700900'}
    end_col = {0 : '#b2d8ff', 1 : '#8df086', 2 : '#eda866', 3 : '#ed776d'}
    N = len(queried_mins)
    colour_range = [x.hex for x in list(Color(start_col[state]).range_to(Color(end_col[state]), N))]

    for c, (run, min) in enumerate(zip(response_dict['list_'][0], queried_mins)):

        median_list = [np.mean(np.array(run)[np.abs(zscore(run)) < 3])]
        q3_list = [bootstrap(np.array(run)[np.abs(zscore(run)) < 3])[1]]
        q1_list = [bootstrap(np.array(run)[np.abs(zscore(run)) < 3])[0]]

        trace_box = go.Box(
            showlegend = False,
            median = median_list,
            q3 = q3_list,
            q1 = q1_list,
            x = [''],
            xaxis = f'x{c+1}',
            marker = dict(
                color = colour_range[c],
                opacity = 0.5,
                size = 4
            ),
            boxpoints = False
        )
        fig.add_trace(trace_box)

        con_list = np.array(run)[np.abs(zscore(run)) < 3]
        label_list = len(np.array(run)[np.abs(zscore(run)) < 3]) * ['']

        trace_box2 = go.Box(
            showlegend = False,
            y = con_list,
            x = label_list,
            xaxis = f'x{c+1}',
            line = dict(
                color = 'rgba(0,0,0,0)'
            ),
            fillcolor = 'rgba(0,0,0,0)',
            marker = dict(
                color = colour_range[c],
                opacity = 0.5,
                size = 4
            ),
            boxpoints = 'all',
            jitter = 0.75,
            pointpos = 0,
            width = 0.9
        )
        fig.add_trace(trace_box2)

        domains = np.arange(0, 1+(1/len(queried_mins)), 1/len(queried_mins))

        axis = f'xaxis{c+1}'
        fig['layout'][axis] = {}

        fig['layout'][axis].update(
                domain = domains[c:c+2],
                showgrid = False,
                color = 'black',
                linecolor = 'black',
                title = dict(
                    text = str(min),
                    font = dict(
                        size = 18,
                        color = 'black'
                    )
                ),
                ticks = 'outside',
                tickwidth = 2,
                tickfont = dict(
                    size = 18
                ),
                linewidth = 4
            )

    if save is True:
        fig.write_image(location, width=1500, height=650)
        print(f'Saved to {location}')
        fig.show()
    else:
        fig.show()
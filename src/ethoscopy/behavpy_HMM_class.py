import pandas as pd
import numpy as np 
import warnings
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
import pickle

from tabulate import tabulate
from hmmlearn import hmm
from math import floor, ceil
from sys import exit
from colour import Color

from ethoscopy.behavpy_class import behavpy
from ethoscopy.analyse import max_velocity_detector, sleep_annotation
from ethoscopy.misc.rle import rle
from ethoscopy.misc.hmm_functions import hmm_pct_transition, hmm_mean_length, hmm_pct_state
from ethoscopy.misc.bootstrap_CI import bootstrap
from ethoscopy.misc.circadian_bars import circadian_bars

class behavpy_HMM(behavpy):
    """
    The behavpy_HMM class is a subclassed version of behavpy, itself a subclass of pandas. See behavpy doc strings for more information on behavpy.

    Behavpy_HMM has been augmented to include methods that generate trained Hidden Markov Models and 

    """

    def __init__(self, data, meta, check = False, index= None, columns=None, dtype=None, copy=True):
        super(behavpy, self).__init__(data=data,
                                        index=index,
                                        columns=columns,
                                        dtype=dtype,
                                        copy=copy)
        
        self.meta = meta 
        
        if check is True:
            self._check_conform(self)

    _colours_four = ['darkblue', 'dodgerblue', 'red', 'darkred']
    _hmm_labels = ['Deep_sleep', 'Light_sleep', 'Light_awake', 'Full_awake']

    def hmm_train(self, states, observables, var_column, trans_probs = None, emiss_probs = None, start_probs = None, iterations = 10, hmm_iterations = 100, tol = 50, t_column = 't', bin_time = 60, test_size = 10, file_name = '', verbose = False):
        """
        Behavpy wrapper for the hmmlearn package which generates a Hidden Markov Model using the movement data from ethoscope data.
        If users want a restricted framework ,
        E.g. for random:

        There must be no NaNs in the training data

        Resultant hidden markov models will be saved as a .pkl file if file_name is provided
        Final trained model probability matrices will be printed to terminal at the end of the run time

        Params:
        @states = list of sting(s), names of hidden states for the model to train to
        @observables = list of string(s), names of the observable states for the model to train to.
        The length must be the same number as the different categories in you movement column.
        @trans_probs = numpy array, transtion probability matrix with shape 'len(states) x len(states)', 0's restrict the model from training any tranisitons between those states
        @emiss_probs = numpy array, emission probability matrix with shape 'len(observables) x len(observables)', 0's same as above
        @start_probs = numpy array, starting probability matrix with shape 'len(states) x 0', 0's same as above
        @var_column = string, name for the column containing the variable of choice to train the model
        @iterations = int, only used if random is True, number of loops using a different randomised starting matrices, default is 10
        @hmm_iterations = int, argument to be passed to hmmlearn, number of iterations of parameter updating without reaching tol before it stops, default is 100
        @tol = int, convergence threshold, EM will stop if the gain in log-likelihood is below this value, default is 50
        @t_column = string, name for the column containing the time series data, default is 't'
        @bin_time = int, the time in seconds the data will be binned to before the training begins, default is 60 (i.e 1 min)
        @file_name = string, name of the .pkl file the resultant trained model will be saved to, if left as '' and random is False the model won't be saved, default is ''
        @verbose = (bool, optional), argument for hmmlearn, whether per-iteration convergence reports are printed to terminal

        returns a trained hmmlearn HMM Multinomial object
        """
        
        if file_name.endswith('.pkl') is False:
            warnings.warn('enter a file name and type (.pkl) for the hmm object to be saved under')
            exit()

        n_states = len(states)
        n_obs = len(observables)

        def bin_to_list(data, t_var, mov_var, bin):
            """ 
            Bins the time to the given integer and creates a nested list of the movement column by id
            """
            if mov_var == 'moving':
                stat = 'max'
            else:
                stat = 'mean'

            t_delta = data[t_column].iloc[1] - data[t_column].iloc[0]

            if t_delta != bin:
                data[t_var] = data[t_var].map(lambda t: bin * floor(t / bin))
                bin_gb = self.groupby(['id', t_var]).agg(**{
                    mov_var : (var_column, stat)
                })
                bin_gb.reset_index(level = 1, inplace = True)
                gb = np.array(bin_gb.groupby('id')[mov_var].apply(list).tolist(), dtype = 'object')

            else:
                gb = np.array(self.groupby('id')[mov_var].apply(list).tolist(), dtype = 'object')

            return gb

        def hmm_table(start_prob, trans_prob, emission_prob, state_names, observable_names):
            """ 
            Prints a formatted table of the probabilities from a hmmlearn MultinomialHMM object
            """
            df_s = pd.DataFrame(start_prob)
            df_s = df_s.T
            df_s.columns = states
            print("Starting probabilty table: ")
            print(tabulate(df_s, headers = 'keys', tablefmt = "github") + "\n")
            print("Transition probabilty table: ")
            df_t = pd.DataFrame(trans_prob, index = state_names, columns = state_names)
            print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
            print("Emission probabilty table: ")
            df_e = pd.DataFrame(emission_prob, index = state_names, columns = observable_names)
            print(tabulate(df_e, headers = 'keys', tablefmt = "github") + "\n")

        if var_column == 'beam_crosses':
            self['active'] = np.where(self[var_column] == 0, 0, 1)
            gb = bin_to_list(self, t_var = t_column, mov_var = var_column, bin = bin_time)

        elif var_column == 'moving':
            self[var_column] = np.where(self[var_column] == True, 1, 0)
            gb = bin_to_list(self, t_var = t_column, mov_var = var_column, bin = bin_time)

        # split runs into test and train lists
        test_train_split = round(len(gb) / test_size)
        rand_runs = np.random.permutation(gb)
        train = rand_runs[test_train_split:]
        test = rand_runs[:test_train_split]

        len_seq_train = []
        len_seq_test = []
        for i, q in zip(train, test):
            len_seq_train.append(len(i))
            len_seq_test.append(len(q))

        seq_train = np.concatenate(train, 0)
        seq_train = seq_train.reshape(-1, 1)
        seq_test = np.concatenate(test, 0)
        seq_test = seq_test.reshape(-1, 1)

        for i in range(iterations):
            print(f"Iteration {i+1} of {iterations}")
            
            init_params = ''
            h = hmm.MultinomialHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', verbose = verbose)

            if start_probs is None:
                init_params += 's'
            else:
                s_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in start_probs], dtype = np.float64)
                s_prob = np.array([[y / sum(x) for y in x] for x in t_prob], dtype = np.float64)
                h.startprob_ = s_prob

            if trans_probs is None:
                init_params += 't'
            else:
                # replace 'rand' with a new random number being 0-1
                t_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in trans_probs], dtype = np.float64)
                t_prob = np.array([[y / sum(x) for y in x] for x in t_prob], dtype = np.float64)
                h.transmat_ = t_prob

            if emiss_probs is None:
                init_params += 'e'
            else:
                # replace 'rand' with a new random number being 0-1
                em_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in emiss_probs], dtype = np.float64)
                em_prob = np.array([[y / sum(x) for y in x] for x in em_prob], dtype = np.float64)
                h.emissionprob_ = em_prob

            h.init_params = init_params

            h.n_features = n_obs # number of emission states

            # call the fit function on the dataset input
            h.fit(seq_train, len_seq_train)

            # Boolean output of if the number of runs convererged on set of appropriate probabilites for s, t, an e
            print("True Convergence:" + str(h.monitor_.history[-1] - h.monitor_.history[-2] < h.monitor_.tol))
            print("Final log liklihood score:" + str(h.score(seq_train, len_seq_train)))

            if i == 0:
                with open(file_name, "wb") as file: pickle.dump(h, file)

            else:
                with open(file_name, "rb") as file: 
                    h_old = pickle.load(file)
                if h.score(seq_test, len_seq_test) > h_old.score(seq_test, len_seq_test):
                    print('New Matrix:')
                    df_t = pd.DataFrame(h.transmat_, index = states, columns = states)
                    print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
                    with open(file_name, "wb") as file: pickle.dump(h, file)

            if i+1 == iterations:
                with open(file_name, "rb") as file: 
                    h = pickle.load(file)
                #print tables of trained emission probabilties, not accessible as objects for the user
                hmm_table(start_prob = h.startprob_, trans_prob = h.transmat_, emission_prob = h.emissionprob_, state_names = states, observable_names = observables)
                return h

    def bin_time(self, column, bin_secs, t_column = 't', function = 'mean'):
        """
        Bin the time series data into entered groups, pivot by the time series column and apply a function to the selected columns.

        This is the same as the behavpy method, but requires re-inilisation.
        
        Params:
        @column = string, column in the data that you want to the function to be applied to post pivot
        @bin_secs = float, the amount of time you want in each bin in seconds, e.g. 60 would be bins for every minutes
        @t_column = string, column same for the time series data that you want to group and pivot by
        @function = string or user defined function. The applied function to the grouped data, can be standard 'mean', 'max'.... ect, can also be a user defined function
        
        returns a behavpy object with a single data column
        """

        from math import floor

        if column not in self.columns:
            warnings.warn('Column heading "{}", is not in the data table'.format(column))
            exit()

        def wrapped_bin_data(data, column = column, bin_column = t_column, function = function, bin_secs = bin_secs):

            index_name = data['id'].iloc[0]

            data[bin_column] = data[bin_column].map(lambda t: bin_secs * floor(t / bin_secs))
            
            output_parse_name = f'{column}_{function}' # create new column name
        
            bout_gb = data.groupby(bin_column).agg(**{
                output_parse_name : (column, function)    
            })

            bin_parse_name = f'{bin_column}_bin'
            bout_gb.rename_axis(bin_parse_name, inplace = True)
            bout_gb.reset_index(level=0, inplace=True)
            old_index = pd.Index([index_name] * len(bout_gb.index), name = 'id')
            bout_gb.set_index(old_index, inplace =True)

            return bout_gb

        self.reset_index(inplace = True)
        return behavpy_HMM(self.groupby('id', group_keys = False).apply(wrapped_bin_data), self.meta)

    def plot_hmm_overtime(self, hmm, variable = 'moving', labels = None, colours = None, wrapped = False, bin = 60, func = 'max', avg_window = 30, circadian_night = 12, save = False, location = ''):
        """
        Creates a plot of all states overlayed with y-axis shows the liklihood of being in a sleep state and the x-axis showing time in hours.
        The plot is generated through the plotly package

        Params:
        @self = behavpy_HMM,
        @hmm = hmmlearn.hmm.MultinomialHMM, this should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
        @variable = string, the column heading of the variable of interest. Default is "moving"
        @labels = list[string], the names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep_sleep', 'Light_sleep', 'Light_awake', 'Full_awake']
        @colours = list[string], the name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red)
        It accepts a specific colour or an array of numbers that are acceptable to plotly
        @wrapped = bool, if True the plot will be limited to a 24 hour day average
        @bin = int, the time in seconds you want to bin the movement data to, default is 60 or 1 minute
        @func = string, when binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
        @avg_window, int, the window in minutes you want the moving average to be applied to. Default is 30 mins
        @circadian_night, int, the hour when lights are off during the experiment. Default is ZT 12
        @save = bool, if true the plot will be saved to local
        @location = string, only needed if save is True, provide the location and file type of the plot
        
        returns None
        """

        df = self.copy()

        if hmm.transmat_.shape[0] == 4 and labels == None:
            labels = self._hmm_labels
            colours = self._colours_four
        elif hmm.transmat_.shape[0] != 4:
            if colours is None or labels is None:
                warnings.warn('Your trained HMM is not 4 states, please provide the lables and colours for this hmm. See doc string,')
                exit()
            elif len(colours) != len(labels):
                warnings.warn('You have more or less states than colours, please rectify so they are equal in length')
                exit()

        list_states = list(range(len(labels)))

        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        # bin the data to 60 second intervals with a selected column and function on that column     
        if variable == 'moving' or variable == 'asleep':
            df[variable] = np.where(df[variable] == True, 1, 0)

        bin_df = df.bin_time(variable, bin, function = func)
        gb = bin_df.groupby(bin_df.index)[f'{variable}_{func}'].apply(list)
        gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

        def decode_array(nested_list):

            # logprob_list = []
            states_list = []

            for i in nested_list:
                seq = np.array(i)
                seq = seq.reshape(-1, 1)
                logprob, states = hmm.decode(seq)

                # logprob_list.append(logprob)
                states_list.append(states)
                
            return states_list #logprob_list

        states = decode_array(gb)

        df_list = pd.DataFrame()
        for l, t in zip(states, gb2):
            df = hmm_pct_state(l, t, list_states, avg_window = int((avg_window * 60)/bin))
            df_list = pd.concat([df_list, df], ignore_index = True)

        if wrapped is True:
            df_list['t'] = df_list['t'].map(lambda t: t % 86400)

        df_list['t'] = df_list['t'] / (60*60)
        t_min = int(12 * floor(df_list.t.min() / 12))
        t_max = int(12 * ceil(df_list.t.max() / 12))    
        t_range = [t_min, t_max]  

        stats_dict = {}

        for state in list_states:
            stats_dict['df' + str(state)] = df_list.groupby('t').agg(**{
                        'mean' : (f'state_{state}', 'mean'), 
                        'SD' : (f'state_{state}', self.pop_std),
                        'count' : (f'state_{state}', 'count')
                    })

        layout = go.Layout(
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
        bar_shapes = circadian_bars(t_min, t_max, circadian_night = circadian_night)
        fig.update_layout(shapes=list(bar_shapes.values()))

        if save is True:
            fig.write_image(location, width=1000, height=650)
            print(f'Saved to {location}')
        else:
            fig.show()

    def plot_hmm_split(self, hmm, variable = 'moving', labels = None, facet_col = None, facet_arg = None, hmm_compare = False, wrapped = False, bin = 60, func = 'max', avg_window = 30, circadian_night = 12, save = False, location = ''):
        """ Only works for 4 state models """

        if facet_col is not None:
            if facet_arg is None:
                facet_arg = list(set(self.meta[facet_col].tolist()))
                if labels is None:
                    labels = facet_arg
                elif len(facet_arg) != len(labels):
                    labels = facet_arg
            else:
                if labels is None:
                    labels = facet_arg
                elif len(facet_arg) != len(labels):
                    labels = facet_arg
        else:
            facet_arg = [None]
            if labels is None:
                labels = ['']

        if hmm_compare is True:
            assert isinstance(hmm, list)
            assert isinstance(bin, list)
            if len(hmm) != len(facet_arg) or len(bin) != len(facet_arg):
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
            h_list = [hmm]
            b_list = [bin]
            if len(h_list) != len(facet_arg):
                h_list = [h_list[0]] * len(facet_arg)
            if len(b_list) != len(facet_arg):
                b_list = [b_list[0]] * len(facet_arg)

        colour_range_dict = {}
        for q in range(0,4):
            colours_dict = {'start' : ['#b2d8ff', '#8df086', '#eda866', '#ed776d'], 'end' : ['#00264c', '#086901', '#8a4300', '#700900']}
            start_color = colours_dict.get('start')[q]
            end_color = colours_dict.get('end')[q]
            N = len(facet_arg)
            colour_range_dict[q] = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]

        for c, (arg, n, h, b) in enumerate(zip(facet_arg, labels, h_list, b_list)):   
            print(f'Decoding {n}...')

            if arg != None:
                d = self.xmv(facet_col, arg)
            else:
                d = self

            # change the movement column of choice to intergers, 1 == active, 0 == inactive
            if variable == 'moving' or variable == 'asleep':
                d[variable] = np.where(d[variable] == True, 1, 0)

            # bin the data to 60 second intervals with a selected column and function on that column
            bin_df = d.bin_time(variable, b, function = func)
            gb = bin_df.groupby(bin_df.index)[f'{variable}_{func}'].apply(list)
            gb2 = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

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
            print(f'Plotting {n}...')
            for l, t in zip(states, gb2):
                temp_df = hmm_pct_state(l, t, [0, 1, 2, 3], avg_window = int((avg_window * 60)/bin))
                analsyed_df = pd.concat([analsyed_df, temp_df], ignore_index = False)

            if wrapped is True:
                analsyed_df['t'] = analsyed_df['t'].map(lambda t: t % 86400)
            analsyed_df['t'] = analsyed_df['t'] / (60*60)

            if 'control' in n.lower() or 'baseline' in n.lower() or 'ctrl' in n.lower():
                black_list = ['black'] * len(facet_arg)
                black_range_dict = {0 : black_list, 1: black_list, 2 : black_list, 3 : black_list}
                marker_col = black_range_dict
            else:
                marker_col = colour_range_dict

            t_min = int(12 * floor(analsyed_df.t.min() / 12))
            t_max = int(12 * ceil(analsyed_df.t.max() / 12))    
            t_range = [t_min, t_max]  

            for i, row, col in zip(range(4), [1,1,2,2], [1,2,1,2]):

                loop_df = analsyed_df.groupby('t').agg(**{
                            'mean' : (f'state_{i}', 'mean'), 
                            'SD' : (f'state_{i}', self._pop_std),
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
        # Light-Dark annotaion bars
        bar_shapes = circadian_bars(t_min, t_max, circadian_night = circadian_night, split = True)
        fig.update_layout(shapes=list(bar_shapes.values()))

        fig.update_layout(shapes=list(bar_shapes.values()))

        if save is True:
            fig.write_image(location, width=1150, height=650)
            print(f'Saved to {location}')
            fig.show()
        else:
            fig.show()
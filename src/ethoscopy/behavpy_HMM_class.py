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
from scipy.stats import zscore
from functools import partial

from ethoscopy.behavpy_class import behavpy
from ethoscopy.misc.hmm_functions import hmm_pct_transition, hmm_mean_length, hmm_pct_state
# from ethoscopy.misc.bootstrap_CI import bootstrap
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
    _hmm_labels = ['Deep sleep', 'Light sleep', 'Quiet awake', 'Active awake']

    @staticmethod
    def _hmm_decode(d, h, b, var, fun, return_type = 'array'):

        # change the movement column of choice to intergers, 1 == active, 0 == inactive
        if var == 'moving' or var == 'asleep':
            d[var] = np.where(d[var] == True, 1, 0)

        # bin the data to 60 second intervals with a selected column and function on that column
        bin_df = d.bin_time(var, b, function = fun)
        gb = bin_df.groupby(bin_df.index)[f'{var}_{fun}'].apply(list)
        time_list = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

        # logprob_list = []
        states_list = []
        df = pd.DataFrame()

        for i, t, id in zip(gb, time_list, time_list.index):
            seq_o = np.array(i)
            seq = seq_o.reshape(-1, 1)
            logprob, states = h.decode(seq)

            #logprob_list.append(logprob)
            if return_type == 'array':
                states_list.append(states)
            if return_type == 'table':
                label = [id] * len(t)
                previous_state = np.array(states[:-1], dtype = float)
                previous_state = np.insert(previous_state, 0, np.nan)
                all = zip(label, t, states, previous_state, seq_o)
                all = pd.DataFrame(data = all)
                df = pd.concat([df, all], ignore_index = False)
        
        if return_type == 'array':
            return states_list, time_list #, logprob_list
        if return_type == 'table':
            df.columns = ['id', 'bin', 'state', 'previous_state', 'moving']
            return df


    # Internal methods for checking data/arguments before plotting
    def _check_hmm_shape(self, hm, lab, col):
        """
        Check the colours and labels passed to a plotting method are of equal length. If None then it will be populated with the defaults.
        """
        if isinstance(hm, list):
            hm = hm[0]

        if hm.transmat_.shape[0] == 4 and lab == None and col == None:
            _labels = self._hmm_labels
            _colours = self._colours_four
        elif hm.transmat_.shape[0] == 4 and lab == None and col != None:
            _labels = self._hmm_labels
            _colours = col
        elif hm.transmat_.shape[0] == 4 and lab != None and col == None:
            _labels = lab
            _colours = self._colours_four
        elif hm.transmat_.shape[0] != 4:
            if col is None or lab is None:
                warnings.warn('Your trained HMM is not 4 states, please provide the lables and colours for this hmm. See doc string for more info')
                exit()
            elif len(col) != len(lab):
                warnings.warn('You have more or less states than colours, please rectify so the lists are equal in length')
                exit()
        else:
            _labels = lab
            _colours = col

        if len(_labels) != len(_colours):
            warnings.warn('You have more or less states than colours, please rectify so they are equal in length')
            exit()
        
        return _labels, _colours

    def _check_lists_hmm(self, f_col, f_arg, f_lab, h, b):
        """
        Check if the facet arguments match the labels or populate from the column if not.
        Check if there is more than one HMM object for HMM comparison. Populate hmm and bin lists accordingly.
        """
        if isinstance(h, list):
            assert isinstance(b, list)
            if len(h) != len(f_arg) or len(b) != len(f_arg):
                warnings.warn('There are not enough hmm models or bin intergers for the different groups or vice versa')
                exit()
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
                            warnings.warn(f'Argument "{i}" is not in the meta column {f_col}')
                            exit()
                        string_args.append(str(i))
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    warnings.warn("The facet labels don't match the length of the variables in the column. Using column variables instead")
                    f_lab = f_arg
            else:
                if f_lab is None:
                    string_args = []
                    for i in f_arg:
                        string_args.append(str(i))
                    f_lab = string_args
                elif len(f_arg) != len(f_lab):
                    warnings.warn("The facet labels don't match the entered facet arguments in length. Using column variables instead")
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
    def _hmm_table(start_prob, trans_prob, emission_prob, state_names, observable_names):
        """ 
        Prints a formatted table of the probabilities from a hmmlearn MultinomialHMM object
        """
        df_s = pd.DataFrame(start_prob)
        df_s = df_s.T
        df_s.columns = state_names
        print("Starting probabilty table: ")
        print(tabulate(df_s, headers = 'keys', tablefmt = "github") + "\n")
        print("Transition probabilty table: ")
        df_t = pd.DataFrame(trans_prob, index = state_names, columns = state_names)
        print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
        print("Emission probabilty table: ")
        df_e = pd.DataFrame(emission_prob, index = state_names, columns = observable_names)
        print(tabulate(df_e, headers = 'keys', tablefmt = "github") + "\n")

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

        hmm_df = self.copy(deep = True)

        def bin_to_list(data, t_var, mov_var, bin):
            """ 
            Bins the time to the given integer and creates a nested list of the movement column by id
            """
            stat = 'max'
            data = data.reset_index()
            t_delta = data[t_column].iloc[1] - data[t_column].iloc[0]
            if t_delta != bin:
                data[t_var] = data[t_var].map(lambda t: bin * floor(t / bin))
                bin_gb = data.groupby(['id', t_var]).agg(**{
                    mov_var : (var_column, stat)
                })
                bin_gb.reset_index(level = 1, inplace = True)
                gb = bin_gb.groupby('id')[mov_var].apply(np.array)
            else:
                gb = data.groupby('id')[mov_var].apply(np.array)
            return gb

        if var_column == 'beam_crosses':
            hmm_df['active'] = np.where(hmm_df[var_column] == 0, 0, 1)
            gb = bin_to_list(hmm_df, t_var = t_column, mov_var = var_column, bin = bin_time)

        elif var_column == 'moving':
            hmm_df[var_column] = np.where(hmm_df[var_column] == True, 1, 0)
            gb = bin_to_list(hmm_df, t_var = t_column, mov_var = var_column, bin = bin_time)

        else:
            gb = bin_to_list(hmm_df, t_var = t_column, mov_var = var_column, bin = bin_time)

        # split runs into test and train lists
        test_train_split = round(len(gb) * (test_size/100))
        rand_runs = np.random.permutation(gb)
        train = rand_runs[test_train_split:]
        test = rand_runs[:test_train_split]

        len_seq_train = [len(ar) for ar in train]
        len_seq_test = [len(ar) for ar in test]

        seq_train = np.concatenate(train, 0)
        seq_train = seq_train.reshape(-1, 1)
        seq_test = np.concatenate(test, 0)
        seq_test = seq_test.reshape(-1, 1)

        for i in range(iterations):
            print(f"Iteration {i+1} of {iterations}")
            
            init_params = ''
            # h = hmm.MultinomialHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', verbose = verbose)
            h = hmm.CategoricalHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', verbose = verbose)

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
                try:
                    h_old = pickle.load(open(file_name, "rb"))
                    if h.score(seq_test, len_seq_test) > h_old.score(seq_test, len_seq_test):
                        print('New Matrix:')
                        df_t = pd.DataFrame(h.transmat_, index = states, columns = states)
                        print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
                        
                        with open(file_name, "wb") as file: pickle.dump(h, file)
                except OSError as e:
                    with open(file_name, "wb") as file: pickle.dump(h, file)

            else:
                h_old = pickle.load(open(file_name, "rb"))
                if h.score(seq_test, len_seq_test) > h_old.score(seq_test, len_seq_test):
                    print('New Matrix:')
                    df_t = pd.DataFrame(h.transmat_, index = states, columns = states)
                    print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
                    with open(file_name, "wb") as file: pickle.dump(h, file)

            if i+1 == iterations:
                h = pickle.load(open(file_name, "rb"))
                #print tables of trained emission probabilties, not accessible as objects for the user
                self._hmm_table(start_prob = h.startprob_, trans_prob = h.transmat_, emission_prob = h.emissionprob_, state_names = states, observable_names = observables)
                return h

    def sleep_bout_analysis(self, sleep_column = 'asleep', as_hist = False, bin_size = 1, max_bins = 30, time_immobile = 5, asleep = True):
        """ 
        Behavpy_HMM version wrapped from behavpy, see behavpy_class for doc string
        """

        if sleep_column not in self.columns:
            warnings.warn(f'Column heading "{sleep_column}", is not in the data table')
            exit()

        tdf = self.reset_index().copy(deep = True)
        return behavpy_HMM(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bout_analysis, 
                                                                                                            var_name = sleep_column, 
                                                                                                            as_hist = as_hist, 
                                                                                                            bin_size = bin_size, 
                                                                                                            max_bins = max_bins, 
                                                                                                            time_immobile = time_immobile, 
                                                                                                            asleep = asleep
            )), tdf.meta, check = True)

    def curate_dead_animals(self, t_column = 't', mov_column = 'moving', time_window = 24, prop_immobile = 0.01, resolution = 24):
        """ 
        Behavpy_HMM version wrapped from behavpy, see behavpy_class for doc string
        """

        if t_column not in self.columns.tolist():
            warnings.warn('Variable name entered, {}, for t_column is not a column heading!'.format(t_column))
            exit()
        
        if mov_column not in self.columns.tolist():
            warnings.warn('Variable name entered, {}, for mov_column is not a column heading!'.format(mov_column))
            exit()

        tdf = self.reset_index().copy(deep=True)
        return behavpy_HMM(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_curate_dead_animals,
                                                                                                            time_var = t_column,
                                                                                                            moving_var = mov_column,
                                                                                                            time_window = time_window, 
                                                                                                            prop_immobile = prop_immobile,
                                                                                                            resolution = resolution
        )), tdf.meta, check = True)

    def bin_time(self, column, bin_secs, t_column = 't', function = 'mean'):
        """
        Behavpy_HMM version wrapped from behavpy, see behavpy_class for doc string
        """

        if column not in self.columns:
            warnings.warn('Column heading "{}", is not in the data table'.format(column))
            exit()

        tdf = self.reset_index().copy(deep=True)
        return behavpy_HMM(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bin_data,
                                                                                                column = column, 
                                                                                                bin_column = t_column,
                                                                                                function = function, 
                                                                                                bin_secs = bin_secs
        )), tdf.meta, check = True)

    def motion_detector(self, time_window_length = 10, velocity_correction_coef = 3e-3, masking_duration = 0, optional_columns = None):
        """
        Behavpy_HMM version wrapped from behavpy, see behavpy_class for doc string
        """

        if optional_columns is not None:
            if optional_columns not in self.columns:
                warnings.warn('Column heading "{}", is not in the data table'.format(optional_columns))
                exit()

        tdf = self.reset_index().copy(deep=True)
        return  behavpy_HMM(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_motion_detector,
                                                                                                        time_window_length = time_window_length,
                                                                                                        velocity_correction_coef = velocity_correction_coef,
                                                                                                        masking_duration = masking_duration,
                                                                                                        optional_columns = optional_columns
        )), tdf.meta, check = True)

    def sleep_contiguous(self, mov_column = 'moving', t_column = 't', time_window_length = 10, min_time_immobile = 300):
        """
        Method version of the sleep annotation function.
        This function first uses a motion classifier to decide whether an animal is moving during a given time window.
        Then, it defines sleep as contiguous immobility for a minimum duration.
        See function for paramater details

        returns a behavpy object with added columns like 'moving' and 'asleep'
        """
        if mov_column not in self.columns.tolist():
            warnings.warn(f'The movement column {mov_column} is not in the dataset')
            exit()
        if t_column not in self.columns.tolist():
            warnings.warn(f'The time column {t_column} is not in the dataset')
            exit()  

        tdf = self.reset_index().copy(deep = True)
        return behavpy_HMM(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_sleep_contiguous,
                                                                                                        mov_column = mov_column,
                                                                                                        t_column = t_column,
                                                                                                        time_window_length = time_window_length,
                                                                                                        min_time_immobile = min_time_immobile
        )), tdf.meta, check = True)

    def hmm_display(self, hmm, states, observables):
        """
        Prints to screen the transion probabilities for the hidden state and observables for a given hmmlearn hmm object
        """
        self._hmm_table(start_prob = hmm.startprob_, trans_prob = hmm.transmat_, emission_prob = hmm.emissionprob_, state_names = states, observable_names = observables)

    def plot_hmm_overtime(self, hmm, variable = 'moving', labels = None, colours = None, wrapped = False, bin = 60, func = 'max', avg_window = 30, day_length = 24, lights_off = 12, title = '', grids = False):
        """
        Creates a plot of all states overlayed with y-axis shows the liklihood of being in a sleep state and the x-axis showing time in hours.
        The plot is generated through the plotly package

        Params:
        @self = behavpy_HMM,
        @hmm = hmmlearn.hmm.MultinomialHMM, this should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
        @variable = string, the column heading of the variable of interest. Default is "moving"
        @labels = list[string], the names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake']
        @colours = list[string], the name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red)
        It accepts a specific colour or an array of numbers that are acceptable to plotly
        @wrapped = bool, if True the plot will be limited to a 24 hour day average
        @bin = int, the time in seconds you want to bin the movement data to, default is 60 or 1 minute
        @func = string, when binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
        @avg_window, int, the window in minutes you want the moving average to be applied to. Default is 30 mins
        @circadian_night, int, the hour when lights are off during the experiment. Default is ZT 12
        @save = bool/string, if not False then save as the location and file name of the save file

        returns None
        """
        assert isinstance(wrapped, bool)

        df = self.copy(deep = True)

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)

        states_list, time_list = self._hmm_decode(df, hmm, bin, variable, func)

        df = pd.DataFrame()
        for l, t in zip(states_list, time_list):
            tdf = hmm_pct_state(l, t, list(range(len(labels))), avg_window = int((avg_window * 60)/bin))
            df = pd.concat([df, tdf], ignore_index = True)

        if wrapped is True:
            df['t'] = df['t'].map(lambda t: t % (60*60*day_length))

        df['t'] = df['t'] / (60*60)
        t_min = int(12 * floor(df.t.min() / 12))
        t_max = int(12 * ceil(df.t.max() / 12))    
        t_range = [t_min, t_max]  

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [-0.025, 1.01], t0 = 0, dtick = 0.2, ylabel = 'Probability of being in state', title = title, grid = grids)
        self._plot_xlayout(fig, xrange = t_range, t0 = 0, dtick = day_length/4, xlabel = 'ZT (Hours)')

        for c, (col, n) in enumerate(zip(colours, labels)):

            column = f'state_{c}'

            gb_df = df.groupby('t').agg(**{
                        'mean' : (column, 'mean'), 
                        'SD' : (column, 'std'),
                        'count' : (column, 'count')
                    })

            gb_df['SE'] = (1.96*gb_df['SD']) / np.sqrt(gb_df['count'])
            gb_df['y_max'] = gb_df['mean'] + gb_df['SE']
            gb_df['y_min'] = gb_df['mean'] - gb_df['SE']
            gb_df = gb_df.reset_index()

            upper, trace, lower, _ = self._plot_line(df = gb_df, x_col = 't', name = n, marker_col = col)
            fig.add_trace(upper)
            fig.add_trace(trace) 
            fig.add_trace(lower)

        # Light-Dark annotaion bars
        bar_shapes, min_bar = circadian_bars(t_min, t_max, max_y = 1, day_length = day_length, lights_off = lights_off)
        fig.update_layout(shapes=list(bar_shapes.values()))

        return fig

    def plot_hmm_split(self, hmm, variable = 'moving', labels = None, colours= None, facet_labels = None, facet_col = None, facet_arg = None, wrapped = False, bin = 60, func = 'max', avg_window = 30, day_length = 24, lights_off = 12, title = '', grids = False):
        """ works for any number of states """

        assert isinstance(wrapped, bool)

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        start_colours, end_colours = self._adjust_colours(colours)

        if len(labels) <= 2:
            nrows = 1
            ncols =2
        else:
            nrows =  2
            ncols = round(len(labels) / 2)

        fig = make_subplots(rows= nrows, cols= ncols, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02, horizontal_spacing=0.02)

        col_list = list(range(1, ncols+1)) * nrows
        row_list = list([i] * ncols for i in range(1, nrows+1))
        row_list = [item for sublist in row_list for item in sublist]

        colour_range_dict = {}
        colours_dict = {'start' : start_colours, 'end' : end_colours}
        for q in range(0,len(labels)):
            start_color = colours_dict.get('start')[q]
            end_color = colours_dict.get('end')[q]
            N = len(facet_arg)
            colour_range_dict[q] = [x.hex for x in list(Color(start_color).range_to(Color(end_color), N))]

        for c, (arg, n, h, b) in enumerate(zip(facet_arg, facet_labels, h_list, b_list)):   

            if arg != None:
                d = self.xmv(facet_col, arg)
            else:
                d = self.copy(deep = True)

            states_list, time_list = self._hmm_decode(d, h, b, variable, func)

            analysed_df = pd.DataFrame()
            for l, t in zip(states_list, time_list):
                temp_df = hmm_pct_state(l, t, [0, 1, 2, 3], avg_window = int((avg_window * 60)/b))
                analysed_df = pd.concat([analysed_df, temp_df], ignore_index = False)

            if wrapped is True:
                analysed_df['t'] = analysed_df['t'].map(lambda t: t % (60*60*day_length))
            analysed_df['t'] = analysed_df['t'] / (60*60)

            if 'control' in n.lower() or 'baseline' in n.lower() or 'ctrl' in n.lower():
                black_list = ['black'] * len(facet_arg)
                black_range_dict = {0 : black_list, 1: black_list, 2 : black_list, 3 : black_list}
                marker_col = black_range_dict
            else:
                marker_col = colour_range_dict

            t_min = int(12 * floor(analysed_df.t.min() / 12))
            t_max = int(12 * ceil(analysed_df.t.max() / 12))    
            t_range = [t_min, t_max]  

            for i, (lab, row, col) in enumerate(zip(labels, row_list, col_list)):

                column = f'state_{i}'

                gb_df = analysed_df.groupby('t').agg(**{
                            'mean' : (column, 'mean'), 
                            'SD' : (column, 'std'),
                            'count' : (column, 'count')
                        })

                gb_df['SE'] = (1.96*gb_df['SD']) / np.sqrt(gb_df['count'])
                gb_df['y_max'] = gb_df['mean'] + gb_df['SE']
                gb_df['y_min'] = gb_df['mean'] - gb_df['SE']
                gb_df = gb_df.reset_index()

                upper, trace, lower, _ = self._plot_line(df = gb_df, x_col = 't', name = n, marker_col = marker_col.get(i)[c])
                fig.add_trace(upper,row=row, col=col)
                fig.add_trace(trace, row=row, col=col) 
                fig.add_trace(lower, row=row, col=col)

                if c == 0:
                    fig.add_annotation(xref='x domain', yref='y domain',x=0.1, y=0.9, text = lab, font = {'size': 18, 'color' : 'black'}, showarrow = False,
                    row=row, col=col)

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
            showgrid = grids
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
                    text = 'Likelihood to be in sleep state',
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
        bar_shapes, min_bar = circadian_bars(t_min, t_max, max_y = 1, day_length = day_length, lights_off = lights_off, split = len(labels))
        fig.update_layout(shapes=list(bar_shapes.values()))

        return fig

    def plot_hmm_quantify(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', grids = False):
        """
        
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func) for n, d, h, b in zip(facet_arg, df_list, h_list, b_list)}

        def analysis(array_states):
            rows = []
            for array in array_states:
                unique, counts = np.unique(array, return_counts=True)
                row = dict(zip(unique, counts))
                rows.append(row)
            counts_all =  pd.DataFrame(rows)
            counts_all['sum'] = counts_all.sum(axis=1)
            counts_all = counts_all.iloc[:, list_states[0]: list_states[-1]+1].div(counts_all['sum'], axis=0)
            counts_all.fillna(0, inplace = True)
            return counts_all

        analysed_dict = {f'df{n}' : analysis(decoded_dict[f'df{n}'][0]) for n in facet_arg}
        
        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 1.01], t0 = 0, dtick = 0.2, ylabel = 'Fraction of time in each state', title = title, grid = grids)

        stats_dict = {}

        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):
                
                try:
                    median, q3, q1, zlist = self._zscore_bootstrap(analysed_dict[f'df{arg}'][state].to_numpy())
                except KeyError:
                    median, q3, q1, zlist = [0], [0], [0], [np.nan]
                
                stats_dict[f'{arg}_{lab}'] = zlist

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

                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [i], colour =  marker_col, showlegend = False, name = i, xaxis = f'x{state+1}'))

                label_list = [i] * len(zlist)
                fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                showlegend = False, name = i, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))
        
        return fig, stats_df
    
    def plot_hmm_quantify_length(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', grids = False):
        
        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func) for n, d, h, b in zip(facet_arg, df_list, h_list, b_list)}

        def analysis(states, t_diff):
            df_lengths = pd.DataFrame()
            for l in states:
                length = hmm_mean_length(l, delta_t = t_diff) 
                df_lengths = pd.concat([df_lengths, length], ignore_index= True)
            return df_lengths

        analysed_dict = {f'df{n}' : analysis(decoded_dict[f'df{n}'][0], b) for n, b in zip(facet_arg, b_list)}

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = 0.69897000433, ylabel = 'Length of state bout (mins)', ytype = 'log', title = title, grid = grids)

        gb_dict = {f'gb{n}' : analysed_dict[f'df{n}'].groupby('state') for n in facet_arg}

        stats_dict = {}

        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):

                try:
                    median, q3, q1, zlist = self._zscore_bootstrap(gb_dict[f'gb{arg}'].get_group(state)['mean_length'].to_numpy())
                except KeyError:
                    median, q3, q1, zlist = [0], [0], [0], [np.nan]
                stats_dict[f'{arg}_{lab}'] = zlist

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

                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [i], colour =  marker_col, showlegend = False, name = i, xaxis = f'x{state+1}'))

                label_list = [i] * len(zlist)
                fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                showlegend = False, name = i, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)
        
        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_hmm_quantify_length_min_max(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', grids = False):
            
        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func) for n, d, h, b in zip(facet_arg, df_list, h_list, b_list)}

        def analysis(states, t_diff):
            df_lengths = pd.DataFrame()
            for l in states:
                length = hmm_mean_length(l, delta_t = t_diff, raw = True) 
                df_lengths = pd.concat([df_lengths, length], ignore_index= True)
            return df_lengths

        analysed_dict = {f'df{n}' : analysis(decoded_dict[f'df{n}'][0], b) for n, b in zip(facet_arg, b_list)}

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = False, t0 = 0, dtick = 0.69897000433, ylabel = 'Length of state bout (mins)', ytype = 'log', title = title, grid = grids)

        gb_dict = {f'gb{n}' : analysed_dict[f'df{n}'].groupby('state') for n in facet_arg}

        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):
                
                try:
                    median, q3, q1, _ = self._zscore_bootstrap(gb_dict[f'gb{arg}'].get_group(state)['length_adjusted'].to_numpy(), min_max = True)
                except KeyError:
                    median, q3, q1 = [0], [0], [0]

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
                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [i], colour =  marker_col, showlegend = False, name = i, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)
        
        return fig
            
    def plot_hmm_quantify_transition(self, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', grids = False):

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func) for n, d, h, b in zip(facet_arg, df_list, h_list, b_list)}

        def analysis(states):
            df_trans = pd.DataFrame()
            for l in states:
                trans = hmm_pct_transition(l, list_states) 
                df_trans = pd.concat([df_trans, trans], ignore_index= True)
            return df_trans

        analysed_dict = {f'df{n}' : analysis(decoded_dict[f'df{n}'][0]) for n in facet_arg}

        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 1.05], t0 = 0, dtick = 0.2, ylabel = 'Fraction of runs of each state', title = title, grid = grids)

        stats_dict = {}

        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):
                
                try:
                    median, q3, q1, zlist = self._zscore_bootstrap(analysed_dict[f'df{arg}'][str(state)].to_numpy())  
                except KeyError:
                    median, q3, q1, zlist = [0], [0], [0], [np.nan]

                stats_dict[f'{arg}_{lab}'] = zlist

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

                fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                x = [i], colour =  marker_col, showlegend = False, name = i, xaxis = f'x{state+1}'))

                label_list = [i] * len(zlist)
                fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                showlegend = False, name = i, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))

            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))

        return fig, stats_df

    def plot_hmm_raw(self, hmm, variable = 'moving', colours = None, num_plots = 5, bin = 60, mago_df = None, func = 'max', title = ''):
        """ plots the raw dedoded hmm model per fly (total = num_plots) 
            If hmm is a list of hmm objects, the number of plots will equal the length of that list. Use this to compare hmm models.
            If hmm_compare is True the same specimens data will be used for each plot with a different hmm """

        if colours is None:
            if isinstance(hmm, list):
                h = hmm[0]
            else:
                h = hmm
            states = h.transmat_.shape[0]
            if states == 4:
                colours = self._colours_four
            else:
                warnings.warn(f'Your trained HMM is not 4 states, please provide the {h.transmat_.shape[0]} colours for this hmm. See doc string for more info')
                exit() 

        colours_index = {c : col for c, col in enumerate(colours)}

        if mago_df is not None:
            assert isinstance(mago_df, behavpy) or isinstance(mago_df, behavpy_HMM), 'The mAGO dataframe is not a behavpy or behavpy_HMM class'

        if isinstance(hmm, list):
            num_plots = len(hmm)
            rand_flies = [np.random.permutation(list(set(self.meta.index)))[0]] * num_plots
            h_list = hmm
            if isinstance(bin, list):
                b_list = bin 
            else:
                b_list = [bin] * num_plots
        else:
            rand_flies = np.random.permutation(list(set(self.meta.index)))[:num_plots]
            h_list = [hmm] * num_plots
            b_list = [bin] * num_plots

        df_list = [self.xmv('id', id) for id in rand_flies]
        decoded = [self._hmm_decode(d, h, b, variable, func) for d, h, b in zip(df_list, h_list, b_list)]

        def analyse(data, df_variable):
            states = data[0][0]
            time = data[1][0]
            id = df_variable.index.tolist()
            var = df_variable[variable].tolist()
            previous_states = np.array(states[:-1], dtype = float)
            previous_states = np.insert(previous_states, 0, np.nan)
            previous_var = np.array(var[:-1], dtype = float)
            previous_var = np.insert(previous_var, 0, np.nan)
            all_df = pd.DataFrame(data = zip(id, states, time, var, previous_states, previous_var))
            all_df.columns = ['id','state', 't', 'var','previous_state', 'previous_var']
            all_df.dropna(inplace = True)
            all_df['colour'] = all_df['previous_state'].map(colours_index)
            all_df.set_index('id', inplace = True)
            return all_df

        analysed = [analyse(i, v) for i, v in zip(decoded, df_list)]

        fig = make_subplots(
        rows= num_plots, 
        cols=1,
        shared_xaxes=True, 
        shared_yaxes=True, 
        vertical_spacing=0.02,
        horizontal_spacing=0.02
        )

        for c, (df, b) in enumerate(zip(analysed, b_list)):
            id = df.first_valid_index()
            print(f'Plotting: {id}')
            if mago_df is not None:
                df2 = mago_df.xmv('id', id)
                df2 = df2[df2['has_interacted'] == 1]
                df2['t'] = df2['interaction_t'].map(lambda t:  b * floor(t / b))
                df2.reset_index(inplace = True)
                df = pd.merge(df, df2, how = 'outer', on = ['id', 't'])
                df['colour'] = np.where(df['has_responded'] == True, 'purple', df['colour'])
                df['colour'] = np.where(df['has_responded'] == False, 'lime', df['colour'])
                df['t'] = df['t'].map(lambda t: t / (60*60))
            
            else:
                df['t'] = df['t'].map(lambda t: t / (60*60))

            trace1 = go.Scatter(
                showlegend = False,
                y = df['previous_state'],
                x = df['t'],
                mode = 'markers+lines', 
                marker = dict(
                    color = df['colour'],
                    ),
                line = dict(
                    color = 'black',
                    width = 0.5
                )
                )
            fig.add_trace(trace1, row = c+1, col= 1)

        y_range = [-0.2, states-0.8]
        self._plot_ylayout(fig, yrange = y_range, t0 = 0, dtick = False, ylabel = '', title = title)

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
                    size = 18,
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
                    size = 18,
                    color = 'black'
                )
            ),
            row = num_plots,
            col = 1
        )

        return fig
    
    def plot_hmm_response(self, mov_df, hmm, variable = 'moving', labels = None, colours = None, facet_col = None, facet_arg = None, bin = 60, facet_labels = None, func = 'max', title = '', grids = False):
        """
        
        """

        labels, colours = self._check_hmm_shape(hm = hmm, lab = labels, col = colours)
        list_states = list(range(len(labels)))
        facet_arg, facet_labels, h_list, b_list = self._check_lists_hmm(facet_col, facet_arg, facet_labels, hmm, bin)

        if facet_col is not None:
            df_list = [self.xmv(facet_col, arg) for arg in facet_arg]
            mov_df_list = [mov_df.xmv(facet_col, arg) for arg in facet_arg]
        else:
            df_list = [self.copy(deep = True)]
            mov_df_list = [mov_df.copy(deep = True)]

        decoded_dict = {f'df{n}' : self._hmm_decode(d, h, b, variable, func, return_type = 'table') for n, d, h, b in zip(facet_arg, mov_df_list, h_list, b_list)}
        puff_dict = {f'pdf{n}' : d for n, d in zip(facet_arg, df_list)}

        def alter_merge(data, puff):
            puff['bin'] = puff['interaction_t'].map(lambda t:  60 * floor(t / 60))
            puff.reset_index(inplace = True)

            merged = pd.merge(data, puff, how = 'inner', on = ['id', 'bin'])
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

        analysed_dict = {f'df{n}' : alter_merge(decoded_dict[f'df{n}'], puff_dict[f'pdf{n}']) for n in facet_arg}
        
        fig = go.Figure()
        self._plot_ylayout(fig, yrange = [0, 1.01], t0 = 0, dtick = 0.2, ylabel = 'Response Rate', title = title, grid = grids)

        stats_dict = {}

        for state, col, lab in zip(list_states, colours, labels):

            for arg, i in zip(facet_arg, facet_labels):

                for q in [2, 1]:
                    try:
                        median, q3, q1, zlist = self._zscore_bootstrap(analysed_dict[f'df{arg}'][f'int_{q}'][state])
                    except KeyError:
                        continue

                    stats_dict[f'{arg}_{lab}_{q}'] = zlist

                    if q == 2:
                        lab = f'{i} Spon. mov.'
                    else:
                        lab = i

                    if 'baseline' in lab.lower() or 'control' in lab.lower() or 'ctrl' in lab.lower():
                            marker_col = 'black'
                    elif 'spon. mov.' in lab.lower():
                            marker_col = 'grey'
                    else:
                        marker_col = col

                    fig.add_trace(self._plot_meanbox(median = [median], q3 = [q3], q1 = [q1], 
                    x = [lab], colour =  marker_col, showlegend = False, name = lab, xaxis = f'x{state+1}'))

                    label_list = [lab] * len(zlist)
                    fig.add_trace(self._plot_boxpoints(y = zlist, x = label_list, colour = marker_col, 
                    showlegend = False, name = lab, xaxis = f'x{state+1}'))

            domains = np.arange(0, 1+(1/len(labels)), 1/len(labels))
            axis = f'xaxis{state+1}'
            self._plot_xlayout(fig, xrange = False, t0 = False, dtick = False, xlabel = lab, domains = domains[state:state+2], axis = axis)

        stats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in stats_dict.items()]))
        
        return fig, stats_df
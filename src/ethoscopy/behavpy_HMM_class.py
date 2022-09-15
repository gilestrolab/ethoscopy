import pandas as pd
import numpy as np 
import warnings

import pickle
from tabulate import tabulate
from hmmlearn import hmm
from math import floor
from sys import exit

from ethoscopy.behavpy_class import behavpy
from ethoscopy.misc.format_warning import format_warning
from ethoscopy.analyse import max_velocity_detector, sleep_annotation
from ethoscopy.misc.rle import rle

class behavpy_HMM(behavpy):
    """
    The behavpy_HMM class is a subclassed version of behavpy, itself a subclass of pandas. See behavpy doc strings for more information on behavpy.

    Behavpy_HMM has been augmented to include methods that generate trained Hidden Markov Models and 

    """
    warnings.formatwarning = format_warning

    @property
    def _constructor(self):
        return behavpy_HMM

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




import ethoscopy as etho
print(etho.__version__)
import pickle
import pandas as pd
import numpy as np
from functools import partial
import plotly.graph_objs as go

hmm_list = []

# hmm_arg_list = ['D.vir', 'D.ere', 'D.wil', 'D.sec', 'D.yak', 'D.sims', 'D.mel', 'D.mel', 'wild_d.mel']
# arg_list = ['D.vir', 'D.ere', 'D.wil', 'D.sec', 'D.yak', 'D.sims', 'D.mel', 'wild_d.mel', 'wild_d.mel']
# labels = ['D.vir', 'D.ere', 'D.wil', 'D.sec', 'D.yak', 'D.sims', 'D.mel', 'wild_d.mel (CS hmm)', 'wild_d.mel']

labels = ['D.virilis', 'D.erecta', 'D.willistoni', 'D.sechellia', 'D.yakuba']
hmm_arg_list = ['D.vir', 'D.ere', 'D.wil', 'D.sec', 'D.yak']
arg_list = ['D.vir', 'D.ere', 'D.wil', 'D.sec', 'D.yak']

# bin_list = [60] * len(hmm_arg_list)
# for i in hmm_arg_list:
#     with open(rf'C:\Users\lab\Modelling_Deep_Sleep\species\hmm_species\{i}_hmm.pkl', 'rb') as file: 
#         h = pickle.load(file)
#         hmm_list.append(h)

meta = pd.read_pickle(r'C:\Users\lab\Modelling_Deep_Sleep\species\species_baseline_meta.pkl')
data = pd.read_pickle(r'C:\Users\lab\Modelling_Deep_Sleep\species\species_baseline_data.pkl')

df = etho.behavpy_HMM(data, meta, check = True)

df = df.t_filter(start_time = 24, end_time = 144)

# df.plot_overtime(
# variable = 'moving' ,
# facet_col = 'species',
# facet_arg = arg_list,
# facet_labels = labels,
# day_length = 35,
# lights_off = 28,
# save = False, 
# location = '')

# df.plot_quantify(
# variable = 'moving' ,
# facet_col = 'species',
# facet_arg = arg_list,
# # facet_labels = labels,
# title = 'This is a plot for quantifying',
# grids = True,
# save = False, 
# location = '')


# hist_df = df.sleep_bout_analysis(sleep_column = 'moving', as_hist = True, max_bins = 30, asleep = True)

# df.plot_actogram(facet_col = 'species', facet_arg = arg_list)

# df = df.xmv('species', 'D.vir')
# df.meta['acto_labels'] = [f'D.Vir {i+1}' for i in range(len(df.meta))]

# df.plot_actogram(mov_variable = 'moving', individual = True, individual_label = 'acto_labels')

# df1 = df.xmv('species', 'D.sec')
# df2 = df.xmv('species', 'D.vir')
# df2.meta['new_col'] = ['poo'] * len(df2.meta)
# print(df2.meta)
# new_df = df1.concat(df2)
# print(new_df.meta)

# df.plot_anticipation_score(
# facet_col = 'species',
# facet_arg = arg_list,
# facet_labels = labels,
# save = False, 
# location = '')

# df = df.t_filter(start_time = 24, end_time = 144)
# df.plot_overtime(
# variable = 'moving', 
# facet_col = 'species',
# facet_arg = arg_list,
# facet_labels = labels,
# save = False, 
# location = '')

# df.plot_quantify(
# variable = 'max_velocity', 
# facet_col = 'species',
# facet_arg = arg_list,
# facet_labels = labels,
# save = False, 
# location = '', 
# title = 'this is a title')

# df.plot_day_night(
# variable = 'max_velocity', 
# facet_col = 'species',
# facet_arg = arg_list,
# facet_labels = labels,
# save = False, 
# location = '')

# df.plot_compare_variables(
# variables = ['micro', 'walk', 'mean_velocity'],
# facet_col = 'species',
# facet_arg = arg_list,
# facet_labels = labels,
# grids = True,
# save = False, 
# location = '')

# df = df.xmv('species','D.vir')
# df.plot_hmm_overtime(
# variable = 'moving', 
# hmm = hmm_list[0],
# save = False, 
# location = '')

# df.plot_hmm_split(
# hmm = hmm_list, 
# labels = ['purple', 'gold', 'black'],
# colours = ['purple', 'gold', 'black'],
# variable = 'moving', 
# facet_col = 'species',
# facet_arg = arg_list, 
# facet_labels = labels,
# wrapped = True, 
# bin = bin_list, 
# save = False, 
# location = '')

# df.plot_hmm_quantify(
# hmm = hmm_list, 
# variable = 'moving', 
# facet_col = 'species',
# facet_arg = arg_list, 
# facet_labels = labels,
# bin = bin_list, 
# save = False, 
# location = '')

# df.plot_hmm_quantify_length(
# hmm = hmm_list, 
# variable = 'moving', 
# facet_col = 'species',
# facet_arg = arg_list, 
# facet_labels = labels,
# bin = bin_list, 
# save = False, 
# location = '')

# df.plot_hmm_quantify_length_min_max(
# hmm = hmm_list, 
# variable = 'moving', 
# facet_col = 'species',
# facet_arg = arg_list, 
# facet_labels = labels,
# bin = bin_list, 
# save = False, 
# location = '')

# df.plot_hmm_quantify_transition(
# hmm = hmm_list, 
# variable = 'moving', 
# facet_col = 'species',
# facet_arg = arg_list, 
# facet_labels = labels,
# bin = bin_list, 
# save = False, 
# location = '')

# df.plot_hmm_quantify_transition(
# hmm = hmm_list, 
# variable = 'moving', 
# facet_col = 'species',
# facet_arg = arg_list, 
# facet_labels = labels,
# bin = bin_list, 
# save = False, 
# location = '')

# df = df.xmv('species','D.vir')
# df.plot_hmm_raw(hmm = hmm_list[0], num_plots = 7)
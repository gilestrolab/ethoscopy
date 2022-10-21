import ethoscopy as etho
print(etho.__version__)
import pickle
import pandas as pd
import numpy as np

import plotly.graph_objs as go

labels = ['D.virilis', 'D.erecta', 'D.willistoni', 'D.sechellia', 'D.yakuba']
arg_list = ['D.vir', 'D.ere', 'D.wil', 'D.sec', 'D.yak']

meta = pd.read_pickle(r'C:\Users\lab\Modelling_Deep_Sleep\species\species_baseline_meta.pkl')
data = pd.read_pickle(r'C:\Users\lab\Modelling_Deep_Sleep\species\species_baseline_data.pkl')

df = etho.behavpy(data, meta, check = True)

def moving_average(df, n = 60):
    ar = df['norm_movement'].to_numpy()
    ret = np.cumsum(ar, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    ret = np.insert(ret, 0, [np.nan]*(n-1))
    df['avg_norm_mov'] = ret
    return df

# data = data.reset_index()
# m = data.meta
# data = data.groupby('id', group_keys = False).apply(moving_average)
# data = set_behavpy(m, data)

# data.dropna(subset = 'avg_norm_mov', inplace = True)

df11 = data1.xmv('genotype', 'ko')
df21 = data1.xmv('genotype', 'wt')

df12 = data2.xmv('genotype', 'ko')
df22 = data2.xmv('genotype', 'wt')

# for i in list(set(df22.index.tolist())):
#     print(i)
#     print(len(df22[df22.index == i]))

def fftPlot(sig, dt=None, title = None, location = None):

    # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

    layout = go.Layout(
        title = title,
        yaxis = dict(
            color = 'black',
            linecolor = 'black',
            title = dict(
                text = 'Magnitude',
                font = dict(
                    size = 24,
                )
            ),
            tick0 = 0,
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
                text = 'freq [per 24 hours]',
                font = dict(
                    size = 24,
                    color = 'black'
                )
            ),
            range = [0, 4],
            dtick = 0.5,
            ticks = 'outside',
            tickwidth = 2,
            tickfont = dict(
                size = 18
            )
        ),
        plot_bgcolor = 'white',
        yaxis_showgrid = False,
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
    for c, n, s in zip(['red', 'blue', 'darkorange', 'darkgreen'], ['KO LD', 'WT LD', 'KO DD', 'WT DD'], sig):

        for count, i in enumerate(list(set(s.index.tolist()))):

            # i = s['avg_norm_mov'][s.index == i].to_numpy()
            i = s['norm_movement'][s.index == i].to_numpy()

            if dt is None:
                dt = 1
                t = np.arange(0, i.shape[-1])
            else:
                t = np.arange(0, i.shape[-1]) * dt

            if i.shape[0] % 2 != 0:
                warnings.warn("signal preferred to be even in size, autoFixing it...")
                t = t[0:-1]
                i = i[0:-1]

            sigFFT = np.fft.fft(i - np.mean(i)) / t.shape[0]  # Divided by size t for coherent magnitude
            freq = np.fft.fftfreq(t.shape[0], d=dt)

            # Plot analytic signal - right half of frequence axis needed only...
            firstNegInd = np.argmax(freq < 0)
            freqAxisPos = freq[0:firstNegInd]
            sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

            if count == 0:
                all_sigFFTPos = sigFFTPos
            else:
                all_sigFFTPos = all_sigFFTPos + sigFFTPos

            if count == len(list(set(s.index.tolist()))) - 1:
                all_sigFFTPos = all_sigFFTPos / len(list(set(s.index.tolist())))

        fig.add_trace(go.Scatter(
            showlegend = True,
            x = freqAxisPos * 86400,
            # y = np.abs(sigFFTPos),
            y = np.abs(all_sigFFTPos),
            name = n,
            # name = f'{n}',
            mode= 'lines',
            marker=dict(
                color = c
                )
            ))

    fig.show()
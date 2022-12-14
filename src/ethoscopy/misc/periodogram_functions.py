import pandas as pd
import numpy as np 

from pywt import cwt

from functools import partial

from scipy.stats import chi2
from scipy.special import chdtri
from scipy.signal import welch
from scipy.fftpack import fft

from astropy.timeseries import LombScargle

def chi_squared(data, t_col, var, period_range = [10, 36], freq = 1 / 60, alpha = 0.01, time_resolution = 0.1 * (60*60)):

    def calc_Qp(target_period, values, freq):

        col_num = round(target_period * freq)
        row_num = len(values) / col_num
        repeat_int = round((len(values)-1) / col_num) + 1
        mod_col = np.array(list(range(1, col_num+1))*repeat_int)[:len(values)]
        calc_df = pd.DataFrame(data = {'x' : mod_col, 'values' : values})
        avg_P = calc_df.groupby('x').agg(**{
                    'values' : ('values', 'mean')
        })
        avg_P = avg_P['values'].to_numpy()
        avg_all = np.mean(values)
        numerator = sum((avg_P - avg_all)**2) * (len(calc_df) * row_num)
        denomenator = sum((values - avg_all)**2)
        
        return numerator / denomenator    

    id  = data.name
    start = 60*60*period_range[0]
    end = 60*60*period_range[1]
    t, y =  data[t_col].to_numpy(), data[var].to_numpy()
    
    out = pd.DataFrame(data = {'id' : id, 'period' : np.arange(start, end + time_resolution, time_resolution)})
    corrected_alpha = 1 - (1 - alpha)**(1/len(out))
    cal_fun = partial(calc_Qp, values = y, freq = freq)
    out['power'] = out['period'].apply(cal_fun)
    round_period = [round(x) for x in out['period'].to_numpy()*freq]

    out['sig_threshold'] = chdtri(round_period, corrected_alpha)
    out['p_value'] = out.apply(lambda x: 1 - chi2.cdf(x.power, x.period*freq), axis = 1)
    out['period'] = out['period'] / (60*60)

    out.set_index('id', inplace = True)
    out = out.sort_values('period', ascending = True)

    return out

def lomb_scargle(data, t_col, var, period_range = [10, 36], alpha = 0.01, **kwargs):

    id = data.name
    start = 60*60*period_range[0]
    end = 60*60*period_range[1]
    t, y =  data[t_col].to_numpy(), data[var].to_numpy()
    ls = LombScargle(t, y)
    period, power = ls.autopower(minimum_frequency= 1/end, maximum_frequency = 1/start, samples_per_peak = 20)
    period = 1 / period
    period = period / (60*60)
    out = pd.DataFrame(data = {'id' : id, 'period' : period, 'power' : power, 'sig_threshold' : ls.false_alarm_level(alpha)})

    out.set_index('id', inplace = True)
    out = out.sort_values('period', ascending = True)

    return out

def fourier(data, t_col, var, period_range = [10, 36], alpha = 0.01, **kwargs):

    id = data.name
    start = 60*60*period_range[0]
    end = 60*60*period_range[1]
    t, y =  data[t_col].to_numpy(), data[var].to_numpy()

    N = len(y)
    dt = t[1] - t[0]
    period = np.linspace(0.0, 1.0/(2.0*dt), N//2)
    power = fft(y)
    power = 2.0/N * np.abs(power[0:N//2])
    long_period = 1 / period  
    period = long_period[(long_period > start) & (long_period < end)]
    period = period / (60*60)
    short_power = power[(long_period > start) & (long_period < end)]

    sig_thresh = -np.mean(power) * np.log((1 - ((1 - alpha) ** (1 / len(power)))))
    out = pd.DataFrame(data = {'id' : id, 'period' : period, 'power' : short_power, 'sig_threshold' : sig_thresh})    
    
    out.set_index('id', inplace = True)
    out = out.sort_values('period', ascending = True)

    return out

def welch(data, t_col, var, period_range = [10, 36], alpha = 0.01, **kwargs):

    id = data.name
    start = 60*60*period_range[0]
    end = 60*60*period_range[1]
    t, y =  data[t_col].to_numpy(), data[var].to_numpy()

    dt = t[1] - t[0]
    period, power = welch(y, fs = 1/dt)
    long_period = 1 / period  
    period = long_period[(long_period > start) & (long_period < end)]
    period = period / (60*60)
    short_power = power[(long_period > start) & (long_period < end)]

    sig_thresh = -np.mean(power) * np.log((1 - ((1 - alpha) ** (1 / len(power)))))
    out = pd.DataFrame(data = {'id' : id, 'period' : period, 'power' : short_power, 'sig_threshold' : sig_thresh})    
    
    out.set_index('id', inplace = True)
    out = out.sort_values('period', ascending = True)

    return out

def wavelet(data, t_col, var, scale = 156, wavelet_type = 'morl', **kwargs):

    scales = np.arange(1, scale)
    t, y =  data[t_col].to_numpy(), data[var].to_numpy()
    dt = t[1] - t[0]
    dt = dt / (60*60)
    [coefficients, frequencies] = cwt(y, scales, wavelet_type, dt)
    power = (abs(coefficients)) ** 2
    period = 1/frequencies
    return t/(60*60), np.log2(period), np.log2(power)
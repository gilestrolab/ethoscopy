import pandas as pd
import numpy as np 

from pywt import cwt

from scipy.stats import chi2
from scipy.special import chdtri
# from scipy.signal import welch as internal_welch
from scipy.fftpack import fft

from astropy.timeseries import LombScargle

def chi_squared(data: pd.DataFrame, t_col: str, var: str, 
                period_range: list = [10, 36], freq: float = 1/60, 
                alpha: float = 0.01, time_resolution: float = 0.1 * (60*60)) -> pd.DataFrame:
    """
    Calculate Chi-squared periodogram for time series data.
    
    Implements an efficient Chi-squared periodogram calculation using vectorized operations.
    Detects periodic patterns in behavioral data by analyzing power across different periods.

    Args:
        data (pd.DataFrame): Input DataFrame containing time series data
        t_col (str): Name of column containing timestamps
        var (str): Name of column containing variable to analyze
        period_range (list, optional): Min and max periods to analyze in hours. Default [10, 36]
        freq (float, optional): Sampling frequency in Hz. Default 1/60
        alpha (float, optional): Significance level for threshold calculation. Default 0.01
        time_resolution (float, optional): Time resolution in seconds. Default 0.1 * (60*60)

    Returns:
        pd.DataFrame: DataFrame containing:
            - period: Analyzed periods in hours
            - power: Power at each period
            - sig_threshold: Significance threshold
            - p_value: Statistical p-value
    """
    
    def calc_Qp(target_period: float, values: np.ndarray, freq: float) -> float:
        """Calculate Q_p statistic efficiently using numpy operations."""
        col_num = round(target_period * freq)
        row_num = len(values) / col_num
        repeat_int = round((len(values)-1) / col_num) + 1
        
        # Vectorized operations
        mod_col = np.tile(np.arange(1, col_num+1), repeat_int)[:len(values)]
        avg_P = pd.Series(values).groupby(mod_col).mean().to_numpy()
        avg_all = np.mean(values)
        
        # Efficient calculation using numpy
        numerator = np.sum((avg_P - avg_all)**2) * (len(values) * row_num)
        denominator = np.sum((values - avg_all)**2)
        
        return numerator / denominator

    # Extract data efficiently
    id_val = data.name
    start, end = 60*60*period_range[0], 60*60*period_range[1]
    t, y = data[t_col].to_numpy(), data[var].to_numpy()
    
    # Pre-allocate periods array
    periods = np.arange(start, end + time_resolution, time_resolution)
    
    # Create output DataFrame efficiently
    out = pd.DataFrame({
        'id': id_val,
        'period': periods,
        'power': np.array([calc_Qp(p, y, freq) for p in periods])
    })
    
    # Calculate thresholds efficiently
    round_period = np.round(out['period'].to_numpy() * freq)
    corrected_alpha = 1 - (1 - alpha)**(1/len(out))
    out['sig_threshold'] = chdtri(round_period, corrected_alpha)
    out['p_value'] = 1 - chi2.cdf(out['power'], out['period']*freq)
    out['period'] = out['period'] / (60*60)
    
    return out.set_index('id').sort_values('period', ascending=True)

def lomb_scargle(data: pd.DataFrame, t_col: str, var: str, 
                 period_range: list = [10, 36], alpha: float = 0.01, 
                 **kwargs) -> pd.DataFrame:
    """
    Calculate Lomb-Scargle periodogram for unevenly sampled time series.
    
    Implements the Lomb-Scargle algorithm for detecting periodic signals in behavioral data.
    Particularly useful for data with irregular sampling intervals.

    Args:
        data (pd.DataFrame): Input DataFrame containing time series data
        t_col (str): Name of column containing timestamps
        var (str): Name of column containing variable to analyze
        period_range (list, optional): Min and max periods to analyze in hours. Default [10, 36]
        alpha (float, optional): Significance level for threshold calculation. Default 0.01
        freq (float, optional): Ignored - included for API compatibility
        **kwargs: Additional arguments for astropy.LombScargle, such as:
            - normalization: Type of normalization ('standard', 'model', 'log', 'psd')
            - nterms: Number of terms in the Fourier fit
            - center_data: Whether to center the data before fitting
            - fit_mean: Whether to fit for the mean of the data

    Returns:
        pd.DataFrame: DataFrame containing:
            - period: Analyzed periods in hours
            - power: Power at each period
            - sig_threshold: Significance threshold
    """
    id_val = data.name
    start, end = 60*60*period_range[0], 60*60*period_range[1]
    t, y = data[t_col].to_numpy(), data[var].to_numpy()
    
    # Remove freq from kwargs if present
    if 'freq' in kwargs:
        del kwargs['freq']
        
    # Pass kwargs to LombScargle constructor
    ls = LombScargle(t, y, **kwargs)
    period, power = ls.autopower(
        minimum_frequency=1/end,
        maximum_frequency=1/start,
        samples_per_peak=20
    )
    
    period = (1/period) / (60*60)
    return pd.DataFrame({
        'id': id_val,
        'period': period,
        'power': power,
        'sig_threshold': ls.false_alarm_level(alpha)
    }).set_index('id').sort_values('period', ascending=True)

def fourier(data: pd.DataFrame, t_col: str, var: str, 
            period_range: list = [10, 36], alpha: float = 0.01, **kwargs) -> pd.DataFrame:
    """
    Calculate Fourier transform periodogram for time series data.
    
    Implements Fast Fourier Transform (FFT) for spectral analysis of behavioral data.
    Efficient for detecting periodic patterns in regularly sampled data.

    Args:
        data (pd.DataFrame): Input DataFrame containing time series data
        t_col (str): Name of column containing timestamps
        var (str): Name of column containing variable to analyze
        period_range (list, optional): Min and max periods to analyze in hours. Default [10, 36]
        alpha (float, optional): Significance level for threshold calculation. Default 0.01
        **kwargs: Additional arguments for FFT calculation, such as:
            - window: Window function to apply ('hamming', 'hanning', etc.)
            - detrend: Type of detrending (None, 'constant', 'linear')
            - pad_to: Number of points to pad the signal to

    Returns:
        pd.DataFrame: DataFrame containing:
            - period: Analyzed periods in hours
            - power: Power at each period
            - sig_threshold: Significance threshold
    """
    id_val = data.name
    start, end = 60*60*period_range[0], 60*60*period_range[1]
    t, y = data[t_col].to_numpy(), data[var].to_numpy()
    
    # Remove freq from kwargs if present
    if 'freq' in kwargs:
        del kwargs['freq']   

    # Apply optional window function
    if 'window' in kwargs:
        from scipy.signal import get_window
        window = get_window(kwargs['window'], len(y))
        y = y * window
    
    # Apply optional detrending
    if 'detrend' in kwargs:
        from scipy.signal import detrend
        y = detrend(y, type=kwargs['detrend'])
    
    # Calculate FFT with optional padding
    N = kwargs.get('pad_to', len(y))
    dt = t[1] - t[0]
    period = np.delete(np.linspace(0.0, 1.0/(2.0*dt), N//2), 0)
    power = 2.0/N * np.abs(fft(y, n=N)[1:N//2])
    
    # Efficient period filtering
    long_period = 1/period
    mask = (long_period > start) & (long_period < end)
    period = long_period[mask] / (60*60)
    
    sig_thresh = -np.mean(power) * np.log(1 - (1 - alpha) ** (1/len(power)))
    
    return pd.DataFrame({
        'id': id_val,
        'period': period,
        'power': power[mask],
        'sig_threshold': sig_thresh
    }).set_index('id').sort_values('period', ascending=True)

## Welch doesn't work as expected

# def welch(data: pd.DataFrame, t_col: str, var: str, 
#           period_range: list = [10, 36], alpha: float = 0.01, **kwargs) -> pd.DataFrame:
#     """
#     Calculate Welch's periodogram for time series data.
    
#     Implements Welch's method for estimating power spectral density.
#     Reduces noise in spectral estimation compared to standard periodogram.

#     Args:
#         data (pd.DataFrame): Input DataFrame containing time series data
#         t_col (str): Name of column containing timestamps
#         var (str): Name of column containing variable to analyze
#         period_range (list, optional): Min and max periods to analyze in hours. Default [10, 36]
#         alpha (float, optional): Significance level for threshold calculation. Default 0.01
#         **kwargs: Additional arguments for scipy.signal.welch, such as:
#             - window: Window function ('hanning' by default)
#             - nperseg: Length of each segment
#             - noverlap: Number of points to overlap between segments
#             - scaling: Spectrum scaling ('density' or 'spectrum')

#     Returns:
#         pd.DataFrame: DataFrame containing:
#             - period: Analyzed periods in hours
#             - power: Power at each period
#             - sig_threshold: Significance threshold
#     """
#     id_val = data.name
#     start, end = 60*60*period_range[0], 60*60*period_range[1]
#     t, y = data[t_col].to_numpy(), data[var].to_numpy()
    
#     # Remove freq from kwargs if present
#     if 'freq' in kwargs:
#         del kwargs['freq']
    
#     # Calculate Welch periodogram with kwargs
#     dt = t[1] - t[0]
#     period, power = internal_welch(y, fs=1/dt, **kwargs)
#     period, power = np.delete(period, 0), np.delete(power, 0)
    
#     long_period = 1/period
#     mask = (long_period > start) & (long_period < end)
#     period = long_period[mask] / (60*60)
    
#     sig_thresh = -np.mean(power) * np.log(1 - (1 - alpha) ** (1/len(power)))
    
#     return pd.DataFrame({
#         'id': id_val,
#         'period': period,
#         'power': power[mask],
#         'sig_threshold': sig_thresh
#     }).set_index('id').sort_values('period', ascending=True)

def wavelet(data: pd.DataFrame, t_col: str, var: str, scale: int = 156, 
            wavelet_type: str = 'morl', **kwargs) -> tuple:
    """
    Calculate continuous wavelet transform for time series data.
    
    Implements wavelet analysis for time-frequency decomposition of behavioral data.
    Useful for detecting localized periodic patterns and their evolution over time.

    Args:
        data (pd.DataFrame): Input DataFrame containing time series data
        t_col (str): Name of column containing timestamps
        var (str): Name of column containing variable to analyze
        scale (int, optional): Maximum scale for wavelet transform. Default 156
        wavelet_type (str, optional): Type of wavelet to use. Default 'morl'
        **kwargs: Additional arguments for pywt.cwt, such as:
            - method: Method to use ('conv' or 'fft')
            - axis: Axis over which to compute CWT
            - sampling_period: Sampling period for frequencies output

    Returns:
        tuple: Contains:
            - time: Time points in hours
            - period: Log2 of analyzed periods
            - power: Log2 of power values at each time-period point
    """
    scales = np.arange(1, scale)
    t, y = data[t_col].to_numpy(), data[var].to_numpy()
    dt = t[1] - t[0]
    dt /= (60*60)
    
    # Calculate wavelet transform with kwargs
    coefficients, frequencies = cwt(y, scales, wavelet_type, dt, **kwargs)
    power = np.log2((abs(coefficients)) ** 2)
    period = np.log2(1/frequencies)
    
    return t/(60*60), period, power
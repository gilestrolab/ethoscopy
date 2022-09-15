import plotly.graph_objs as go 
import warnings
from ethoscopy.misc.format_warning import format_warning
warnings.formatwarning = format_warning
from itertools import cycle

def fancy_range(start, stop, steps=(1,)):
    """
    Creat a range that has alternating step values
    """
    steps = cycle(steps)
    val = start
    while val < stop:
        yield val
        val += next(steps)

def make_bars(bar, bar_col, size):
    shaped_bar = go.layout.Shape(type="rect", 
                                                    x0=bar, 
                                                    y0=-0.025, 
                                                    x1=bar+size, 
                                                    y1=0, 
                                                    line=dict(
                                                        color="black", 
                                                        width=1) ,
                                                    fillcolor=bar_col
                                                )
    return shaped_bar

def circadian_bars(t_min, t_max, circadian_night = 12):
    """ 
    create boxes within plotly to represent the light, dark phases in light sensitive experiments
    @t_min = int, the minimum time point as a multiple of 12 
    @t_max = int, the maximum time point as a multiple of 12
    @circadian_night = int, the hour the lights turn off, must be between 1 and 23
    """

    if circadian_night < 1 or circadian_night > 23:
        warnings.warn("The arugment for circadian_night must be between 1 and 23")
        exit()

    # Light-Dark annotaion bars
    bar_shapes = {}

    if (t_min - circadian_night) % 12 == 0:
        used_range = fancy_range(t_min, t_max, (24-circadian_night, circadian_night))
    else:
        used_range = fancy_range(t_min, t_max, (circadian_night, 24-circadian_night))

    for i, bars in enumerate(used_range):

        if bars % 24 == 0:
            white_bar = make_bars(bar = bars, bar_col = 'white', size = circadian_night)
            bar_shapes[f'shape_{i}'] = white_bar
        else:
            black_bar = make_bars(bar = bars, bar_col = 'black', size = 24 - circadian_night)
            bar_shapes[f'shape_{i}'] = black_bar


    return bar_shapes
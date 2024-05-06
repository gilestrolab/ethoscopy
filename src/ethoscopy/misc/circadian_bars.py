import plotly.graph_objs as go 
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

def make_bars(bar, bar_col, size, split, y_size1, y_size2):

    shaped_bar = go.layout.Shape(type="rect", 
                                x0=bar, 
                                y0= y_size1, 
                                x1=bar+size, 
                                y1= y_size2,
                                line=dict(
                                    color="black", 
                                    width=0.5) ,
                                fillcolor=bar_col
                            )
    if split != 0:
        shaped_bar['xref'] = f'x{split+1}'
        shaped_bar['yref'] = f'y{split+1}'

    return shaped_bar

def circadian_bars(t_min, t_max, min_y, max_y, day_length = 24, lights_off = 12, split = False, canvas = 'plotly'):
    """ 
    create boxes within plotly to represent the light, dark phases in light sensitive experiments
    @t_min = int, the minimum time point as a multiple of 12 
    @t_max = int, the maximum time point as a multiple of 12
    @circadian_night = int, the hour the lights turn off, must be between 1 and 23
    """
    if split != False:
        scale = 20
    else:
        scale = 40
        split = 1

    y_size1 = min_y
    y_size2 = min_y - ((max_y-min_y) / scale)   

    if lights_off < 1 or lights_off > day_length:
        raise ValueError(f"The argument for lights_off must be between 1 and {day_length}")

    # Light-Dark annotaion bars
    bar_shapes = {}

    if (t_min - lights_off) % (day_length/2) == 0:
        used_range = fancy_range(t_min, t_max, (day_length-lights_off, lights_off))
    else:
        used_range = fancy_range(t_min, t_max, (lights_off, day_length-lights_off))

    if canvas == 'seaborn':
        size = (max_y-min_y) / scale
        return used_range, size

    for i, bars in enumerate(used_range):
        for c in range(split):
            if bars % day_length == 0:
                white_bar = make_bars(bar = bars, bar_col = 'white', size = lights_off, split = c, y_size1 = y_size1, y_size2 = y_size2)
                bar_shapes[f'shape_{i}{c}'] = white_bar
            else:
                black_bar = make_bars(bar = bars, bar_col = 'black', size = day_length - lights_off, split = c, y_size1 = y_size1, y_size2 = y_size2)
                bar_shapes[f'shape_{i}{c}'] = black_bar

    return bar_shapes, y_size2
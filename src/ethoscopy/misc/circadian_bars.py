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
    """ The function to generate plotly shapes for the boxes """
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

def circadian_bars(t_min:int|float, t_max:int|float, min_y:int|float, max_y:int|float, day_length:int|float = 24, 
    lights_off:int|float = 12, split:bool|int = False, canvas:str = 'plotly'):
    """ 
    Generates boxes that indicate when the lights are ona dn off for both Plotly and Seaborn.
    Typically used internally for use with overtime plots.

        Args:
            t_min (int | float): the minimum value of t (time) as a function of the lights_off interval
            t_mmax (int | float): the maximum value of t (time) as a function of the lights_off interval
            min_y (int | float): the minimum value of the y-axis to scale the height of the boxes
            max_y (int | float): the maximum value of the y-axis to scale the height of the boxes
            day_length (int | float): the length (in hours) of the experimental day.
                The combined length of both the white (day) and black (night) boxes will this length.
                Default is 24.
            lights_off (int | float): the timepoint (in hours) when the lights go off in an experimental day.
                The length of both the white (day) box will be this length. The black (night) will be 
                day_length - lights_off.
                Default is 12.
            split (bool | int, optional): If not False then must be an int that corresponds to the number 
                of subplots in the figure. Default is False.
            canvas (str, optional); The canvas type the boxes are generated for, either Plotly or Seaborn.
                Default is 'plotly'.

    Returns:
        If canvas == 'plotly':
            A dictionary containing plotly shape objects
        If canvas == 'seaborn':
            used_range - a list of the x values of the boxes
            size - the height of the boxes
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
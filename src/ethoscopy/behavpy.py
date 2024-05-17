from ethoscopy.behavpy_core import behavpy_core
from ethoscopy.behavpy_plotly import behavpy_plotly
from ethoscopy.behavpy_seaborn import behavpy_seaborn

def behavpy(data, meta, palette = None, long_palette = None, check = False, index= None, columns=None, dtype=None, copy=True, canvas='plotly'):

    if canvas == 'plotly':

        # If no palette is privided choose the defaults
        if palette is None:
            palette = 'Safe'
        if long_palette is None:
            long_palette = 'Dark24'

        return behavpy_plotly(data, meta, palette, long_palette, check, index, columns, dtype, copy)

    elif canvas == 'seaborn':

        # If no palette is privided choose the defaults
        if palette is None:
            palette = 'deep'
        if long_palette is None:
            long_palette = 'deep'
        if palette is not None and long_palette is None:
            long_palette = palette

        return behavpy_seaborn(data, meta, palette, long_palette, check, index, columns, dtype, copy)

    elif canvas == None:
        return behavpy_core(data, meta, palette, long_palette, check, index, columns, dtype, copy)

    else:
        raise ValueError('Invalid canvas specified')
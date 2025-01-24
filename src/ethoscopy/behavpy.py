from ethoscopy.behavpy_core import behavpy_core
from ethoscopy.behavpy_plotly import behavpy_plotly
from ethoscopy.behavpy_seaborn import behavpy_seaborn

def behavpy(data, meta, canvas='seaborn', palette = None, long_palette = None, check = False, index= None, columns=None, dtype=None, copy=True):
    """
    Factory function that creates a behavpy object with the specified visualisation backend.

    The behavpy class extends pandas DataFrame to provide specialised functionality for handling and analysing
    behavioural data from ethoscope experiments. It maintains a link between experimental data and corresponding 
    metadata, with methods for data manipulation, analysis, and visualisation.

    Args:
        data (pd.DataFrame): Experimental data, typically loaded via load_ethoscope(). Must contain:
            - 'id' column with unique specimen IDs
            - 't' column with timestamps in seconds
        meta (pd.DataFrame): Metadata containing experimental conditions, genotypes etc. Must have:
            - One row per unique specimen ID
            - IDs matching those in the data DataFrame
        palette (str, optional): Color palette name for visualizations with ≤11 groups.
            Defaults to 'Safe' for Plotly and 'deep' for Seaborn.
        long_palette (str, optional): Color palette name for visualizations with >11 groups.
            Defaults to 'Dark24' for Plotly and 'deep' for Seaborn.
        check (bool, optional): If True, validates that all data IDs exist in metadata and removes
            redundant columns from link_meta_index. Defaults to False.
        index (pd.Index, optional): Index to use for the DataFrame. Defaults to None.
        columns (pd.Index, optional): Column labels to use for the DataFrame. Defaults to None.
        dtype (np.dtype, optional): Data type to force. Defaults to None.
        copy (bool, optional): Copy data from inputs. Defaults to True.
        canvas (str, optional): Visualisation backend to use - 'plotly', 'seaborn', or None. Defaults to 'plotly'.

    Returns:
        behavpy_core: A behavpy object with methods for manipulating, analysing and plotting
            time series behavioural data, using the specified visualisation backend.

    Raises:
        ValueError: If an invalid canvas type is specified.
    """

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
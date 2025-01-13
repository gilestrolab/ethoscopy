from ethoscopy.behavpy_core import behavpy_core

class behavpy_HMM(behavpy_core):
    """
    A compatibility class that inherits from behavpy_core to support loading data from versions prior to 2.0.0.
    
    This class allows loading of legacy pickled behavpy data files while maintaining core functionality.
    Note that plotting methods are not available in the initial loaded state.

    To access plotting functionality, the data must be re-instantiated with either the plotly or 
    seaborn canvas option.

    Both HMM and circadian classes have been folded into the one class as of 2.0.0.

    Example
    -------
    >>> old_hmm_df = pd.read_pickle('path_to_data')
    >>> new_df = etho.behavpy(old_df, old_df.meta, check=True, canvas='seaborn')
    """
    # set meta as permenant attribute
    _metadata = ['meta']
    _canvas = None
    _hmm_colours = None
    _hmm_labels = None


    @property
    def _constructor(self):
        return behavpy_core._internal_constructor(self.__class__)

    class _internal_constructor(object):
        def __init__(self, cls):
            self.cls = cls

        def __call__(self, *args, **kwargs):
            kwargs['meta'] = None
            return self.cls(*args, **kwargs)

        def _from_axes(self, *args, **kwargs):
            return self.cls._from_axes(*args, **kwargs)

    def __init__(self, data, meta, palette = None, long_palette = None, check = False, index= None, columns=None, dtype=None, copy=True):
        super(behavpy_core, self).__init__(data=data,
                                        index=index,
                                        columns=columns,
                                        dtype=dtype,
                                        copy=copy)

        self.meta = meta  
        self.attrs = {'sh_pal' : palette, 'lg_pal' : long_palette}
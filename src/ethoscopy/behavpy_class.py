from ethoscopy.behavpy_core import behavpy_core

class behavpy(behavpy_core):
    """
    An inheritor class to aid with backwards compatability prior to version 2.0.0, so that old saved pickles of data can still be loaded

    When old data is loaded it will only have the core functionalities  no plotting methods. The user must re-intiated either a plotly or
    seaborn behavpy class with the data and metadata, i.e.

    old_df = pd.read_pickle('path_to_data')
    new_df = etho.behavpy(old_df, old_df.meta, check = True, canvas = 'seaborn')
    
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
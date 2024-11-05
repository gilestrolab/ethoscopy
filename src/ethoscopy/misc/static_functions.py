import pandas as pd

def concat(*args):
    """
    Wrapper for pd.concat that also concats metadata of multiple behavpy objects

    Args:
        args (behvapy): Behavpy tables to be concatenated to the original behavpy table, each behavpy object should be entered as its own argument and not a list.

    returns:
        A new instance of a combined behavpy object. Palette selection comes from the first dataframe.

    Example:
        etho.concat(df1, df2, ...)
        # if a list, unpack with *
        etho.concat(*[df1, df2, df3])
    """

    meta_list = []
    data_list = []

    class_type = args[0].__class__

    for df in args:

        if isinstance(df, class_type) is not True:
            raise TypeError('Object(s) to concat are not the same behavpy class')

        meta_list.append(df.meta)
        data_list.append(df)

    meta = pd.concat(meta_list)
    new = pd.concat(data_list)

    try:
        new = class_type(new, meta, palette=args[0].attrs['sh_pal'], long_palette=args[0].attrs['lg_pal'], check = True)
    except KeyError:
        new = class_type(new, meta, check = True)
    
    return new
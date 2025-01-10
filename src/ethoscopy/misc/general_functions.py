import pandas as pd

def concat(*args):
    """
    Concatenates multiple behavpy objects while preserving metadata and attributes.
    
    Args:
        *args (behavpy): Behavpy tables to concatenate, each as a separate argument
            or unpacked from a list.
    
    Returns:
        behavpy: A new combined behavpy object with merged metadata and preserved attributes.
    
    Example:
        etho.concat(df1, df2, df3)
        # or with a list
        etho.concat(*[df1, df2, df3])
    """
    if not args:
        raise ValueError("At least one behavpy object required for concatenation")
    
    # Get class and validate all inputs are the same type
    class_type = args[0].__class__
    if not all(isinstance(df, class_type) for df in args):
        raise TypeError('All objects must be the same behavpy class')
    
    meta = pd.concat([df.meta for df in args])
    data = pd.concat(args)
    
    # Get palette attributes from first dataframe if they exist
    attrs = {}
    first_df = args[0]
    if hasattr(first_df, 'attrs'):
        if 'sh_pal' in first_df.attrs:
            attrs['palette'] = first_df.attrs['sh_pal']
        if 'lg_pal' in first_df.attrs:
            attrs['long_palette'] = first_df.attrs['lg_pal']
    
    # Create new instance with metadata and preserved attributes
    return class_type(data, meta, check=True, **attrs)
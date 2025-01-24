import pandas as pd
import numpy as np

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

def bootstrap(data: np.ndarray, n: int = 1000, func: callable = np.mean, confidence_interval: float = 0.95) -> tuple:
    """ 
    Generate n bootstrap samples and evaluate confidence intervals.
    
    Uses resampling with replacement to estimate confidence intervals for a given statistic.

    Args:
        data (np.ndarray): Array of data to be bootstrapped
        n (int, optional): Number of bootstrap iterations. Default is 1000.
        func (callable, optional): Function to compute on resampled data. Default is np.mean.
    
    Returns:
        tuple: Lower and upper 95% confidence interval bounds
    """
    # Pre-allocate array instead of using list append
    simulations = np.zeros(n)
    sample_size = len(data)
    
    # Vectorize the random sampling for all iterations at once
    random_indices = np.random.choice(sample_size, size=(n, sample_size), replace=True)
    
    # Calculate all simulations
    for i in range(n):
        simulations[i] = func(data[random_indices[i]])
    
    # Sort in-place 
    simulations.sort()

    ci = confidence_interval
    # Calculate the lower and upper bounds for the 95% confidence interval
    l_indx = int(np.floor(n * ((1-ci)/2) ))  # (1-0.95)/2 = 0.025
    u_indx = int(np.floor(n * ((1+ci)/2) ))  # (1+0.95)/2 = 0.975
    
    return (simulations[l_indx], simulations[u_indx])

def rle(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find runs of consecutive items in an array.
    
    Identifies continuous sequences of identical values and their properties.
    Optimized for performance using numpy vectorization.

    Args:
        x (np.ndarray): Array containing runs of data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
            - run values (unique values in runs)
            - run starts (starting indices)
            - run lengths (duration of each run)

    Raises:
        ValueError: If input array is not 1-dimensional
    """
    # Ensure array and validate dimension
    x = np.asarray(x)  # Specify dtype for better performance
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    
    # Handle empty array efficiently
    n = len(x)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Find run boundaries using vectorized operations
    # Pre-allocate with correct size and dtype
    loc_run_start = np.ones(n, dtype=bool)
    loc_run_start[1:] = x[1:] != x[:-1]
    
    # Get run properties using efficient numpy operations
    run_starts = np.nonzero(loc_run_start)[0]
    run_values = x[run_starts]
    
    # Calculate run lengths using vectorized subtraction
    run_lengths = np.empty_like(run_starts)
    run_lengths[:-1] = run_starts[1:] - run_starts[:-1]
    run_lengths[-1] = n - run_starts[-1]
    
    return run_values, run_starts, run_lengths
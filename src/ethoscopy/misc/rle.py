import numpy as np

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
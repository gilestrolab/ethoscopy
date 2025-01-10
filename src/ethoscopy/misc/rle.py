import numpy as np 

def rle(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find runs of consecutive items in an array.
    
    Identifies continuous sequences of identical values and their properties.

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

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
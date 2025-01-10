import numpy as np 

def bootstrap(data: np.ndarray, n: int = 1000, func: callable = np.mean) -> tuple:
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
    
    # Calculate all simulations at once using vectorization
    for i in range(n):
        simulations[i] = func(data[random_indices[i]])
    
    # Sort in-place instead of creating new array
    simulations.sort()

    # Move ci function outside to avoid recreation in each call
    l_indx = int(np.floor(n * 0.025))  # (1-0.95)/2 = 0.025
    u_indx = int(np.floor(n * 0.975))  # (1+0.95)/2 = 0.975
    
    return (simulations[l_indx], simulations[u_indx])

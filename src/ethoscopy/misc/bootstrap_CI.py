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
    simulations = list()
    sample_size = len(data)
    # xbar_init = np.mean(data)
    
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()

    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx] , simulations[u_indx])

    return(ci(0.95))

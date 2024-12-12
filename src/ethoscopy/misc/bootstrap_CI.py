import numpy as np 

def bootstrap(data, n=1000, func=np.mean):
    """ 
    Generate n bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest

        Args:
            data (np.array): The numpy array of data to be bootstrapped.
            n (int, optional): The number of iterations of the simulation.
                Default is 1000.
            func (np.function, optional): The function to find average of 
                all simulation outputs. Default is numpy mean.
    
    Returns:
        a tuple containing the 95% confidence intervals
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

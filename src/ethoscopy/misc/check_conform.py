import warnings
import pandas as pd
from sys import exit

from ethoscopy.misc.format_warning import format_warning

def check_conform(data, metadata = None, skip = False):
    """ 
    Checks the data augument is a pandas dataframe
    If metadata is provided and skip is False it will check as above and check the ID's in
    metadata match those in the data
    params: 
    @data / @metadata = can be any class but will force an exit if not a pandas dataframe
    @skip = boolean indicating whether to skip a check that unique id's are in both meta and data match 
    """
    
    # formats warming method to not double print and allow string formatting
    warnings.formatwarning = format_warning

    if isinstance(data, pd.DataFrame) is not True:
        warnings.warn('Data input is not a pandas dataframe')
        exit()

    if metadata is not None: 
        if isinstance(metadata, pd.DataFrame) is not True:
            warnings.warn('Metadata input is not a pandas dataframe')
            exit()

        if skip is False:
            metadata_id_list = metadata.index.tolist()
            data_id_list = set(data.index.tolist())
            # checks if all id's of data are in the metadata dataframe
            check_data = all(elem in metadata_id_list for elem in data_id_list)
            if check_data is not True:
                warnings.warn("There are ID's in the data not in the metadata, please check")
                exit()
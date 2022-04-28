import warnings
import pandas as pd
from behavpy_class import behavpy

from ethoscopy.misc.format_warning import format_warning
from ethoscopy.misc.format_warning import format_warning

def set_behavpy(metadata, data, skip = False):
    """ 
    Takes two data frames, one metadata and the other the recorded values and makes an instance of the behavpy class
    Data becomes the pandas dataframe with metadata added as an attribute from a subclassed version of pd.DataFrame
    must both contain an 'id' column with matching ids
    params: 
    @ metadata : pandas dataframe object containing the meta information from an ethoscope experiment as processed by the function link_meta_index()
    @ data : pandas dataframe object containing the data from an ethoscope experiment as processed by the function load_ethoscope()
    @ skip : boolean indicating whether to skip a check that unique id's are in both meta and data match
    """
        
    warnings.formatwarning = format_warning

    # check both metadata and data for an id column and set it as the index if found and is not
    if metadata.index.name != 'id':
        try:
            metadata.set_index('id', inplace = True)
        except:
            warnings.warn('There is no "id" as a column or index in the metadata')
            exit()

    if data.index.name != 'id':
        try:
            data.set_index('id', inplace = True)
        except:
            warnings.warn('There is no "id" as a column or index in the data')
            exit()

    # checks both metadata and data are pandas dataframes, also checks if unique id's in meta and data match 
    check_conform(data, metadata, skip = skip)

    # drop some left over columns from loading that arean't needed in analysis
    drop_col_names = ['path', 'file_name', 'file_size', 'machine_id']
    metadata.drop(columns=[col for col in metadata if col in drop_col_names], axis = 1, inplace =True)

    # make instance of behavpy class and set metadata as attribute
    df = behavpy(data)
    df.meta = metadata

    return df

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
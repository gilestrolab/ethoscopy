import warnings
from sys import exit
from ethoscopy.behavpy_class import behavpy
from ethoscopy.misc.format_warning import format_warning
from ethoscopy.misc.check_conform import check_conform

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
    metadata.drop(columns=[col for col in metadata if col in drop_col_names], inplace =True)

    # make instance of behavpy class and set metadata as attribute
    df = behavpy(data)
    df.meta = metadata

    return df
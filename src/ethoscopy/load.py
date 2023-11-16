import ftplib
import os
import pandas as pd 
import numpy as np
import errno
import time 
import sqlite3
from pathlib import Path, PurePosixPath, PurePath
from functools import partial
from urllib.parse import urlparse

from ethoscopy.misc.validate_datetime import validate_datetime

pd.options.mode.chained_assignment = None

def download_from_remote_dir(meta, remote_dir, local_dir):
    """ 
    This function is used to import data from the ethoscope node platform to your local directory for later use. The ethoscope files must be saved on a
    remote FTP server and saved as .db files, see the Ethoscope manual for how to setup a node correctly
    https://www.notion.so/giorgiogilestro/Ethoscope-User-Manual-a9739373ae9f4840aa45b277f2f0e3a7
    
    Args:
        meta (str): The path to a csv file containing columns with machine_name, date, and time if multiple files on the same day
        remote_dir (str): The url containing the location of the ftp server up to the folder contain the machine id's, server must not have a username or password (anonymous login)
            e.g. 'ftp://YOUR_SERVER//auto_generated_data//ethoscope_results'
        local_dir (str): The path of the local directory to save .db files to, files will be saved using the structure of the ftp server
            e.g. 'C:\\Users\\YOUR_NAME\\Documents\\ethoscope_databases'

    returns None
    """
    meta = Path(meta)
    local_dir = Path(local_dir)

    #check csv path is real and read to pandas df
    if meta.exists():
        try:
            meta_df = pd.read_csv(meta)         
        except Exception as e:
            print("An error occurred: ", e)
    else:
        raise FileNotFoundError("The metadata is not readable")

    # check and tidy df, removing un-needed columns and duplicated machine names
    if 'machine_name' not in meta_df.columns or 'date' not in meta_df.columns:
        raise KeyError("Column(s) 'machine_name' and/or 'date' missing from metadata file")

    meta_df.dropna(how = 'all', inplace = True)

    if 'time' in meta_df.columns.tolist():
        meta_df['check'] = meta_df['machine_name'] + meta_df['date'] + meta_df['time']
        meta_df.drop_duplicates(subset = ['check'], keep = 'first', inplace = True, ignore_index = False)
    else:
        meta_df['check'] = meta_df['machine_name'] + meta_df['date'] 
        meta_df.drop_duplicates(subset = ['check'], keep = 'first', inplace = True, ignore_index = False)

    # check the date format is YYYY-MM-DD, without this format the df merge will return empty
    # will correct to YYYY-MM-DD in a select few cases
    validate_datetime(meta_df)

    # extract columns as list to identify .db files from ftp server
    ethoscope_list = meta_df['machine_name'].tolist()
    date_list = meta_df['date'].tolist()

    if 'time' in meta_df.columns.tolist():
        time_list = pd.Series(meta_df['time'].tolist())
        bool_list = time_list.isna().tolist()
    else:
        nan_list = [np.nan] * len(meta_df['date'])
        time_list = pd.Series(nan_list)
        bool_list = time_list.isna().tolist()

    # connect to ftp server and parse the given ftp link
    parse = urlparse(remote_dir)
    ftp = ftplib.FTP(parse.netloc)
    ftp.login()
    ftp.cwd(parse.path)
    files = ftp.nlst()

    paths = []
    check_list = []
    # iterate through the first level of directories looking for ones that match the ethoscope names given, 
    # find the susequent files that match the date and time and add to paths list
    # this is slow, should change to walk directory once, get all information and then match to csv

    for dir in files:
        temp_path = parse.path / PurePosixPath(dir)
        try:
            ftp.cwd(str(temp_path))
            directories_2 = ftp.nlst()
            for c, name in enumerate(ethoscope_list):
                if name in directories_2:
                    temp_path_2 = temp_path / PurePosixPath(name)
                    ftp.cwd(str(temp_path_2))
                    directories_3 = ftp.nlst()
                    for exp in directories_3:
                        date_time = exp.split('_')
                        if date_time[0] == date_list[c]:
                            if bool_list[c] is False:
                                if date_time[1] == time_list[c]:
                                    temp_path_3 = temp_path_2 / PurePosixPath(exp)
                                    ftp.cwd(str(temp_path_3))
                                    directories_4 = ftp.nlst()
                                    for db in directories_4:
                                        if db.endswith('.db'):
                                            size = ftp.size(db)
                                            final_path = f'{dir}/{name}/{exp}/{db}'
                                            path_size_list = [final_path, size]
                                            paths.append(path_size_list)
                                            check_list.append([name, date_time[0]])

                            else:
                                temp_path_3 = temp_path_2 / PurePosixPath(exp)
                                ftp.cwd(str(temp_path_3))
                                directories_4 = ftp.nlst()
                                for db in directories_4:
                                    if db.endswith('.db'):
                                        size = ftp.size(db)
                                        final_path = f'{dir}/{name}/{exp}/{db}'
                                        path_size_list = [final_path, size]
                                        paths.append(path_size_list)
                                        check_list.append([name, date_time[0]])
                                    
        except:
            continue

    if len(paths) == 0:
        raise RuntimeError("No Ethoscope data could be found, please check the metadata file")

    for i in zip(ethoscope_list, date_list):
        if list(i) in check_list:
            continue
        else:
            print(f'{i[0]}_{i[1]} has not been found for download')

    def download_database(remote_dir, folders, work_dir, local_dir, file_name, file_size):
        """ 
        Connects to remote FTP server and saves to designated local path, retains file name and path directory structure 
        
        Params:
        @remote_dir = ftp server netloc 
        @work_dir = ftp server path
        @local_dir = local directory path for the file and subsequent directory structure to be saved to
        @file_name = name of .db file to be download
        @file_size = size of file above in bytes

        returns None
        """
        
        #create local copy of directory tree from ftp server
        os.chdir(local_dir)

        win_path = local_dir / work_dir 
        
        try:
            os.makedirs(win_path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(win_path):
                pass
            else:
                raise

        file_path = win_path / file_name

        if os.access(file_path, os.R_OK):
            if os.path.getsize(file_path) < file_size:
                ftp = ftplib.FTP(remote_dir)
                ftp.login()
                ftp.cwd(folders + '/' + str(work_dir))

                localfile = open(file_path, 'wb')
                ftp.retrbinary('RETR ' + file_name, localfile.write)
                    
                ftp.quit()
                localfile.close()

        else:
            ftp = ftplib.FTP(remote_dir)
            ftp.login()
            ftp.cwd(folders + '/' + str(work_dir))

            localfile = open(file_path, 'wb')
            ftp.retrbinary('RETR ' + file_name, localfile.write)

            ftp.quit()
            localfile.close()

    # iterate over paths, downloading each file
    # provide estimate download time based upon average time of previous downloads in queue
    download = partial(download_database, remote_dir = parse.netloc, folders = parse.path, local_dir = local_dir)
    times = []

    for counter, j in enumerate(paths):
        print('Downloading {}... {}/{}'.format(j[0].split('/')[1], counter+1, len(paths)))
        if counter == 0:
            start = time.time()
            p = PurePosixPath(j[0])
            download(work_dir = p.parents[0], file_name = p.name, file_size = j[1])
            stop = time.time()
            t = stop - start
            times.append(t)

        else:
            av_time = round((np.mean(times)/60) * (len(paths)-(counter+1)))
            print(f'Estimated finish time: {av_time} mins') 
            start = time.time()
            p = PurePosixPath(j[0])
            download(work_dir = p.parents[0], file_name = p.name, file_size = j[1])
            stop = time.time()
            t = stop - start
            times.append(t)

def link_meta_index(metadata, local_dir):
    """ 
    A function to alter the provided metadata file with the path locations of downloaded .db files from the Ethoscope experimental system. The function will check all unique machines against the original ftp server 
    for any errors. Errors will be omitted from the returned metadata table without warning.

        Args:
            metadata (str): The path to a file containing the metadata information of each ROI to be downloaded, must include'ETHOSCOPE_NAME', 'date' in yyyy-mm-dd format or others (see validate_datetime), and 'region_id'
            local_dir (str): The path to the top level parent directory where saved database files are located.

    returns a pandas dataframe containing the csv file information and corresponding path for each entry in the csv 
    """
    metadata = Path(metadata)
    local_dir = Path(local_dir)
    #load metadata csv file
    #check csv path is real and read to pandas df
    if metadata.exists():
        try:
            meta_df = pd.read_csv(metadata) 
        except Exception as e:
            print("An error occurred: ", e)
    else:
        raise FileNotFoundError("The metadata is not readable")

    if len(meta_df[meta_df.isna().any(axis=1)]) >= 1:
        print(meta_df[meta_df.isna().any(axis=1)])
        raise ValueError("When the metadata is read it contained NaN values (empty cells in the csv file can cause this!), please replace with an alterative")

    # check and tidy df, removing un-needed columns and duplicated machine names
    if 'machine_name' not in meta_df.columns or 'date' not in meta_df.columns:
        raise KeyError("Column(s) 'machine_name' and/or 'date' missing from metadata file")

    meta_df.dropna(axis = 0, how = 'all', inplace = True)
    
    # check the date format is YYYY-MM-DD, without this format the df merge will return empty
    # will correct to YYYY-MM-DD in a select few cases
    meta_df = validate_datetime(meta_df)

    meta_df_original = meta_df.copy(deep = True)

    if 'time' in meta_df.columns.tolist():
        meta_df['check'] = meta_df['machine_name'] + meta_df['date'] + meta_df['time']
        meta_df.drop_duplicates(subset = ['check'], keep = 'first', inplace = True, ignore_index = False)
    else:
        meta_df['check'] = meta_df['machine_name'] + meta_df['date'] 
        meta_df.drop_duplicates(subset = ['check'], keep = 'first', inplace = True, ignore_index = False)

    ethoscope_list = meta_df['machine_name'].tolist()
    date_list = meta_df['date'].tolist()

    if 'time' in meta_df.columns.tolist():
        time_list = meta_df['time'].tolist()
    else:
        nan_list = [np.nan] * len(meta_df['date'])
        time_list = nan_list

    paths = []
    sizes = []
    for name, date, time in zip(ethoscope_list, date_list, time_list):
        try:
            if np.isnan(time):
                regex = PurePath('*') / name / f'{date}_*' / '*.db'
                path_lst = local_dir.glob(str(regex))
                if len(list(path_lst)) >= 1:
                    for p in local_dir.glob(str(regex)):
                        paths.append(p)
                        sizes.append(p.stat().st_size)
                else:
                    print(f'{name}_{date} has not been found')
            else:
                regex = PurePath('*') / name / f'{date}_{time}' / '*.db'
                path_lst = local_dir.glob(str(regex))
                if len(list(path_lst)) >= 1:
                    for p in local_dir.glob(str(regex)):
                        paths.append(p)
                        sizes.append(p.stat().st_size)

                else:
                    print(f'{name}_{date} has not been found')
        except TypeError:
            regex = PurePath('*') / name / f'{date}_{time}' / '*.db'
            path_lst = local_dir.glob(str(regex))
            if len(list(path_lst)) >= 1:
                for p in local_dir.glob(str(regex)):
                    paths.append(p)
                    sizes.append(p.stat().st_size)
            else:
                print(f'{name}_{date} has not been found')

    if len(paths) == 0:
        raise RuntimeError("No Ethoscope data could be found, please check the metatadata file")

    # split path into parts
    split_df = pd.DataFrame()
    for path, size in zip(paths, sizes):  
        split_path = str(path).replace(str(local_dir), '').split(os.sep)[1:]
        split_series = pd.DataFrame(data = split_path).T 
        split_series.columns = ['machine_id', 'machine_name', 'date_time', 'file_name']
        split_series['path'] = str(path)
        split_series['file_size'] = size
        split_df = pd.concat([split_df, split_series], ignore_index = True)

    #split the date_time column and add back to df
    split_df[['date', 'time']] = split_df.date_time.str.split('_', expand = True)
    split_df.drop(columns = ["date_time"], inplace = True)

    #merge df's
    if 'time' in meta_df_original.columns.tolist():
        merge_df = meta_df_original.merge(split_df, how = 'outer', on = ['machine_name', 'date', 'time'])
        merge_df.dropna(inplace = True)
    
    else:
        drop_df = split_df.sort_values(['file_size'], ascending = False)
        drop_df = drop_df.drop_duplicates(['machine_name', 'date'])
        droplog = split_df[split_df.duplicated(subset=['machine_name', 'date'])]
        drop_list = droplog['machine_name'].tolist()
        if len(drop_list) >= 1:
            warnings.warn(f'Ethoscopes {*drop_list,} have multiple files for their day, the largest file has been kept. If you want all files for that day please add a time column')
        merge_df = meta_df_original.merge(drop_df, how = 'outer', on = ['machine_name', 'date'])
        merge_df.dropna(inplace = True)

    # make the id for each row
    merge_df.insert(0, 'id', merge_df['file_name'].str.slice(0,26,1) + '|' + merge_df['region_id'].astype(int).map('{:02d}'.format))
    
    return merge_df

def load_ethoscope(metadata, min_time = 0 , max_time = float('inf'), reference_hour = None, cache = None, FUN = None, verbose = True):
    """
    The users function to iterate through the dataframe generated by link_meta_index() and load the corresponding database files 
    and analyse them according to the inputted function.

        Args:
            metadata (pd.DataFrame): The metadata datafframe as returned from link_meta_index function
            min_time (int): The minimum time you want to load data from with 0 being the experiment start (in hours), for all experiments. Default is 0.
            max_time (int): Same as above, but for the maximum time you want to load to. Default is infinity.
            reference_hour (int): The hour at which lights on occurs when the experiment is begun, or when you want the timestamps to equal 0. None equals the start of the experiment. Default is None.
            cache (str): The local path to find and store cached versions of each ROI per database. Directory tree structure is a mirror of ethoscope saved data. Cached files are in a pickle format. Default is None.
            FUN (function): A function to apply indiviual curatation to each ROI, typically using package generated functions (i.e. sleep_annotation). If using a user defined function use the package analyse functions as examples. 
                If None the data remains as found in the database. Default is None.
            verbose (bool): If True (defualt) then the function prints to screen information about each ROI when loading, if False no printing to screen occurs. Default is True.

    returns: 
        A pandas DataFrame object containing the database data and unique ids per fly as the index
    """  

    max_time = max_time * 60 * 60
    min_time = min_time * 60 * 60

    data = pd.DataFrame()

    # iterate over the ROI of each ethoscope in the metadata df
    for i in range(len(metadata.index)):
        try:
            if verbose is True:
                print('Loading ROI_{} from {}'.format(metadata['region_id'].iloc[i], metadata['machine_name'].iloc[i]))
            roi_1 = read_single_roi(file = metadata.iloc[i,:],
                                    min_time = min_time,
                                    max_time = max_time,
                                    reference_hour = reference_hour,
                                    cache = cache
                                    )

            if roi_1 is None:
                if verbose is True:
                    print('ROI_{} from {} was unable to load due to an error formatting roi'.format(metadata['region_id'].iloc[i], metadata['machine_name'].iloc[i]))
                continue

            if FUN is not None:
                roi_1 = FUN(roi_1) 

            if roi_1 is None:
                if verbose is True:
                    print('ROI_{} from {} was unable to load due to an error in applying the function'.format(metadata['region_id'].iloc[i], metadata['machine_name'].iloc[i]))
                continue
            roi_1.insert(0, 'id', metadata['id'].iloc[i])
            data = pd.concat([data, roi_1], ignore_index= True)
        except:
            if verbose is True:
                print('ROI_{} from {} was unable to load due to an error loading roi'.format(metadata['region_id'].iloc[i], metadata['machine_name'].iloc[i]))
            continue

    return data

def read_single_roi(file, min_time = 0, max_time = float('inf'), reference_hour = None, cache = None):
    """
    Loads the data from a single region from an ethoscope according to inputted times
    changes time to reference hour and applies any functions added
    
    Params: 
    @ file = row in a metadata pd.DataFrane containing a column 'path' with .db file Location
    @ min_time = time constraint with which to query database (in hours), default is 0
    @ max_time = same as above
    @ reference_hour = the time in hours when the light begins in the experiment, i.e. the beginning of a 24 hour session
    @ cache = if not None provide path for folder with saved caches or folder to be saved to
    
    returns a pandas dataframe containing raw ethoscope dataframe
    """

    if min_time > max_time:
        raise ValueError('Error: min_time is larger than max_time')

    if cache is not None:
        cache_name = 'cached_{}_{}_{}.pkl'.format(file['machine_id'], file['region_id'], file['date'])
        path = Path(cache) / Path(cache_name)
        if path.exists():
            data = pd.read_pickle(path)
            return data

    try:
        conn = sqlite3.connect(file['path'])

        roi_df = pd.read_sql_query('SELECT * FROM ROI_MAP', conn)
        
        roi_row = roi_df[roi_df['roi_idx'] == file['region_id']]

        if len(roi_row.index) < 1:
            print('ROI {} does not exist, skipping'.format(file['region_id']))
            return None

        var_df = pd.read_sql_query('SELECT * FROM VAR_MAP', conn)
        date = pd.read_sql_query('SELECT value FROM METADATA WHERE field = "date_time"', conn)

        # isolate date_time string and parse to GMT with format YYYY-MM-DD HH-MM-SS
        date = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(float(date.iloc[0])))      

        if max_time == float('inf'):
            max_time_condtion =  ''
        else:
            max_time_condtion = 'AND t < {}'.format(max_time * 1000) 
        
        min_time = min_time * 1000
        #sql_query takes roughyl 2.8 seconds for 2.5 days of data
        sql_query = 'SELECT * FROM ROI_{} WHERE t >= {} {}'.format(file['region_id'], min_time, max_time_condtion)
        data = pd.read_sql_query(sql_query, conn)
        
        if 'id' in data.columns:
            data = data.drop(columns = ['id'])

        if reference_hour != None:
            t = date
            t = t.split(' ')
            hh, mm , ss = map(int, t[1].split(':'))
            hour_start = hh + mm/60 + ss/3600
            t_after_ref = ((hour_start - reference_hour) % 24) * 3600 * 1e3
            data.t = (data.t + t_after_ref) / 1e3
        
        else:
            data.t = data.t / 1e3
            
        roi_width = max(roi_row['w'].iloc[0], roi_row['h'].iloc[0])
        for var_n in var_df['var_name']:
            if var_df['functional_type'][var_df['var_name'] == var_n].iloc[0] == 'distance':
                data[var_n] = data[var_n] / roi_width

        if 'is_inferred' and 'has_interacted' in data.columns:
            data = data[(data['is_inferred'] == False) | (data['has_interacted'] == True)]
            # check if has_interacted is all false / 0, drop if so
            interacted_list = data['has_interacted'].to_numpy()
            if (0 == interacted_list[:]).all() == True:
                data = data.drop(columns = ['has_interacted'])
                # data = data.drop(columns = ['is_inferred'])
        
        elif 'is_inferred' in data.columns:
            data = data[data['is_inferred'] == False]
            data = data.drop(columns = ['is_inferred'])

        if cache is not None:
            data.to_pickle(path)

        return data

    finally:
        conn.close()
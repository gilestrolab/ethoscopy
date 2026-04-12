import errno
import ftplib
import os
import sqlite3
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path, PurePath, PurePosixPath
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ethoscopy.misc.validate_datetime import validate_datetime

pd.options.mode.chained_assignment = None


def _connect_db(path):
    """
    Connect to a SQLite database with smart read-only filesystem detection.

    When the database directory is read-only (e.g., mounted with :ro in Docker),
    SQLite cannot create journal/WAL files and will fail to open the database.
    This function detects read-only filesystems and uses appropriate connection
    parameters to handle WAL-mode databases safely.

    Args:
        path (str): Path to the SQLite database file

    Returns:
        sqlite3.Connection: Database connection object

    Note:
        For WAL-mode databases on read-only mounts, this function uses mode=ro
        with nolock=1 to prevent "database disk image is malformed" errors.
        Any uncommitted WAL data will not be visible, which is acceptable for
        read-only mounts where the data cannot change anyway.
    """
    path_str = str(path)
    dir_path = os.path.dirname(path_str)

    # Check if we can write to the directory
    if not os.access(dir_path, os.W_OK):
        # Read-only filesystem - check if database is in WAL mode
        try:
            # Try to detect WAL mode by opening in read-only mode first
            temp_conn = sqlite3.connect(f"file:{path_str}?mode=ro", uri=True)
            cursor = temp_conn.cursor()
            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0].lower()
            temp_conn.close()

            if journal_mode == 'wal':
                # WAL mode on read-only mount: use mode=ro with nolock
                # This prevents "database disk image is malformed" errors
                # by avoiding operations that require WAL/SHM files
                return sqlite3.connect(
                    f"file:{path_str}?mode=ro&nolock=1",
                    uri=True,
                    timeout=10.0
                )
            else:
                # Non-WAL mode: use immutable mode for better performance
                return sqlite3.connect(f"file:{path_str}?immutable=1", uri=True)
        except Exception:
            # If detection fails, fall back to immutable mode
            return sqlite3.connect(f"file:{path_str}?immutable=1", uri=True)
    else:
        # Normal read-write access
        return sqlite3.connect(path_str)


def download_from_remote_dir(meta, remote_dir, local_dir):
    """
    Download ethoscope data from a remote FTP server to a local directory.

    Imports data from the ethoscope node platform to your local directory for later use. The ethoscope files
    must be saved on a remote FTP server as .db files. See the Ethoscope manual for node setup instructions:
    https://www.notion.so/giorgiogilestro/Ethoscope-User-Manual-a9739373ae9f4840aa45b277f2f0e3a7

    Args:
        meta (str): Path to a CSV file containing columns with machine_name, date, and time (if multiple files on the same day)
        remote_dir (str): URL of the FTP server up to the folder containing machine IDs. Server must allow anonymous login.
            e.g. 'ftp://YOUR_SERVER//auto_generated_data//ethoscope_results'
        local_dir (str): Path to the local directory for saving .db files. Files will be saved using the FTP server's structure.
            e.g. 'C:\\Users\\YOUR_NAME\\Documents\\ethoscope_databases'

    Returns:
        None

    Raises:
        FileNotFoundError: If the metadata file cannot be found or read
        KeyError: If required columns are missing from metadata
        RuntimeError: If no ethoscope data could be found
    """
    meta = Path(meta)
    local_dir = Path(local_dir)

    # check csv path is real and read to pandas df
    if meta.exists():
        try:
            meta_df = pd.read_csv(meta)
        except Exception as e:
            print("An error occurred: ", e)
    else:
        raise FileNotFoundError("The metadata is not readable")

    # check and tidy df, removing un-needed columns and duplicated machine names
    if "machine_name" not in meta_df.columns or "date" not in meta_df.columns:
        raise KeyError(
            "Column(s) 'machine_name' and/or 'date' missing from metadata file"
        )

    meta_df.dropna(how="all", inplace=True)

    if "time" in meta_df.columns.tolist():
        meta_df["check"] = meta_df["machine_name"] + meta_df["date"] + meta_df["time"]
        meta_df.drop_duplicates(
            subset=["check"], keep="first", inplace=True, ignore_index=False
        )
    else:
        meta_df["check"] = meta_df["machine_name"] + meta_df["date"]
        meta_df.drop_duplicates(
            subset=["check"], keep="first", inplace=True, ignore_index=False
        )

    # check the date format is YYYY-MM-DD, without this format the df merge will return empty
    # will correct to YYYY-MM-DD in a select few cases
    validate_datetime(meta_df)

    # extract columns as list to identify .db files from ftp server
    ethoscope_list = meta_df["machine_name"].tolist()
    date_list = meta_df["date"].tolist()

    if "time" in meta_df.columns.tolist():
        time_list = pd.Series(meta_df["time"].tolist())
        bool_list = time_list.isna().tolist()
    else:
        nan_list = [np.nan] * len(meta_df["date"])
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
                        date_time = exp.split("_")
                        if date_time[0] == date_list[c]:
                            if bool_list[c] is False:
                                if date_time[1] == time_list[c]:
                                    temp_path_3 = temp_path_2 / PurePosixPath(exp)
                                    ftp.cwd(str(temp_path_3))
                                    directories_4 = ftp.nlst()
                                    for db in directories_4:
                                        if db.endswith(".db"):
                                            size = ftp.size(db)
                                            final_path = f"{dir}/{name}/{exp}/{db}"
                                            path_size_list = [final_path, size]
                                            paths.append(path_size_list)
                                            check_list.append([name, date_time[0]])

                            else:
                                temp_path_3 = temp_path_2 / PurePosixPath(exp)
                                ftp.cwd(str(temp_path_3))
                                directories_4 = ftp.nlst()
                                for db in directories_4:
                                    if db.endswith(".db"):
                                        size = ftp.size(db)
                                        final_path = f"{dir}/{name}/{exp}/{db}"
                                        path_size_list = [final_path, size]
                                        paths.append(path_size_list)
                                        check_list.append([name, date_time[0]])

        except (OSError, IOError, Exception):
            continue

    if len(paths) == 0:
        raise RuntimeError(
            "No Ethoscope data could be found, please check the metadata file"
        )

    for i in zip(ethoscope_list, date_list):
        if list(i) in check_list:
            continue
        else:
            print(f"{i[0]}_{i[1]} has not been found for download")

    def download_database(
        remote_dir, folders, work_dir, local_dir, file_name, file_size
    ):
        """
        Download a database file from an FTP server to a local directory.

        Connects to remote FTP server and saves to designated local path, retaining file name
        and path directory structure.

        Args:
            remote_dir (str): FTP server netloc
            folders (str): Base path on the FTP server
            work_dir (PurePosixPath): Specific directory path on the FTP server
            local_dir (Path): Local directory path for saving the file and directory structure
            file_name (str): Name of the .db file to download
            file_size (int): Size of the file in bytes

        Returns:
            None
        """

        # create local copy of directory tree from ftp server
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
                ftp.cwd(folders + "/" + str(work_dir))

                localfile = open(file_path, "wb")
                ftp.retrbinary("RETR " + file_name, localfile.write)

                ftp.quit()
                localfile.close()

        else:
            ftp = ftplib.FTP(remote_dir)
            ftp.login()
            ftp.cwd(folders + "/" + str(work_dir))

            localfile = open(file_path, "wb")
            ftp.retrbinary("RETR " + file_name, localfile.write)

            ftp.quit()
            localfile.close()

    # iterate over paths, downloading each file
    # provide estimate download time based upon average time of previous downloads in queue
    download = partial(
        download_database,
        remote_dir=parse.netloc,
        folders=parse.path,
        local_dir=local_dir,
    )
    times = []

    for counter, j in enumerate(paths):
        print(
            "Downloading {}... {}/{}".format(
                j[0].split("/")[1], counter + 1, len(paths)
            )
        )
        if counter == 0:
            start = time.time()
            p = PurePosixPath(j[0])
            download(work_dir=p.parents[0], file_name=p.name, file_size=j[1])
            stop = time.time()
            t = stop - start
            times.append(t)

        else:
            av_time = round((np.mean(times) / 60) * (len(paths) - (counter + 1)))
            print(f"Estimated finish time: {av_time} mins")
            start = time.time()
            p = PurePosixPath(j[0])
            download(work_dir=p.parents[0], file_name=p.name, file_size=j[1])
            stop = time.time()
            t = stop - start
            times.append(t)


def link_meta_index(metadata, local_dir):
    """
    Link metadata with downloaded ethoscope database file paths.

    Alters the provided metadata file with the path locations of downloaded .db files from the Ethoscope
    experimental system. Checks all unique machines for errors, which are omitted from the returned
    metadata table without warning.

    Args:
        metadata (str): Path to a file containing metadata information for each ROI to be downloaded.
            Must include 'machine_name', 'date' (in yyyy-mm-dd format or other formats supported by
            validate_datetime), and 'region_id'.
        local_dir (str): Path to the top level parent directory where saved database files are located.

    Returns:
        pd.DataFrame: DataFrame containing the CSV file information and corresponding path for each entry

    Raises:
        FileNotFoundError: If the metadata file cannot be found or read
        ValueError: If the metadata contains NaN values
        KeyError: If required columns are missing from metadata
        RuntimeError: If no ethoscope data could be found
    """
    metadata = Path(metadata)
    local_dir = Path(local_dir)
    # load metadata csv file
    # check csv path is real and read to pandas df
    if metadata.exists():
        try:
            meta_df = pd.read_csv(metadata)
        except Exception as e:
            print("An error occurred: ", e)
    else:
        raise FileNotFoundError("The metadata is not readable")

    if len(meta_df[meta_df.isna().any(axis=1)]) >= 1:
        print(meta_df[meta_df.isna().any(axis=1)])
        raise ValueError(
            "When the metadata is read it contained NaN values (empty cells in the csv file can cause this!), please replace with an alterative"
        )

    # check and tidy df, removing un-needed columns and duplicated machine names
    if "machine_name" not in meta_df.columns or "date" not in meta_df.columns:
        raise KeyError(
            "Column(s) 'machine_name' and/or 'date' missing from metadata file"
        )

    meta_df.dropna(axis=0, how="all", inplace=True)

    # check the date format is YYYY-MM-DD, without this format the df merge will return empty
    # will correct to YYYY-MM-DD in a select few cases
    meta_df = validate_datetime(meta_df)

    meta_df_original = meta_df.copy(deep=True)

    if "time" in meta_df.columns.tolist():
        meta_df["check"] = meta_df["machine_name"] + meta_df["date"] + meta_df["time"]
        meta_df.drop_duplicates(
            subset=["check"], keep="first", inplace=True, ignore_index=False
        )
    else:
        meta_df["check"] = meta_df["machine_name"] + meta_df["date"]
        meta_df.drop_duplicates(
            subset=["check"], keep="first", inplace=True, ignore_index=False
        )

    ethoscope_list = meta_df["machine_name"].tolist()
    date_list = meta_df["date"].tolist()

    if "time" in meta_df.columns.tolist():
        time_list = meta_df["time"].tolist()
    else:
        nan_list = [np.nan] * len(meta_df["date"])
        time_list = nan_list

    paths = []
    sizes = []
    for name, date, time_val in zip(ethoscope_list, date_list, time_list):
        try:
            if np.isnan(time_val):
                regex = PurePath("*") / name / f"{date}_*" / "*.db"
                path_lst = local_dir.glob(str(regex))
                if len(list(path_lst)) >= 1:
                    for p in local_dir.glob(str(regex)):
                        paths.append(p)
                        sizes.append(p.stat().st_size)
                else:
                    print(f"{name}_{date} has not been found")
            else:
                regex = PurePath("*") / name / f"{date}_{time_val}" / "*.db"
                path_lst = local_dir.glob(str(regex))
                if len(list(path_lst)) >= 1:
                    for p in local_dir.glob(str(regex)):
                        paths.append(p)
                        sizes.append(p.stat().st_size)

                else:
                    print(f"{name}_{date} has not been found")
        except TypeError:
            regex = PurePath("*") / name / f"{date}_{time_val}" / "*.db"
            path_lst = local_dir.glob(str(regex))
            if len(list(path_lst)) >= 1:
                for p in local_dir.glob(str(regex)):
                    paths.append(p)
                    sizes.append(p.stat().st_size)
            else:
                print(f"{name}_{date} has not been found")

    if len(paths) == 0:
        raise RuntimeError(
            "No Ethoscope data could be found, please check the metatadata file"
        )

    # split path into parts
    split_df = pd.DataFrame()
    for path, size in zip(paths, sizes):
        split_path = str(path).replace(str(local_dir), "").split(os.sep)[1:]
        split_series = pd.DataFrame(data=split_path).T
        split_series.columns = ["machine_id", "machine_name", "date_time", "file_name"]
        split_series["path"] = str(path)
        split_series["file_size"] = size
        split_df = pd.concat([split_df, split_series], ignore_index=True)

    # split the date_time column and add back to df
    split_df[["date", "time"]] = split_df.date_time.str.split("_", expand=True)
    split_df.drop(columns=["date_time"], inplace=True)

    # merge df's
    if "time" in meta_df_original.columns.tolist():
        merge_df = meta_df_original.merge(
            split_df, how="outer", on=["machine_name", "date", "time"]
        )
        merge_df.dropna(inplace=True)

    else:
        drop_df = split_df.sort_values(["file_size"], ascending=False)
        drop_df = drop_df.drop_duplicates(["machine_name", "date"])
        droplog = split_df[split_df.duplicated(subset=["machine_name", "date"])]
        drop_list = droplog["machine_name"].tolist()
        if len(drop_list) >= 1:
            print(
                f"Ethoscopes {*drop_list,} have multiple files for their day, the largest file has been kept. If you want all files for that day please add a time column"
            )
        merge_df = meta_df_original.merge(
            drop_df, how="outer", on=["machine_name", "date"]
        )
        merge_df.dropna(inplace=True)

    # make the id for each row
    merge_df.insert(
        0,
        "id",
        merge_df["file_name"].str.slice(0, 26, 1)
        + "|"
        + merge_df["region_id"].astype(int).map("{:02d}".format),
    )

    return merge_df


def load_ethoscope(
    metadata,
    min_time=0,
    max_time=float("inf"),
    reference_hour=None,
    cache=None,
    FUN=None,
    verbose=True,
):
    """
    Load and process ethoscope data from database files.

    Iterates through the dataframe generated by link_meta_index() to load the corresponding database files
    and analyze them according to the provided function.

    Args:
        metadata (pd.DataFrame): Metadata dataframe as returned from link_meta_index function
        min_time (int, optional): Minimum time to load data from, with 0 being experiment start (in hours). Default is 0.
        max_time (int, optional): Maximum time to load data to (in hours). Default is infinity.
        reference_hour (int, optional): Hour at which lights on occurs or when timestamps should equal 0.
            None equals the start of the experiment. Default is None.
        cache (str, optional): Local path to find and store cached versions of each ROI per database.
            Directory structure mirrors ethoscope saved data. Cached files are in pickle format. Default is None.
        FUN (callable, optional): Function to apply individual curation to each ROI, typically using package
            generated functions (e.g., sleep_annotation). If None, data remains as found in the database. Default is None.
        verbose (bool, optional): If True, prints information about each ROI when loading. Default is True.

    Returns:
        pd.DataFrame: DataFrame containing the database data with unique IDs per fly as the index
    """

    max_time = max_time * 60 * 60
    min_time = min_time * 60 * 60

    # Collect all ROI data in a list for efficient concatenation
    roi_data_list = []

    # Handle empty metadata case
    if metadata.empty or "path" not in metadata.columns:
        return pd.DataFrame()

    # Group ROIs by database file to reuse connections and cache metadata
    grouped_metadata = metadata.groupby("path")

    # iterate over each database file
    for db_path, group in grouped_metadata:
        conn = None

        try:
            # Open connection once per database file
            conn = _connect_db(db_path)

            # Cache metadata queries that are the same for all ROIs in this database
            roi_df = pd.read_sql_query("SELECT * FROM ROI_MAP", conn)
            var_df = pd.read_sql_query("SELECT * FROM VAR_MAP", conn)
            date = pd.read_sql_query(
                'SELECT value FROM METADATA WHERE field = "date_time"', conn
            )
            if date.empty:
                raise ValueError("No date_time found in METADATA table")
            date_formatted = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(float(date.iloc[0].iloc[0]))
            )

            # Process each ROI in this database
            for i in group.index:
                file_info = metadata.iloc[metadata.index.get_loc(i), :]

                try:
                    if verbose is True:
                        print(
                            "Loading ROI_{} from {}".format(
                                file_info["region_id"], file_info["machine_name"]
                            )
                        )

                    # Use optimized single ROI reader with cached connection and metadata
                    roi_1 = read_single_roi_optimized(
                        file_info,
                        conn,
                        roi_df,
                        var_df,
                        date_formatted,
                        min_time,
                        max_time,
                        reference_hour,
                        cache,
                    )

                    if roi_1 is None:
                        if verbose is True:
                            print(
                                "ROI_{} from {} was unable to load due to an error formatting roi".format(
                                    file_info["region_id"], file_info["machine_name"]
                                )
                            )
                        continue

                    if FUN is not None:
                        roi_1 = FUN(roi_1)

                    if roi_1 is None:
                        if verbose is True:
                            print(
                                "ROI_{} from {} was unable to load due to an error in applying the function".format(
                                    file_info["region_id"], file_info["machine_name"]
                                )
                            )
                        continue

                    # Check if 'id' column already exists, if not insert it
                    if "id" not in roi_1.columns:
                        roi_1.insert(0, "id", file_info["id"])
                    else:
                        # Replace existing id with the one from metadata for consistency
                        roi_1["id"] = file_info["id"]

                    # Add to list instead of concatenating in loop
                    roi_data_list.append(roi_1)

                except Exception as e:
                    if verbose is True:
                        print(
                            "ROI_{} from {} was unable to load due to an error loading roi: {}".format(
                                file_info["region_id"],
                                file_info["machine_name"],
                                str(e),
                            )
                        )
                        import traceback

                        print("Full traceback:")
                        traceback.print_exc()
                    continue

        finally:
            # Close connection when done with this database
            if conn:
                conn.close()

    # Concatenate all data at once for much better performance
    if roi_data_list:
        data = pd.concat(roi_data_list, ignore_index=True)
    else:
        data = pd.DataFrame()

    return data


def load_ethoscope_metadata(metadata, reference_hour=9.0):
    """
    Extract metadata from ethoscope database files.

    Scrapes the metadata table of each ethoscope in the generated metadata file to provide
    experiment-level information.

    Args:
        metadata (pd.DataFrame): Metadata dataframe as returned from link_meta_index function

    Returns:
        pd.DataFrame: DataFrame containing the metadata from the METADATA table in each ethoscope database,
            with machine_id as the index
    """

    def get_meta(path):
        """
        Extract and process metadata from an ethoscope database file.

        Retrieves metadata from the METADATA table and processes it into a structured dictionary
        containing experiment information, hardware details, and configuration options.

        Args:
            path (str): Path to the ethoscope database file

        Returns:
            dict: Dictionary containing processed metadata from the database
        """
        try:
            conn = _connect_db(path)

            mdf = pd.read_sql_query("SELECT * FROM METADATA", conn)

            cols = mdf["field"].tolist()
            mdf = mdf.T
            mdf.columns = cols
            mdf.reset_index(inplace=True)
            mdf = mdf[1:]

            mdf["date_time"] = pd.to_datetime(pd.to_numeric(mdf["date_time"]), unit="s")

            d = eval(mdf["experimental_info"].iloc[0])
            exi = d

            d = eval(mdf["hardware_info"].iloc[0])
            d.pop("partitions", None)
            try:
                td = pd.DataFrame(d)
                if "version" in td.index:
                    hdi = td.loc["version"].to_dict()
                else:
                    hdi = {}
            except Exception:
                hdi = {}

            d = eval(
                mdf["selected_options"]
                .iloc[0]
                .replace("<", "")
                .replace(">", "")
                .replace("class ", "")
            )["interactor"]
            kw = d.pop("kwargs")
            kw["class"] = d["class"]

            mdf.drop(
                columns=[
                    "experimental_info",
                    "selected_options",
                    "hardware_info",
                    "index",
                    "backup_filename",
                ],
                errors="ignore",
                inplace=True,
            )

            row_dict = mdf.iloc[0].to_dict()
            # If date_range exists in DB (e.g. SD window), move it to stimulus_range
            if "date_range" in kw:
                sr = kw.pop("date_range")
                if " > " not in sr:
                    # Replace double spaces with " > " for clarity
                    sr = sr.replace("  ", " > ")
                row_dict["stimulus_range"] = sr
                
            row_dict.update(kw)
            row_dict.update(exi)
            row_dict.update(hdi)

            # Always compute ZT-aligned date_range and update date_time
            if "stop_date_time" in mdf.columns and "date_time" in mdf.columns:
                try:
                    dt = mdf["date_time"].iloc[0]
                    sdt_val = float(str(mdf["stop_date_time"].iloc[0]).strip("'"))
                    sdt = pd.to_datetime(sdt_val, unit="s")
                    
                    from datetime import timedelta
                    h = int(reference_hour)
                    m = int((reference_hour - h) * 60)
                    s = int(((reference_hour - h) * 60 - m) * 60)
                    
                    zt0_start = dt.replace(hour=h, minute=m, second=s, microsecond=0).floor("s")
                    if dt < zt0_start:
                        zt0_start -= timedelta(days=1)
                        
                    zt0_stop = sdt.replace(hour=h, minute=m, second=s, microsecond=0).floor("s")
                    if sdt < zt0_stop:
                        zt0_stop -= timedelta(days=1)
                        
                    start_str = zt0_start.strftime("%Y-%m-%d %H:%M:%S")
                    stop_str = zt0_stop.strftime("%Y-%m-%d %H:%M:%S")
                    row_dict["date_range"] = f"{start_str} > {stop_str}"
                    row_dict["date_time"] = zt0_start
                except Exception:
                    row_dict["date_range"] = ""
            else:
                row_dict["date_range"] = ""
            
            # Default min_inactive_time (usually 300) only if missing
            if "min_inactive_time" not in row_dict or row_dict["min_inactive_time"] is None:
                row_dict["min_inactive_time"] = 300

            return row_dict

        finally:
            conn.close()

    meta_df = metadata.copy(deep=True)

    if "time" in meta_df.columns.tolist():
        meta_df["check"] = meta_df["machine_name"] + meta_df["date"] + meta_df["time"]
        meta_df.drop_duplicates(
            subset=["check"], keep="first", inplace=True, ignore_index=False
        )
    else:
        meta_df["check"] = meta_df["machine_name"] + meta_df["date"]
        meta_df.drop_duplicates(
            subset=["check"], keep="first", inplace=True, ignore_index=False
        )

    rows = []

    # iterate over each ethoscope in the metadata df
    for i in meta_df["path"]:
        row = get_meta(i)
        rows.append(row)

    return pd.DataFrame(rows).set_index("machine_id")


def read_single_roi(
    file, min_time=0, max_time=float("inf"), reference_hour=None, cache=None
):
    """
    Load data from a single region of interest (ROI) from an ethoscope database.

    Extracts tracking data for a specific ROI according to time constraints, adjusts timestamps
    based on reference hour, and handles data caching.

    Args:
        file (pd.Series): Row in a metadata DataFrame containing a column 'path' with .db file location
        min_time (int, optional): Minimum time to load data from (in hours). Default is 0.
        max_time (int, optional): Maximum time to load data to (in hours). Default is infinity.
        reference_hour (int, optional): Time in hours when light begins in the experiment, used to
            adjust timestamps. None means timestamps start from experiment beginning. Default is None.
        cache (str, optional): Path for folder with saved caches or folder to save to. Default is None.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing raw ethoscope data for the specified ROI,
            or None if the ROI could not be loaded

    Raises:
        ValueError: If min_time is larger than max_time
    """

    if min_time > max_time:
        raise ValueError("Error: min_time is larger than max_time")

    if cache is not None:
        cache_name = "cached_{}_{}_{}.pkl".format(
            file["machine_id"], file["region_id"], file["date"]
        )
        path = Path(cache) / Path(cache_name)
        if path.exists():
            data = pd.read_pickle(path)
            return data

    try:
        conn = _connect_db(file["path"])

        # Check if database file is accessible and contains expected tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        required_tables = ["ROI_MAP", "VAR_MAP", "METADATA"]
        missing_tables = [table for table in required_tables if table not in tables]
        if missing_tables:
            raise ValueError(f"Database missing required tables: {missing_tables}")

        roi_df = pd.read_sql_query("SELECT * FROM ROI_MAP", conn)

        roi_row = roi_df[roi_df["roi_idx"] == file["region_id"]]

        if len(roi_row.index) < 1:
            available_rois = roi_df["roi_idx"].tolist()
            raise ValueError(
                f'ROI {file["region_id"]} does not exist. Available ROIs: {available_rois}'
            )

        # Check if ROI table exists
        roi_table_name = f'ROI_{file["region_id"]}'
        if roi_table_name not in tables:
            raise ValueError(f"ROI table {roi_table_name} does not exist in database")

        var_df = pd.read_sql_query("SELECT * FROM VAR_MAP", conn)
        date = pd.read_sql_query(
            'SELECT value FROM METADATA WHERE field = "date_time"', conn
        )

        # isolate date_time string and parse to GMT with format YYYY-MM-DD HH-MM-SS
        date = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime(float(date.iloc[0].iloc[0]))
        )

        if max_time == float("inf"):
            max_time_condtion = ""
        else:
            max_time_condtion = "AND t < {}".format(max_time * 1000)

        min_time = min_time * 1000
        # sql_query takes roughyl 2.8 seconds for 2.5 days of data
        sql_query = "SELECT * FROM ROI_{} WHERE t >= {} {}".format(
            file["region_id"], min_time, max_time_condtion
        )
        data = pd.read_sql_query(sql_query, conn)

        if "id" in data.columns:
            # Check if 'id' is a primary key (new format) or not (old format)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info(ROI_{file['region_id']})")
            columns = cursor.fetchall()

            is_primary_key = False
            for column in columns:
                if column[1] == "id" and column[5] == 1:  # column[5] is the pk flag
                    is_primary_key = True
                    break

            if not is_primary_key:
                # Old format - drop the id column to avoid conflicts
                data = data.drop(columns=["id"])
            # New format - keep the id column as it's a meaningful primary key

        if reference_hour is not None:
            t = date
            t = t.split(" ")
            hh, mm, ss = map(int, t[1].split(":"))
            hour_start = hh + mm / 60 + ss / 3600
            t_after_ref = ((hour_start - reference_hour) % 24) * 3600 * 1e3
            data.t = (data.t + t_after_ref) / 1e3

        else:
            data.t = data.t / 1e3

        roi_width = max(roi_row["w"].iloc[0], roi_row["h"].iloc[0])
        for var_n in var_df["var_name"]:
            if (
                var_df["functional_type"][var_df["var_name"] == var_n].iloc[0]
                == "distance"
            ):
                data[var_n] = data[var_n] / roi_width

        if "is_inferred" and "has_interacted" in data.columns:
            data = data[
                (data["is_inferred"] == 0)
                | (data["is_inferred"] == "0")
                | (data["has_interacted"] == 1)
            ]
            # check if has_interacted is all false / 0, drop if so
            interacted_list = data["has_interacted"].to_numpy()
            if (0 == interacted_list[:]).all():
                data = data.drop(columns=["has_interacted"])
                # data = data.drop(columns = ['is_inferred'])

        elif "is_inferred" in data.columns:
            data = data[(data["is_inferred"] == 0) | (data["is_inferred"] == "0")]
            data = data.drop(columns=["is_inferred"])

        if cache is not None:
            data.to_pickle(path)

        return data

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Database error for file {file['path']}: {str(e)}")
    except pd.errors.DatabaseError as e:
        raise pd.errors.DatabaseError(
            f"Pandas database error for file {file['path']}: {str(e)}"
        )
    except KeyError as e:
        raise KeyError(f"Missing required column in file {file['path']}: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error processing file {file['path']}: {str(e)}")
    finally:
        if "conn" in locals():
            conn.close()


def read_single_roi_optimized(
    file,
    conn,
    roi_df,
    var_df,
    date_formatted,
    min_time=0,
    max_time=float("inf"),
    reference_hour=None,
    cache=None,
):
    """
    Optimized version of read_single_roi that reuses database connections and cached metadata.

    Args:
        file: File metadata row
        conn: Reused SQLite connection
        roi_df: Cached ROI_MAP data
        var_df: Cached VAR_MAP data
        date_formatted: Pre-formatted date string
        min_time, max_time, reference_hour, cache: Same as read_single_roi
    """
    if min_time > max_time:
        raise ValueError("Error: min_time is larger than max_time")

    if cache is not None:
        cache_name = "cached_{}_{}_{}.pkl".format(
            file["machine_id"], file["region_id"], file["date"]
        )
        path = Path(cache) / Path(cache_name)
        if path.exists():
            data = pd.read_pickle(path)
            return data

    try:
        # Use cached ROI data instead of querying
        roi_row = roi_df[roi_df["roi_idx"] == file["region_id"]]

        if len(roi_row.index) < 1:
            available_rois = roi_df["roi_idx"].tolist()
            raise ValueError(
                f'ROI {file["region_id"]} does not exist. Available ROIs: {available_rois}'
            )

        # Check if ROI table exists (this still requires a query but much faster than full table read)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (f'ROI_{file["region_id"]}',),
        )
        if not cursor.fetchone():
            raise ValueError(
                f"ROI table ROI_{file['region_id']} does not exist in database"
            )

        # Use pre-formatted date instead of querying again
        date = date_formatted

        if max_time == float("inf"):
            max_time_condtion = ""
        else:
            max_time_condtion = "AND t < {}".format(max_time * 1000)

        min_time = min_time * 1000
        # This is the main data query - still needed but now with optimized context
        sql_query = "SELECT * FROM ROI_{} WHERE t >= {} {}".format(
            file["region_id"], min_time, max_time_condtion
        )

        # Execute query with retry logic for WAL-related errors
        try:
            data = pd.read_sql_query(sql_query, conn)
        except sqlite3.DatabaseError as e:
            # Handle "database disk image is malformed" errors
            # This can occur with WAL-mode databases on read-only mounts
            if "malformed" in str(e).lower() or "disk image" in str(e).lower():
                print(f"Warning: Database error for ROI {file['region_id']}, attempting retry with fresh connection...")

                # Get database path from file metadata
                db_path = file.get("path")
                if not db_path:
                    print(f"Error: Cannot retry - database path not found in file metadata")
                    raise

                # Create a fresh connection just for the retry
                retry_conn = None
                try:
                    retry_conn = _connect_db(db_path)
                    data = pd.read_sql_query(sql_query, retry_conn)
                    print(f"Success: ROI {file['region_id']} loaded on retry")
                except Exception as retry_error:
                    print(f"Error: Retry failed for ROI {file['region_id']}: {retry_error}")
                    raise
                finally:
                    # Clean up retry connection
                    if retry_conn:
                        try:
                            retry_conn.close()
                        except Exception:
                            pass
            else:
                # Re-raise other database errors
                raise

        if "id" in data.columns:
            # Check if 'id' is a primary key (reuse cursor)
            cursor.execute(f"PRAGMA table_info(ROI_{file['region_id']})")
            columns = cursor.fetchall()

            is_primary_key = False
            for column in columns:
                if column[1] == "id" and column[5] == 1:  # column[5] is the pk flag
                    is_primary_key = True
                    break

            if not is_primary_key:
                # Old format - drop the id column to avoid conflicts
                data = data.drop(columns=["id"])
            # New format - keep the id column as it's a meaningful primary key

        if reference_hour is not None:
            t = date
            t = t.split(" ")
            hh, mm, ss = map(int, t[1].split(":"))
            hour_start = hh + mm / 60 + ss / 3600
            t_after_ref = ((hour_start - reference_hour) % 24) * 3600 * 1e3
            data.t = (data.t + t_after_ref) / 1e3
        else:
            data.t = data.t / 1e3

        if cache is not None:
            data.to_pickle(path)

        return data

    except Exception as e:
        print(f"Error reading ROI {file['region_id']}: {e}")
        return None

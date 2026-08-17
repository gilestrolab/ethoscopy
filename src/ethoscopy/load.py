import errno
import ftplib
import os
import sqlite3
import time
import warnings

# Reason: newer ethoscope firmwares serialize the "selected_options" METADATA
# field with an ``OrderedDict([...])`` wrapper. ``get_meta`` below round-trips
# that blob through ``eval``, which resolves ``OrderedDict`` from this module's
# globals — so the import must stay even though no source line references it.
from collections import OrderedDict  # noqa: F401
from functools import partial
from pathlib import Path, PurePath, PurePosixPath
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ethoscopy.misc.validate_datetime import validate_datetime

pd.options.mode.chained_assignment = None


# Ordered ladder of read-only SQLite open modes, most faithful first.
#
# Reason: whether an ethoscope database can be opened is not predictable from
# its journal mode alone. Journal mode, the presence and state of the -wal/-shm
# sidecars and the writability of the containing directory interact, and the
# combinations are not rare in practice -- ethoscopes write in WAL mode and the
# results tree is usually mounted read-only. Empirically (sqlite 3.45 and 3.53,
# read-only and writable directories):
#
#   * mode=ro reads every recoverable state, including a WAL database whose
#     -shm exists, a stale or corrupt -shm, and a hot rollback journal, and it
#     is the only mode that sees data still sitting in an uncheckpointed -wal.
#   * immutable=1 is the sole survivor of one real case: a WAL-mode database
#     whose sidecars are absent on a read-only mount (SQLite would have to
#     create the -shm to read it). It ignores the -wal entirely, so it is the
#     fallback, never the first choice -- see _warn_if_wal_ignored.
#
# Note nolock=1 is deliberately absent: SQLite rejects it for *every* WAL
# database, and because sqlite3.connect() never touches the file the rejection
# only surfaces on the first query. That is why each rung below is probed.
_READ_STRATEGIES = ("mode=ro", "immutable=1")

# Errors that mean "this open mode cannot read this file" rather than "this
# query is wrong" -- worth dropping to the next rung of the ladder for.
_UNREADABLE_ERRORS = ("unable to open database file", "readonly database", "malformed")


def _warn_if_wal_ignored(path_str):
    """
    Warn when falling back to immutable=1 would hide committed data.

    immutable=1 reads the main database file only. If an uncheckpointed -wal
    sidecar still holds committed transactions, those rows silently disappear
    from the loaded data -- the worst possible failure for an analysis library,
    so make it loud.

    Args:
        path_str (str): Path to the SQLite database file
    """
    try:
        wal_size = os.path.getsize(f"{path_str}-wal")
    except OSError:
        return

    if wal_size > 0:
        warnings.warn(
            f"{path_str} could only be opened in SQLite's immutable mode, which "
            f"ignores its {wal_size} byte -wal sidecar: data committed to the WAL "
            "but not yet checkpointed will be missing. Checkpoint the database "
            "(scripts/convert_wal_to_delete.py) or make its directory writable.",
            RuntimeWarning,
            stacklevel=3,
        )


def _rebase_time(data, date_formatted, reference_hour):
    """
    Convert an ethoscope 't' column from milliseconds to seconds, optionally re-zeroed.

    Ethoscope tables store 't' as milliseconds since the start of that recording.
    With a reference_hour the series is instead expressed relative to the most
    recent occurrence of that wall-clock hour before the recording started, so
    runs begun at different times of day can be overlaid.

    Args:
        data (pd.DataFrame): Frame with a 't' column in milliseconds; modified in place
        date_formatted (str): Recording start as "%Y-%m-%d %H:%M:%S"
        reference_hour (float or None): Hour of day that should map to t = 0.
            None leaves t = 0 at the start of the recording.

    Returns:
        pd.DataFrame: The same frame, with 't' in seconds
    """
    if reference_hour is not None:
        hh, mm, ss = map(int, date_formatted.split(" ")[1].split(":"))
        hour_start = hh + mm / 60 + ss / 3600
        t_after_ref = ((hour_start - reference_hour) % 24) * 3600 * 1e3
        data.t = (data.t + t_after_ref) / 1e3
    else:
        data.t = data.t / 1e3

    return data


def _cache_path(cache, file, min_time, max_time, reference_hour):
    """
    Build the on-disk cache filename for one ROI.

    Reason: a cached frame is specific to the time window and reference hour it
    was read with - rows outside the window were dropped and 't' has already
    been rebased. Keying on the ROI alone hands back silently wrong timestamps,
    or a short frame, as soon as any of those arguments change. The default
    arguments reproduce the historic filename, so caches written by earlier
    versions stay valid.

    Args:
        cache (str): Directory holding the cached pickles
        file: Metadata row for this ROI
        min_time (float): Start of the loaded window, in seconds
        max_time (float): End of the loaded window, in seconds
        reference_hour (float or None): Hour of day mapped to t = 0

    Returns:
        Path: Full path of the pickle for this ROI and these arguments
    """
    key = "cached_{}_{}_{}".format(file["machine_id"], file["region_id"], file["date"])

    if reference_hour is not None:
        key += "_ref{:g}".format(reference_hour)
    if min_time:
        key += "_min{:g}".format(min_time)
    if max_time != float("inf"):
        key += "_max{:g}".format(max_time)

    return Path(cache) / f"{key}.pkl"


def _connect_db(path, degraded=False):
    """
    Open an ethoscope database read-only, tolerating WAL state and read-only mounts.

    ethoscopy never writes to these files, so the connection is always read-only:
    that also stops a load from checkpointing or truncating the raw data as a side
    effect. Each candidate open mode is probed with a real statement before being
    handed back, because sqlite3.connect() does not touch the file and an
    unusable mode would otherwise only fail deep inside the first data query.

    Args:
        path (str): Path to the SQLite database file
        degraded (bool, optional): Skip the faithful open modes and go straight to
            the last-resort one. Used to escalate after a connection that opened
            cleanly fails part-way through a read. Default is False.

    Returns:
        sqlite3.Connection: A connection that has answered at least one statement

    Raises:
        FileNotFoundError: If path does not exist
        sqlite3.OperationalError: If no open mode can read the file
    """
    path_str = str(path)

    # Reason: immutable=1 happily "opens" a path that is not there, creating an
    # empty database file and leaving the caller with a baffling
    # "no such table: ROI_MAP". Fail on the real problem instead.
    if not os.path.isfile(path_str):
        raise FileNotFoundError(
            errno.ENOENT, "No such ethoscope database file", path_str
        )

    strategies = _READ_STRATEGIES[-1:] if degraded else _READ_STRATEGIES

    last_error = None
    for strategy in strategies:
        conn = None
        try:
            conn = sqlite3.connect(
                f"file:{path_str}?{strategy}", uri=True, timeout=10.0
            )
            # Probe: forces SQLite to actually reach the file and its sidecars
            conn.execute("PRAGMA journal_mode;").fetchone()
        except sqlite3.Error as e:
            last_error = e
            if conn is not None:
                conn.close()
            continue

        if strategy == "immutable=1":
            _warn_if_wal_ignored(path_str)
        return conn

    raise sqlite3.OperationalError(
        f"Could not open {path_str} for reading with any of {list(strategies)}. "
        f"Last error: {last_error}"
    )


def download_from_remote_dir(meta, remote_dir, local_dir, progress=True):
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
        progress (bool, optional): If True, show a tqdm progress bar (ipywidgets-based in Jupyter,
            text in CLI). Default is True.

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

    # iterate over paths, downloading each file. tqdm provides per-file progress
    # plus an aggregate ETA, so we no longer need to hand-roll an estimate.
    download = partial(
        download_database,
        remote_dir=parse.netloc,
        folders=parse.path,
        local_dir=local_dir,
    )

    iterator = tqdm(
        paths,
        desc="Downloading databases",
        unit="db",
        disable=not progress,
    )
    for j in iterator:
        machine_name = j[0].split("/")[1]
        if progress:
            iterator.set_postfix_str(machine_name, refresh=False)
        p = PurePosixPath(j[0])
        download(work_dir=p.parents[0], file_name=p.name, file_size=j[1])


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
    progress=True,
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
        verbose (bool, optional): If True, emits per-ROI warnings when loading fails. Default is True.
        progress (bool, optional): If True, show a tqdm progress bar over ROIs (ipywidgets-based in
            Jupyter, text in CLI). Default is True.

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

    pbar = tqdm(
        total=len(metadata),
        desc="Loading ROIs",
        unit="roi",
        disable=not progress,
    )

    # iterate over each database file
    try:
        for db_path, group in grouped_metadata:
            conn = None
            pbar_at_db_start = pbar.n

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
                        if progress:
                            pbar.set_postfix_str(
                                f"{file_info['machine_name']} ROI_{file_info['region_id']}",
                                refresh=False,
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
                                tqdm.write(
                                    "ROI_{} from {} was unable to load due to an error formatting roi".format(
                                        file_info["region_id"],
                                        file_info["machine_name"],
                                    )
                                )
                            continue

                        if FUN is not None:
                            roi_1 = FUN(roi_1)

                        if roi_1 is None:
                            if verbose is True:
                                tqdm.write(
                                    "ROI_{} from {} was unable to load due to an error in applying the function".format(
                                        file_info["region_id"],
                                        file_info["machine_name"],
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
                            tqdm.write(
                                "ROI_{} from {} was unable to load due to an error loading roi: {}".format(
                                    file_info["region_id"],
                                    file_info["machine_name"],
                                    str(e),
                                )
                            )
                            import traceback

                            tqdm.write("Full traceback:")
                            tqdm.write(traceback.format_exc())
                        continue
                    finally:
                        pbar.update(1)

            except Exception as e:
                # Reason: opening the database or reading its shared tables sits
                # outside the per-ROI handler below, so without this one
                # unreadable file aborts the whole load and discards every ROI
                # already read from the other databases. Report it and move on.
                if verbose is True:
                    tqdm.write(
                        "Skipping {} - none of its {} ROIs could be loaded: {}".format(
                            db_path, len(group), e
                        )
                    )
                pbar.update(len(group) - (pbar.n - pbar_at_db_start))

            finally:
                # Close connection when done with this database
                if conn:
                    conn.close()
    finally:
        pbar.close()

    # Concatenate all data at once for much better performance
    if roi_data_list:
        data = pd.concat(roi_data_list, ignore_index=True)
    else:
        data = pd.DataFrame()

    return data


def _one_row_per_database(metadata):
    """
    Reduce a per-ROI metadata table to one row per database file.

    link_meta_index() returns a row per ROI, but device-level tables (METADATA,
    DIAGNOSTICS) are written once per recording. Collapse to the first ROI of
    each machine/date (plus time, when the metadata distinguishes several runs
    on one day) so those tables are read once each.

    Args:
        metadata (pd.DataFrame): Metadata dataframe as returned from link_meta_index

    Returns:
        pd.DataFrame: Copy of the metadata with one row per recording
    """
    meta_df = metadata.copy(deep=True)

    keys = ["machine_name", "date"]
    if "time" in meta_df.columns.tolist():
        keys.append("time")

    meta_df["check"] = meta_df[keys].astype(str).agg("".join, axis=1)
    meta_df.drop_duplicates(
        subset=["check"], keep="first", inplace=True, ignore_index=False
    )

    return meta_df


def load_ethoscope_diagnostics(
    metadata,
    min_time=0,
    max_time=float("inf"),
    reference_hour=None,
    progress=True,
):
    """
    Load the recording-quality DIAGNOSTICS table from each ethoscope database.

    DIAGNOSTICS is sampled periodically by the tracking daemon and describes the
    *recording* rather than any one animal - achieved frame rate, image and frame
    noise, focus, camera jitter and CPU temperature. It is therefore keyed by
    machine rather than by ROI, and is returned as a tidy DataFrame rather than a
    behavpy object. Merge it onto experimental variables with the 'machine_name'
    and 'date' columns.

    Only ethoscope firmwares that record diagnostics write this table. Databases
    without one are skipped and reported, so a mixed-firmware experiment loads
    whatever data exists instead of failing.

    Args:
        metadata (pd.DataFrame): Metadata dataframe as returned from link_meta_index function
        min_time (int, optional): Minimum time to load data from, with 0 being experiment
            start (in hours). Default is 0.
        max_time (int, optional): Maximum time to load data to (in hours). Default is infinity.
        reference_hour (int, optional): Hour at which lights on occurs or when timestamps
            should equal 0. None equals the start of the experiment. Default is None.
        progress (bool, optional): If True, show a tqdm progress bar (ipywidgets-based in
            Jupyter, text in CLI). Default is True.

    Returns:
        pd.DataFrame: One row per diagnostics sample with columns 'machine_id',
            'machine_name', 'date', 't' (seconds) and every variable recorded by the
            firmware. Empty if no database carried a DIAGNOSTICS table.

    Raises:
        ValueError: If min_time is larger than max_time
    """
    if min_time > max_time:
        raise ValueError("Error: min_time is larger than max_time")

    if metadata.empty or "path" not in metadata.columns:
        return pd.DataFrame()

    meta_df = _one_row_per_database(metadata)

    time_condition = "WHERE t >= {}".format(min_time * 60 * 60 * 1000)
    if max_time != float("inf"):
        time_condition += " AND t < {}".format(max_time * 60 * 60 * 1000)

    frames = []
    without_table = []

    for i in tqdm(
        meta_df.index,
        desc="Reading diagnostics",
        unit="db",
        disable=not progress,
    ):
        row = meta_df.loc[i]
        conn = None

        try:
            conn = _connect_db(row["path"])

            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='DIAGNOSTICS'"
            )
            if not cursor.fetchone():
                without_table.append(row["machine_name"])
                continue

            diagnostics = pd.read_sql_query(
                f"SELECT * FROM DIAGNOSTICS {time_condition}", conn
            )

            date = pd.read_sql_query(
                'SELECT value FROM METADATA WHERE field = "date_time"', conn
            )
            if date.empty:
                raise ValueError("No date_time found in METADATA table")
            date_formatted = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(float(date.iloc[0].iloc[0]))
            )

        except Exception as e:
            tqdm.write(
                "Diagnostics from {} could not be read: {}".format(
                    row["machine_name"], e
                )
            )
            continue

        finally:
            if conn is not None:
                conn.close()

        if diagnostics.empty:
            continue

        diagnostics = _rebase_time(diagnostics, date_formatted, reference_hour)
        diagnostics.insert(0, "date", row["date"])
        diagnostics.insert(0, "machine_name", row["machine_name"])
        diagnostics.insert(0, "machine_id", row["machine_id"])
        frames.append(diagnostics)

    if without_table:
        warnings.warn(
            "No DIAGNOSTICS table in the databases for {}: these ethoscopes ran a "
            "firmware that does not record diagnostics, and are absent from the "
            "returned data.".format(", ".join(sorted(set(without_table)))),
            RuntimeWarning,
            stacklevel=2,
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_ethoscope_metadata(metadata, progress=True):
    """
    Extract metadata from ethoscope database files.

    Scrapes the metadata table of each ethoscope in the generated metadata file to provide
    experiment-level information.

    Args:
        metadata (pd.DataFrame): Metadata dataframe as returned from link_meta_index function
        progress (bool, optional): If True, show a tqdm progress bar (ipywidgets-based in Jupyter,
            text in CLI). Default is True.

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
        conn = None
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

            # Reason: older ethoscope firmwares omit "partitions" and/or store
            # hardware_info without a nested "version" mapping. Tolerate both
            # so a single quirky database doesn't abort the whole load.
            d = eval(mdf["hardware_info"].iloc[0])
            d.pop("partitions", None)
            try:
                td = pd.DataFrame(d)
                hdi = td.loc["version"].to_dict() if "version" in td.index else {}
            except (ValueError, KeyError):
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

            # Reason: interactors such as sleep-deprivation store their active
            # time window under the kwarg "date_range". Surface it as
            # "stimulus_range" so it cannot be confused with the experiment's
            # own acquisition date range.
            if "date_range" in kw:
                kw["stimulus_range"] = kw.pop("date_range")

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
            row_dict.update(kw)
            row_dict.update(exi)
            row_dict.update(hdi)

            return row_dict

        finally:
            if conn is not None:
                conn.close()

    meta_df = _one_row_per_database(metadata)

    rows = []

    # iterate over each ethoscope in the metadata df
    for i in tqdm(
        meta_df["path"],
        desc="Reading metadata",
        unit="db",
        disable=not progress,
    ):
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
        path = _cache_path(cache, file, min_time, max_time, reference_hour)
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

        data = _rebase_time(data, date, reference_hour)

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
        path = _cache_path(cache, file, min_time, max_time, reference_hour)
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

        # Execute query, escalating to a degraded open mode if the shared
        # connection turns out to be unable to read this database after all.
        # Reason: the connection was probed at open time, but the -wal/-shm
        # sidecars can change underneath a long read on a live mount, so the
        # failure can still land here. Retrying with _connect_db() alone would
        # just pick the same open mode again -- degraded=True is what makes the
        # retry a different attempt rather than a repeat of the failed one.
        try:
            data = pd.read_sql_query(sql_query, conn)
        except sqlite3.DatabaseError as e:
            if not any(marker in str(e).lower() for marker in _UNREADABLE_ERRORS):
                # A genuine query error - retrying will not help
                raise

            tqdm.write(
                f"Warning: Database error for ROI {file['region_id']} ({e}), "
                "retrying with a fresh read-only connection..."
            )

            db_path = file.get("path")
            if not db_path:
                tqdm.write(
                    "Error: Cannot retry - database path not found in file metadata"
                )
                raise

            retry_conn = None
            try:
                retry_conn = _connect_db(db_path, degraded=True)
                data = pd.read_sql_query(sql_query, retry_conn)
                tqdm.write(f"Success: ROI {file['region_id']} loaded on retry")
            except Exception as retry_error:
                tqdm.write(
                    f"Error: Retry failed for ROI {file['region_id']}: {retry_error}"
                )
                raise
            finally:
                if retry_conn:
                    try:
                        retry_conn.close()
                    except Exception:
                        pass

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

        data = _rebase_time(data, date, reference_hour)

        if cache is not None:
            data.to_pickle(path)

        return data

    except Exception as e:
        tqdm.write(f"Error reading ROI {file['region_id']}: {e}")
        return None

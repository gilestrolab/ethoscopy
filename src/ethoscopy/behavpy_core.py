import pandas as pd
import numpy as np 
import pickle

from math import floor
from functools import partial

from tabulate import tabulate
from hmmlearn import hmm

from scipy.signal import find_peaks
from scipy.stats import zscore

from ethoscopy.misc.general_functions import concat, rle
from ethoscopy.analyse import max_velocity_detector
from ethoscopy.misc.periodogram_functions import chi_squared, lomb_scargle, fourier, wavelet

from typing import Optional, List, Union, Tuple, Dict, Any, Callable

class behavpy_core(pd.DataFrame):
    """
    A specialised DataFrame subclass for analysing ethoscope behavioural data.

    The behavpy class extends pandas DataFrame to provide specialised functionality for handling and analysing
    behavioural data from ethoscope experiments. It maintains a link between experimental data and corresponding 
    metadata, with methods for data manipulation, analysis, and visualization.

    The class stores metadata as a DataFrame attribute accessible via behavpy.meta. Both the data and metadata
    must share unique specimen IDs in their 'id' columns. Use .display() to view both data and metadata together,
    as print() will only show the data portion.

    Behavpy_core contains the base functionality of the behavpy class, such as xmv(), curate(), summary() etc.
    The behavpy_core class is not intended to be used directly, but rather as a base class for the behavpy plotly
    and seaborn classes to extend.

    Attributes:
        _metadata (list[str]): ensures that meta is a permenant attribute
        canvas (str): the canvas to be used for plotting, either 'plotly' or 'seaborn'. This is set in behavpy_plotly or behavpy_seaborn.
        _hmm_colours (list[str]): A list of default colors for HMM states if only 4 states are provided. This is set in behavpy_draw.
        _hmm_labels (list[str]): A list of default labels for HMM states if only 4 states are provided. This is set in behavpy_draw.
    """

    # set meta as permenant attribute
    _metadata = ['meta']
    canvas = None
    _hmm_colours = None
    _hmm_labels = None

    @property
    def _constructor(self):
        return behavpy_core._internal_constructor(self.__class__)

    class _internal_constructor(object):
        def __init__(self, cls):
            self.cls = cls

        def __call__(self, *args, **kwargs):
            kwargs['meta'] = None
            return self.cls(*args, **kwargs)

        def _from_axes(self, *args, **kwargs):
            return self.cls._from_axes(*args, **kwargs)

    def __init__(self, data: pd.DataFrame, meta: pd.DataFrame, palette: Optional[str] = None, long_palette: Optional[str] = None, 
                 check: bool = False, index: Optional[pd.Index] = None, columns: Optional[pd.Index] = None, 
                 dtype: Optional[np.dtype] = None, copy: bool = True):
        super(behavpy_core, self).__init__(data=data,
                                        index=index,
                                        columns=columns,
                                        dtype=dtype,
                                        copy=copy)

        self.meta = meta  
        if check is True:
            self._check_conform(self)
        self.attrs = {'sh_pal' : palette, 'lg_pal' : long_palette}

    @staticmethod
    def _check_conform(dataframe: pd.DataFrame) -> None:
        """ 
        Validates data format and metadata matching.
        
        Checks if metadata is a pandas DataFrame and if all IDs in the data exist in metadata.
        Removes unnecessary columns from metadata.

        Args:
            dataframe (pd.DataFrame): The behavpy DataFrame to validate

        Raises:
            TypeError: If metadata is not a pandas dataframe
            KeyError: If 'id' column missing from data or metadata
            RuntimeError: If IDs in data don't match metadata
        """
        if not isinstance(dataframe.meta, pd.DataFrame):
            raise TypeError('Metadata input is not a pandas dataframe')

        drop_col_names = ['path', 'file_name', 'file_size', 'machine_id']
        dataframe.meta = dataframe.meta.drop(columns=[col for col in dataframe.meta if col in drop_col_names])

        if dataframe.index.name != 'id':
            try:
                dataframe.set_index('id', inplace = True)
            except:
                raise KeyError("There is no 'id' as a column or index in the data'")

        if dataframe.meta.index.name != 'id':
            try:
                dataframe.meta.set_index('id', inplace = True)
            except:
                raise KeyError("There is no 'id' as a column or index in the metadata'")

        # checks if all id's of data are in the metadata dataframe
        check_data = all(elem in set(dataframe.meta.index.tolist()) for elem in set(dataframe.index.tolist()))
        if check_data is not True:
            raise RuntimeError("There are ID's in the data that are not in the metadata, please check")

    def _check_lists(self, f_col: Optional[str], f_arg: Optional[List], f_lab: Optional[List]) -> tuple[List, List]:
        """
        Validate and populate facet arguments and labels.
        
        Checks if facet column exists and matches provided arguments/labels.
        Auto-populates missing arguments/labels from column values.

        Args:
            f_col (list[str]): Name of facet column in metadata
            f_arg (list[str]): List of facet arguments to filter by
            f_lab (list[str]): List of labels for facet arguments

        Returns:
            tuple[List, List]: Validated facet arguments and labels

        Raises:
            KeyError: If facet column doesn't exist in metadata
            ValueError: If label length doesn't match argument length
        """

        # Handle case when no facet column specified
        if f_col is None:
            return [None], [''] if f_lab is None else f_lab

        # Validate facet column exists
        if f_col not in self.meta.columns:
            raise KeyError(f'Column "{f_col}" is not a metadata column')

        # Get unique values from column if no arguments provided
        if f_arg is None:
            f_arg = list(set(self.meta[f_col].tolist()))
        else:
            # Validate provided arguments exist in column
            for arg in f_arg:
                if arg not in self.meta[f_col].tolist():
                    raise KeyError(f'Argument "{arg}" is not in the meta column {f_col}')

        # Use string arguments as labels if none provided
        if f_lab is None:
            f_lab = [str(arg) for arg in f_arg] # Convert arguments to strings

        # Validate label length matches argument length
        if len(f_arg) != len(f_lab):
            raise ValueError("The facet labels don't match the length of facet arguments")

        return f_arg, f_lab

    def display(self) -> None:
        """
        Display both metadata and data with formatted headers.
        
        Alternative to print() that shows both the metadata and data components
        with clear section headers.

        Returns:
            None: Prints formatted output to console showing metadata and data tables
        """
        print('\n ==== METADATA ====\n\n{}\n ====== DATA ======\n\n{}'.format(self.meta, self))

    def xmv(self, column: str, *args: Any) -> "behavpy_core":
        """
        Filter behavpy data and metadata by matching column values.

        Filters both the data and metadata DataFrames based on values in a specified metadata column.
        For 'id' column filtering, matches are done against the index. For other columns, matches
        are done against column values.

        Args:
            column (str): Column name from metadata to filter by
            *args (Any): One or more values to filter by. Can be passed as separate arguments
                or as a single list/array. Values must exist in the specified column.

        Returns:
            behavpy_core: New behavpy object containing only rows where metadata column 
                matches the specified values

        Raises:
            KeyError: If column doesn't exist in metadata or if any filter value isn't found
                in the specified column

        Examples:
            # Filter by single genotype
            df.xmv('genotype', 'wild_type')
            
            # Filter by multiple treatments
            df.xmv('treatment', 'control', 'drug_1', 'drug_2')
            
            # Filter by list of IDs
            df.xmv('id', ['fly1', 'fly2', 'fly3'])
        """
        # Convert args to list if first arg is list/array
        filter_values = args[0] if isinstance(args[0], (list, np.ndarray)) else args
        
        # Get unique data IDs once
        data_ids = np.array(list(set(self.index.values)))
        
        if column == 'id':
            # Build list of valid IDs from metadata
            index_list = []
            for value in filter_values:
                if value not in self.meta.index:
                    raise KeyError(f'Metavariable "{value}" is not in the id column')
                index_list.extend(self.meta[self.meta.index == value].index.values)
                
            # Find intersection of meta and data IDs
            new_index_list = np.intersect1d(index_list, data_ids)
            
        else:
            # Validate column exists
            if column not in self.meta.columns:
                raise KeyError(f'Column heading "{column}" is not in the metadata table')
            
            # Build list of valid IDs from metadata column
            index_list = []
            for value in filter_values:
                if value not in self.meta[column].values:
                    raise KeyError(f'Metavariable "{value}" is not in the column')
                index_list.extend(self.meta[self.meta[column] == value].index.values)
                
            # Find intersection of meta and data IDs
            new_index_list = np.intersect1d(index_list, data_ids)
        
        # Filter both data and metadata using the intersected IDs
        filtered_data = self[self.index.isin(new_index_list)]
        filtered_data.meta = self.meta[self.meta.index.isin(new_index_list)]
        
        return filtered_data

    def remove(self, column: str, *args: Any) -> "behavpy_core":
        """
        Remove rows from data and metadata whose IDs match specified metadata values.
        
        Inverse operation of xmv() - removes matching rows instead of keeping them.

        Args:
            column (str): Column name from metadata to filter by
            *args (Any): One or more values to remove. Can be passed as separate arguments
                or as a single list/array. Values must exist in the specified column.

        Returns:
            behavpy_core: New behavpy object with specified rows removed

        Examples:
            # Remove single genotype
            df.remove('genotype', 'mutant_1')
            
            # Remove multiple treatments
            df.remove('treatment', 'drug_1', 'drug_2')
            
            # Remove list of IDs
            df.remove('id', ['fly1', 'fly2'])
        """
        # Convert args to list if first arg is list/array
        filter_values = args[0] if isinstance(args[0], (list, np.ndarray)) else args
        
        if column == 'id':
            # Get all IDs except those to remove
            all_ids = set(self.meta.index.tolist())
            remove_ids = set(filter_values)
            keep_ids = list(all_ids - remove_ids)
            
            # Use xmv to keep only the non-removed IDs
            return self.xmv('id', keep_ids)
            
        else:
            # Get all unique values in column except those to remove
            all_values = set(self.meta[column].unique())
            remove_values = set(filter_values)
            keep_values = list(all_values - remove_values)
            
            # Use xmv to keep only the non-removed values
            return self.xmv(column, keep_values)


    def summary(self, detailed: bool = False, t_column: str = 't') -> None:
        """
        Print a table with summary statistics of metadata and data counts.

        Args:
            detailed (bool, optional): If True, show count and time range of data points per ID.
                Defaults to False.
            t_column (str, optional): Column containing timestamps. 
                Defaults to 't'.

        Returns:
            None
        """

        def print_table(table: List[List[Any]]) -> None:
            longest_cols = [
                (np.nanmax([len(str(row[i])) for row in table]) + 3)
                for i in range(len(table[0]))
            ]
            row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
            for row in table:
                print(row_format.format(*row))

        if detailed is False:
            individuals = len(self.meta.index)
            metavariable = len(self.meta.columns)
            variables = len(self.columns)
            measurements = len(self.index)
            table = [
                ['individuals', individuals],
                ['metavariable', metavariable],
                ['variables', variables],
                ['measurements', measurements],
            ]
            print('behavpy table with: ')
            print_table(table)

        if detailed is True:

            def time_range(data: pd.Series) -> str:
                return (str(min(data)) + '  ->  ' + str(np.nanmax(data)))


            group = self.groupby('id').agg(
                data_points = pd.NamedAgg(column = t_column, aggfunc = 'count'),
                time_range = pd.NamedAgg(column = t_column, aggfunc = time_range)
            )

            print(group)

    def add_day_phase(self, t_column: str = 't', day_length: int = 24, 
                     lights_off: int = 12, inplace: bool = True) -> Optional["behavpy_core"]:
        """
        Add phase and day columns based on time data.
        
        Adds 'phase' column with 'light'/'dark' categories and 'day' column with sequential day numbers.

        Args:
            t_column (str, optional): Column containing timestamps in seconds. Default is 't'.
            day_length (int, optional): Length of experimental day in hours. Default is 24.
            lights_off (int, optional): Hour when lights turn off (0-day_length). Default is 12.
            inplace (bool, optional): Modify existing DataFrame if True, return new one if False.

        Returns:
            Optional[behavpy_core]: Modified behavpy object if inplace=False, None otherwise
        """
        day_in_secs = 60*60*day_length
        night_in_secs = lights_off * 60 * 60

        if inplace == True:
            self['day'] = self[t_column].map(lambda t: floor(t / day_in_secs))
            self['phase'] = np.where(((self[t_column] % day_in_secs) > night_in_secs), 'dark', 'light')
            self['phase'] = self['phase'].astype('category')

        elif inplace == False:
            new_df = self.copy(deep = True)
            new_df['day'] = new_df[t_column].map(lambda t: floor(t / day_in_secs)) 
            new_df['phase'] = np.where(((new_df[t_column] % day_in_secs) > night_in_secs), 'dark', 'light')
            new_df['phase'] = new_df['phase'].astype('category')

            return new_df

    def t_filter(self, start_time: int = 0, end_time: int = np.inf, t_column: str = 't') -> "behavpy_core":
        """
        Filter data between start and end times.
        
        Filters the behavpy data to only include rows between the provided start and end times.
        Times are provided in hours and converted to seconds internally.

        Args:
            start_time (int, optional): Start time in hours to filter from. Default is 0.
            end_time (int, optional): End time in hours to filter to. Default is infinity.
            t_column (str, optional): Column containing timestamps in seconds. Default is 't'.

        Returns:
            behavpy_core: Filtered behavpy object containing only data between start and end times.

        Note:
            Times in t_column must be in seconds.
        """
        if start_time > end_time:
            raise ValueError('Error: end_time is larger than start_time')

        return self[(self[t_column] >= (start_time * 60 * 60)) & (self[t_column] < (end_time * 60 * 60))]

    def rejoin(self, new_column: pd.DataFrame, check: bool = True) -> None:
        """
        Joins a new column to the metadata.
        
        Adds a new column to the metadata by joining on the 'id' index. The behavpy object
        is modified in place.

        Args:
            new_column (pd.DataFrame): Column to add. Must contain an 'id' index matching
                the original metadata.
            check (bool, optional): Whether to verify all data IDs exist in new column.
                Default is True.

        Returns:
            None
        """

        if check is True:
            check_data = all(elem in new_column.index.tolist() for elem in set(self.index.tolist()))
            if check_data is not True:
                raise KeyError("There are ID's in the data that are not in the metadata, please check. You can skip this process by changing the parameter skip to False")

        m = self.meta.join(new_column, on = 'id')
        self.meta = m

    # Moved concat to be a function rather than a method. Access it by etho.concat(df, df2, ...)

    def stitch_consecutive(self, machine_name_col: str = 'machine_name', region_id_col: str = 'region_id', 
                           date_col: str = 'date', t_column: str = 't') -> "behavpy_core":
        """
        Stitch together data from ROIs of the same machine in the dataframe.

        Used when an experiment needed to be stopped and restarted with the same specimens.
        Merges matching machine names and ROI numbers by adjusting timestamps based on 
        experiment start dates. Only retains metadata from the earliest experiment and 
        updates IDs to match the first run to avoid confusion in later processing.

        Args:
            machine_name_col (str): Column containing machine names (e.g. 'ETHOSCOPE_030'). 
                Default is 'machine_name'.
            region_id_col (str): Column containing ROI identifiers per machine.
                Default is 'region_id'.
            date_col (str): Column containing experiment dates.
                Default is 'date'.
            t_column (str): Column containing timestamps in seconds.
                Default is 't'.

        Returns:
            behavpy_core: New behavpy object with stitched data and metadata for each
                machine/ROI combination.

        Note:
            Dataframe should only contain experiments from machines you want to stitch!
        """
        mach_list = set(self.meta[machine_name_col].tolist())
        roi_list = set(self.meta[region_id_col].tolist())

        def augment_time(d: "behavpy_core") -> "behavpy_core":
            d.meta['day_diff'] = (pd.to_datetime(d.meta[date_col]) - pd.to_datetime(d.meta[date_col].min())).dt.days
            indx_map = d.meta[['day_diff']].to_dict()['day_diff']
            d['day'] = indx_map
            d[t_column] = (d['day'] * 86400) + d[t_column]
            d['id'] = [list(indx_map.keys())[list(indx_map.values()).index(0)]] * len(d)
            d = d.set_index('id')
            d.meta = d.meta[d.meta['day_diff'] == 0]
            d.meta.drop(['day_diff'], axis = 1, inplace = True)
            d.drop(['day'], axis = 1, inplace = True)
            return d

        df_list = []

        for i in mach_list:
            mdf = self.xmv(machine_name_col, i)
            for q in roi_list:
                try:
                    rdf = mdf.xmv(region_id_col, q)
                    ndf = augment_time(rdf)
                    df_list.append(ndf)
                except KeyError:
                    continue

        return concat(*df_list)

    def analyse_column(self, column: Union[str, List[str]], function: Union[str, Callable]) -> pd.DataFrame:
        """
        Apply an aggregation function to one or more columns, grouping by specimen ID.

        A wrapper around pandas groupby() that applies an aggregation function to specified columns,
        with results grouped by specimen ID. The function can be a string name of a pandas aggregation 
        function (e.g. 'mean', 'max') or a custom callable.

        Args:
            column (Union[str, List[str]]): Name of column(s) to aggregate. Can be a single column name 
                or list of column names.
            function (Union[str, Callable]): Aggregation function to apply. Can be:
                - String name of pandas aggregation function (e.g. 'mean', 'sum', 'max')
                - Custom callable that takes a Series/DataFrame and returns a single value

        Returns:
            pd.DataFrame: DataFrame containing aggregated results with:
                - Index: Specimen IDs from original data
                - Columns: Original column names with function name appended
                    (e.g. 'column_mean' for function='mean')

        Raises:
            KeyError: If any specified column does not exist in the data
            
        Examples:
            # Apply mean to single column
            df.analyse_column('distance', 'mean')
            
            # Apply custom function to multiple columns
            df.analyse_column(['x', 'y'], lambda x: x.max() - x.min())
        """

        # Convert single column to list
        if isinstance(column, str):
            column = [column]
        
        # Validate inputs
        if not column:
            raise ValueError("At least one column must be specified")
            
        # Check column existence
        for col in column:
            if col not in self.columns:
                raise KeyError(f'Column heading "{col}" is not in the data table')

        # Validate function type
        if not (isinstance(function, str) or callable(function)):
            raise TypeError("Function must be either a string name or callable")   
             
        # Create column name mapping
        name_dict = {}
        for col in column:
            try:
                # For callable functions
                parse_name = f'{col}_{function.__name__}'
            except AttributeError:
                # For string function names
                parse_name = f'{col}_{function}'
            name_dict[col] = parse_name

        # Create aggregation dictionary and perform groupby
        agg_dict = dict.fromkeys(column, function)
        pivot = self.groupby(self.index).agg(agg_dict)
        pivot = pivot.rename(name_dict, axis=1)

        return pivot

    def wrap_time(self, wrap_time: int = 24, time_column: str = 't') -> "behavpy_core":
        """
        Wraps time values to a specified period by applying modulo operation.

        Takes linear time values and converts them to cyclical values within the specified
        period (e.g., 24 hours). For example, with wrap_time=24, times of 25, 49, and 73 
        hours would become 1, 1, and 1 respectively.

        Args:
            wrap_time (int, optional): Period in hours to wrap the time series by. 
                Must be positive. Defaults to 24 hours.
            time_column (str, optional): Name of column containing time values in seconds.
                Must exist in DataFrame. Defaults to 't'.

        Returns:
            behavpy_core: New behavpy object with wrapped time values.

        Raises:
            ValueError: If wrap_time is not positive.
            KeyError: If time_column does not exist in DataFrame.
        """
        # Validate inputs
        if wrap_time <= 0:
            raise ValueError("wrap_time must be positive")
            
        if time_column not in self.columns:
            raise KeyError(f"Column '{time_column}' not found in data")

        hours_in_seconds = wrap_time * 60 * 60
        
        new = self.copy(deep=True)
        new[time_column] = new[time_column] % hours_in_seconds
        return new

    def baseline(self, column: str, t_column: str = 't', day_length: int = 24) -> "behavpy_core":
        """
        Shifts time series data by adding days to align interaction times across specimens.
        
        Adds a specified number of days to each specimen's timestamps based on values in a metadata column.
        This allows alignment of experimental start times when specimens were started on different days.

        Args:
            column (str): Name of metadata column containing the number of days to add.
                Must contain integers (0 = no shift).
            t_column (str, optional): Name of column containing timestamps in seconds.
                Defaults to 't'.
            day_length (int, optional): Length of experimental day in hours.
                Defaults to 24.

        Returns:
            behavpy_core: New behavpy object with shifted timestamps.

        Raises:
            KeyError: If column not found in metadata or t_column not found in data
            ValueError: If column contains non-integer values
            ValueError: If day_length is not positive
        """
        # Validate inputs
        if column not in self.meta.columns:
            raise KeyError(f'Column "{column}" not found in metadata')
        if t_column not in self.columns:
            raise KeyError(f'Column "{t_column}" not found in data')
        if day_length <= 0:
            raise ValueError("day_length must be positive")

        # Create mapping dictionary from metadata
        day_dict = self.meta[column].to_dict()

        # Create new dataframe and add shifted times
        new = self.copy(deep=True)
        new['tmp_col'] = new.index.to_series().map(day_dict)
        new[t_column] = new[t_column] + (new['tmp_col'] * (60 * 60 * day_length))
        
        return new.drop(columns=['tmp_col'])

    def curate(self, points: int) -> "behavpy_core":
        """
        Remove specimens that have fewer than the specified minimum number of data points.

        Args:
            points (int): Minimum number of data points a specimen must have to be retained.
                Specimens with fewer points will be removed.

        Returns:
            behavpy_core: New behavpy object with low-data specimens removed from both 
                metadata and data.

        Example:
            # Remove specimens with less than 1000 data points
            df = df.curate(1000)
        """
        # Get IDs of specimens with too few points
        id_counts = self.groupby(level='id').size()
        ids_to_remove = id_counts[id_counts < points].index.tolist()
        
        # Remove specimens with too few points
        return self.remove('id', ids_to_remove)

    @staticmethod
    def _find_runs(mov: pd.Series, time: pd.Series, id: str) -> Dict[str, Any]:
        """
        Find runs of consecutive values in movement data and calculate activity counts.

        Args:
            mov (pd.Series): Series containing movement data (boolean/binary values)
            time (pd.Series): Series containing timestamps
            id (str): Identifier for the specimen

        Returns:
            Dict[str, Any]: Dictionary containing:
                - id: specimen identifier
                - t: timestamps
                - moving: original movement data
                - previous_moving: movement data shifted by 1 position
                - activity_count: consecutive count within each run
                - previous_activity_count: activity count shifted by 1 position

        Notes:
            This is an internal helper method used for analyzing movement patterns.
            The activity count represents how many consecutive values appear in each run.
        """
        _, _, l = rle(mov)
        # count_list = np.concatenate([np.append(np.arange(1, cnt + 1, 1)[: -1], np.nan) for cnt in l], dtype = float)
        count_list = np.concatenate([np.arange(1, cnt + 1, 1) for cnt in l], dtype = float)
        previous_count_list = count_list[:-1]
        previous_count_list = np.insert(previous_count_list, 0, np.nan)
        previous_mov = mov[:-1].astype(float)
        previous_mov = np.insert(previous_mov, 0, np.nan)
        return {'id': id, 't' : time, 'moving' : mov, 'previous_moving' : previous_mov, 'activity_count' : count_list, 'previous_activity_count' : previous_count_list}


    def remove_sleep_deprived(self, start_time: int, end_time: int, remove: Union[bool, float] = False,
                         sleep_column: str = 'asleep', t_column: str = 't') -> Union["behavpy_core", pd.DataFrame]:
        """
        Filter specimens based on their sleep percentage during a specified time period.

        Calculates the percentage of time each specimen spends asleep during the given time window.
        Can either return these percentages or remove specimens that sleep more than a specified threshold.

        Args:
            start_time (int): Start time of analysis window in hours from experiment start
            end_time (int): End time of analysis window in hours from experiment start
            remove (Union[bool, float], optional): If False, returns sleep percentages for all specimens.
                If float between 0 and 1, removes specimens sleeping more than this percentage.
                Defaults to False.
            sleep_column (str, optional): Column containing boolean sleep state data.
                Defaults to 'asleep'.
            t_column (str, optional): Column containing timestamps in seconds.
                Defaults to 't'.

        Returns:
            Union[behavpy_core, pd.DataFrame]: 
                - If remove=False: DataFrame with sleep percentages per specimen
                - If remove=float: behavpy object with high-sleeping specimens removed

        Raises:
            KeyError: If start_time > end_time or start_time < 0
            ValueError: If remove is not False or a float between 0 and 1
            KeyError: If sleep_column or t_column not found in data

        Examples:
            # Get sleep percentages for all specimens
            df.remove_sleep_deprived(12, 24)

            # Remove specimens sleeping more than 30% of time
            df.remove_sleep_deprived(12, 24, remove=0.3)
        """
        # Input validation
        if start_time > end_time:
            raise KeyError('Start time cannot be greater than end time')

        if start_time < 0:
            raise KeyError('Start time cannot be negative')

        if remove is not False and not (isinstance(remove, (int, float)) and 0 <= remove < 1):
            raise ValueError('Remove must be False or a float between 0 and 1')

        # Validate columns exist
        if sleep_column not in self.columns:
            raise KeyError(f"Column '{sleep_column}' not found in data")
        if t_column not in self.columns:
            raise KeyError(f"Column '{t_column}' not found in data")

        # Filter by time period and keep only relevant columns
        fdf = self.t_filter(start_time=start_time, end_time=end_time, 
                        t_column=t_column)[[t_column, sleep_column]]

        # Calculate sleep percentages
        gb = fdf.groupby(fdf.index).agg(**{
            'time asleep': (sleep_column, 'sum'),
            'min t': (t_column, 'min'),
            'max t': (t_column, 'max'),
        })

        # Convert boolean counts to seconds using time step
        tdiff = fdf[t_column].diff().mode()[0]
        gb['time asleep(s)'] = gb['time asleep'] * tdiff
        gb['time(s)'] = gb['max t'] - gb['min t']
        gb['Percent Asleep'] = gb['time asleep(s)'] / gb['time(s)']
        gb.drop(columns=['time asleep', 'max t', 'min t'], inplace=True)

        if remove is False:
            return self.__class__(gb, self.meta, 
                                palette=self.attrs['sh_pal'], 
                                long_palette=self.attrs['lg_pal'], 
                                check=True)
        else:
            remove_ids = gb[gb['Percent Asleep'] > remove].index.tolist()
            return self.remove('id', remove_ids)

    @staticmethod
    def _time_alive(df: pd.DataFrame, facet_col: Optional[str] = None, repeat: bool = False, 
                    t_column: str = 't') -> pd.DataFrame:
        """
        Calculate survival statistics over time for specimens.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing time series data
            facet_col (optional, Optional[str]): Column name to group data by
            repeat (optional, bool): If True, subdivide data by 'repeat' column. Default False
            t_column (optional, str): Column containing timestamps in seconds. Default 't'
        
        Returns:
            pd.DataFrame: DataFrame with columns:
                - hour: Time points in hours
                - survived: Percentage of specimens alive at each hour
                - label: Group identifier from facet_col
                
        Raises:
            KeyError: If required columns are missing
            ValueError: If time data is invalid
        """
        def _wrapped_time_alive(df: pd.DataFrame, name: str) -> pd.DataFrame:
            # Validate input data
            if not len(df):
                raise ValueError("Empty DataFrame provided")
                
            # Calculate survival times
            gb = df.groupby(df.index).agg(**{
                'tmin': (t_column, 'min'),
                'tmax': (t_column, 'max')
            })
            
            # Convert seconds to hours
            gb['time_alive'] = round(((gb['tmax'] - gb['tmin']) / 86400) * 24)
            gb.drop(columns=['tmax', 'tmin'], inplace=True)
            
            # Create survival matrix
            max_time = int(gb['time_alive'].max())

            cols = []
            for _, hours in gb.to_dict()['time_alive'].items():
                y = np.repeat(1, hours)
                cols.append(np.pad(y, (0, max_time - len(y))))

            # Calculate survival percentages
            col = np.vstack(cols).sum(axis=0)
            max_survived = np.max(col)
            if max_survived == 0:
                raise ValueError("No survivors detected")
                
            col = (col / max_survived) * 100

            return pd.DataFrame({
                'hour': range(0, len(col)),
                'survived': col,
                'label': [name] * len(col)
            })

        # Get group name if facet_col is provided
        if facet_col is not None:
            name = df[facet_col].iloc[0]
        else:
            name = ''

        # Process data
        if not repeat:
            return _wrapped_time_alive(df, name)
        
        # Handle repeated measurements
        if 'repeat' not in df.columns:
            raise KeyError("'repeat' column not found for repeat analysis")
            
        results = []
        for rep in df['repeat'].unique():
            try:
                result = _wrapped_time_alive(df[df['repeat'] == rep], name)
                results.append(result)
            except (ValueError, KeyError) as e:
                # Log error but continue processing other repeats
                print(f"Warning: Error processing repeat {rep}: {str(e)}")
                continue
                
        if not results:
            raise ValueError("No valid data found across repeats")
            
        return pd.concat(results, ignore_index=True)

    # GROUPBY SECTION

    @staticmethod
    def _wrapped_bout_analysis(data: pd.DataFrame, var_name: str, as_hist: bool, bin_size: int, max_bins: int, 
                               time_immobile: int, asleep: bool, t_column: str = 't') -> pd.DataFrame:
        """
        Internal method to analyze bouts of activity/inactivity and optionally create histograms.

        Finds runs of consecutive values in behavioral data (e.g., movement or sleep) and calculates 
        bout durations. Can return either raw bout data or histogram data of bout durations.

        Args:
            data (pd.DataFrame): DataFrame containing behavioral data for a single specimen
            var_name (str): Name of column containing boolean behavioral state data
            as_hist (bool): If True, return histogram data of bout durations. If False, return raw bout data
            bin_size (int): Size of histogram bins in minutes (only used if as_hist=True)
            max_bins (int): Maximum number of bins for histogram (only used if as_hist=True)
            time_immobile (int): Minimum bout duration in minutes to include in histogram
            asleep (bool): Which boolean state to analyze in histogram (True=sleep, False=wake)
            t_column (str, optional): Name of column containing timestamps. Defaults to 't'.

        Returns:
            pd.DataFrame: Either:
                - Raw bout data with columns: [var_name, t_column, duration] if as_hist=False
                - Histogram data with columns: [bins, count, prob] if as_hist=True

        Raises:
            KeyError: If var_name or t_column not found in data DataFrame
            ValueError: If data DataFrame is empty or contains data for multiple specimens
            TypeError: If var_name column contains non-boolean values

        Note:
            This is an internal helper method used by sleep_bout_analysis().
            The input DataFrame must contain data for only one specimen.
        """

        index_name = data['id'].iloc[0]
        bin_width = bin_size*60
        
        dt = data[[t_column,var_name]].copy(deep = True)
        dt['deltaT'] = dt[t_column].diff()
        bout_rle = rle(dt[var_name])
        vals = bout_rle[0]
        bout_range = list(range(1,len(vals)+1))

        bout_id = []
        for c, i in enumerate(bout_range):
            bout_id += ([i] * bout_rle[2][c])

        bout_df = pd.DataFrame({'bout_id' : bout_id, 'deltaT' : dt['deltaT']})
        bout_times = bout_df.groupby('bout_id').agg(
        duration = pd.NamedAgg(column='deltaT', aggfunc='sum')
        )
        bout_times[var_name] = vals
        time = np.array([dt[t_column].iloc[0]])
        time = np.concatenate((time, bout_times['duration'].iloc[:-1]), axis = None)
        time = np.cumsum(time)
        bout_times[t_column] = time
        bout_times.reset_index(level=0, inplace=True)
        bout_times.drop(columns = ['bout_id'], inplace = True)
        old_index = pd.Index([index_name] * len(bout_times.index), name = 'id')
        bout_times.set_index(old_index, inplace =True)

        if as_hist is True:

            filtered = bout_times[bout_times[var_name] == asleep]
            filtered['duration_bin'] = filtered['duration'].map(lambda d: bin_width * floor(d / bin_width))
            bout_gb = filtered.groupby('duration_bin').agg(**{
                        'count' : ('duration_bin', 'count')
            })
            bout_gb['prob'] = bout_gb['count'] / bout_gb['count'].sum()
            bout_gb.rename_axis('bins', inplace = True)
            bout_gb.reset_index(level=0, inplace=True)
            old_index = pd.Index([index_name] * len(bout_gb.index), name = 'id')
            bout_gb.set_index(old_index, inplace =True)
            bout_gb = bout_gb[(bout_gb['bins'] >= time_immobile * 60) & (bout_gb['bins'] <= max_bins*bin_width)]

            return bout_gb

        else:
            return bout_times

    def sleep_bout_analysis(self, sleep_column: str = 'asleep', as_hist: bool = False, bin_size: int = 1, 
                       max_bins: int = 60, time_immobile: int = 5, asleep: bool = True, 
                       t_column: str = 't') -> "behavpy_core":
        """
        Analyse sleep/wake bout durations and optionally create histograms.

        Takes a column of boolean sleep/wake states and calculates bout durations. Can either return 
        raw bout data or histogram data of bout durations for analysis.

        Args:
            sleep_column (str, optional): Column containing boolean sleep state data.
                Defaults to 'asleep'.
            as_hist (bool, optional): If True, return histogram data instead of raw bout data.
                Defaults to False.
            bin_size (int, optional): Size of histogram bins in minutes. Only used if as_hist=True.
                Defaults to 1.
            max_bins (int, optional): Maximum number of bins for histogram. Only used if as_hist=True.
                Defaults to 60.
            time_immobile (int, optional): Minimum bout duration in minutes to include in histogram.
                Only used if as_hist=True. Defaults to 5.
            asleep (bool, optional): Which boolean state to analyze in histogram:
                - True: analyze sleep bouts
                - False: analyze wake bouts
                Only used if as_hist=True. Defaults to True.
            t_column (str, optional): Column containing timestamps in seconds.
                Defaults to 't'.

        Returns:
            behavpy_core: New behavpy object containing either:
                - Raw bout data with columns: [sleep_column, t_column, duration] if as_hist=False
                - Histogram data with columns: [bins, count, prob] if as_hist=True

        Raises:
            KeyError: If sleep_column not found in data

        Examples:
            # Get raw bout data
            df.sleep_bout_analysis()

            # Get histogram of sleep bouts
            df.sleep_bout_analysis(as_hist=True, bin_size=5)

            # Get histogram of wake bouts
            df.sleep_bout_analysis(as_hist=True, asleep=False)
        """

        tdf = self.reset_index().copy(deep = True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(partial(self._wrapped_bout_analysis, 
                                                                                                var_name = sleep_column, 
                                                                                                as_hist = as_hist, 
                                                                                                bin_size = bin_size, 
                                                                                                max_bins = max_bins, 
                                                                                                time_immobile = time_immobile, 
                                                                                                asleep = asleep,
                                                                                                t_column = t_column
            )), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    @staticmethod
    def _wrapped_curate_dead_animals(data: pd.DataFrame, time_var: str, moving_var: str, time_window: int, 
                                    prop_immobile: float, resolution: int, 
                                    time_dict: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
        """
        Internal method to detect and remove data after presumed death for a single specimen.

        Analyses movement data within sliding time windows to identify when an animal likely died,
        based on extended periods of immobility. Data after the first detected death point is removed.

        Args:
            data (pd.DataFrame): DataFrame containing time series data for a single specimen
            time_var (str): Name of column containing timestamps in seconds
            moving_var (str): Name of column containing movement data (typically boolean or numeric)
            time_window (int): Size of sliding window in hours to check for immobility
            prop_immobile (float): Threshold of mean movement below which animal is considered dead
            resolution (int): Number of segments to divide each time window into
            time_dict (Optional[Dict[str, List[int]]], optional): Dictionary to store valid time ranges.
                Keys are specimen IDs, values are [start_time, death_time]. Defaults to None.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only data before detected death point

        Note:
            This is an internal helper method used by curate_dead_animals().
            The time_window is converted from hours to seconds internally.
        """

        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if resolution > time_window:
            raise ValueError("resolution cannot be larger than time_window")

        time_window = (60 * 60 * time_window)
        d = data[[time_var, moving_var]].copy(deep = True)
        target_t = np.array(list(range(d[time_var].min().astype(int), d[time_var].max().astype(int), floor(time_window / resolution))))
        local_means = np.array([d[d[time_var].between(i, i + time_window)][moving_var].mean() for i in target_t])

        # Find indices where animal is considered dead
        death_points = np.where(local_means <= prop_immobile)[0]

        # If no death points found, return original data
        if len(death_points) == 0:
            if time_dict is not None:
                time_dict[data['id'].iloc[0]] = [data[time_var].min(), data[time_var].max()]
            return data

        # Get first death point
        first_death_time = target_t[death_points[0]]
        
        if time_dict is not None:
            time_dict[data['id'].iloc[0]] = [data[time_var].min(), first_death_time]
            
        return data[data[time_var].between(data[time_var].min(), first_death_time)]
    
    def curate_dead_animals(self, t_column: str = 't', mov_column: str = 'moving', time_window: int = 24, 
                            prop_immobile: float = 0.01, resolution: int = 24) -> "behavpy_core":
        """
        Detect and remove data after specimens are presumed dead based on extended immobility.

        Analyzes movement data within sliding time windows to identify when specimens likely died,
        based on the proportion of time spent immobile. Data after the first detected death point
        is removed.

        Args:
            t_column (str, optional): Column containing timestamps in seconds.
                Defaults to 't'.
            mov_column (str, optional): Column containing boolean movement data.
                Defaults to 'moving'.
            time_window (int, optional): Size of sliding window in hours to check for immobility.
                Defaults to 24.
            prop_immobile (float, optional): Proportion of time window that must be immobile
                to consider specimen dead (0-1). Defaults to 0.01 (1%).
            resolution (int, optional): Number of segments to divide each time window into.
                Controls overlap between windows. Defaults to 24.

        Returns:
            behavpy_core: Filtered behavpy object containing only data before detected death points.

        Raises:
            KeyError: If t_column or mov_column not found in data
            ValueError: If resolution is not positive or larger than time_window
            TypeError: If mov_column does not contain boolean values
        """
        # # Check
        # if not pd.api.types.is_bool_dtype(self[mov_column]):
        #     raise TypeError(f'Column {mov_column} must contain boolean values')

        tdf = self.reset_index().copy(deep=True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(
            partial(self._wrapped_curate_dead_animals,
                time_var = t_column,
                moving_var = mov_column,
                time_window = time_window, 
                prop_immobile = prop_immobile,
                resolution = resolution)
            ), 
        tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    def curate_dead_animals_interactions(self, mov_df: "behavpy_core", t_column: str = 't', mov_column: str = 'moving', 
                                         time_window: int = 24, prop_immobile: float = 0.01, 
                                         resolution: int = 24) -> Tuple["behavpy_core", "behavpy_core"]:
        """
        Remove interaction data after specimens are presumed dead based on movement data.

        A variation of curate_dead_animals that filters both an interaction dataset and its
        corresponding movement dataset. Uses movement data to detect death points and removes
        data after those points from both datasets.

        Args:
            mov_df (behavpy_core): Behavpy object containing movement time series data for
                specimens in the interaction dataset.
            t_column (str, optional): Column containing timestamps in seconds.
                Defaults to 't'.
            mov_column (str, optional): Column containing boolean movement data.
                Must exist in mov_df. Defaults to 'moving'.
            time_window (int, optional): Size of sliding window in hours to check for immobility.
                Defaults to 24.
            prop_immobile (float, optional): Proportion of time window that must be immobile
                to consider specimen dead (0-1). Defaults to 0.01 (1%).
            resolution (int, optional): Number of segments to divide each time window into.
                Controls overlap between windows. Defaults to 24.

        Returns:
            Tuple[behavpy_core, behavpy_core]: Tuple containing:
                - Filtered interaction dataset
                - Filtered movement dataset
                Both filtered to remove data after detected death points.

        Raises:
            KeyError: If t_column not found in either dataset or mov_column not found in mov_df
        """

        # function that uses time filter dict produced from _wrapped_curate_dead_animals
        def curate_filter(df: pd.DataFrame, dict: Dict[str, List[int]]) -> pd.DataFrame:
            specimen_id = df['id'].iloc[0]
            if specimen_id not in dict:
                return pd.DataFrame()  # Return empty frame for missing IDs
            return df[df[t_column].between(dict[specimen_id][0], dict[specimen_id][1])]

        # # Validate movement column contains boolean values
        # if not pd.api.types.is_bool_dtype(mov_df[mov_column]):
        #     raise TypeError(f'Column {mov_column} must contain boolean values')

        # Check ID consistency
        interaction_ids = set(self.index.unique())
        movement_ids = set(mov_df.index.unique())
        if not interaction_ids.issubset(movement_ids):
            raise ValueError("Some specimens in interaction dataset not found in movement dataset")

        tdf = self.reset_index().copy(deep=True)
        tdf2 = mov_df.reset_index().copy(deep=True)

        time_dict = {}
        curated_df = tdf2.groupby('id', group_keys=False).apply(
            partial(self._wrapped_curate_dead_animals,
                time_var=t_column,
                moving_var=mov_column,
                time_window=time_window, 
                prop_immobile=prop_immobile,
                resolution=resolution, 
                time_dict=time_dict)
        )

        curated_puff = tdf.groupby('id', group_keys = False, sort = False).apply(
            partial(curate_filter, dict = time_dict)
            )
        
        return (self.__class__(curated_puff, tdf.meta, palette=self.attrs['sh_pal'], 
                              long_palette=self.attrs['lg_pal'],check = True), 
                self.__class__(curated_df, tdf2.meta, palette=self.attrs['sh_pal'], 
                               long_palette=self.attrs['lg_pal'],check = True))

    @staticmethod
    def _wrapped_interpolate_lin(data: pd.DataFrame, var: str, step: int, t_col: str = 't') -> pd.DataFrame:
        """
        Internal helper method to linearly interpolate time series data for a single specimen.

        Takes a DataFrame containing time series data for one specimen and creates a new 
        regularly-spaced time series with linear interpolation of the variable values.

        Args:
            data (pd.DataFrame): DataFrame containing data for a single specimen
            var (str): Name of column containing values to interpolate
            step (int): Time step size in seconds between interpolated points
            t_col (str, optional): Name of column containing timestamps in seconds. 
                Defaults to 't'.

        Returns:
            pd.DataFrame: DataFrame with interpolated values containing columns:
                - id: Specimen identifier
                - t_col: Regular time series
                - var: Interpolated values

            None if resulting time series would have fewer than 3 points.
        """

        # Get specimen ID and create regular time series
        id = data['id'].iloc[0]
        sample_seq = np.arange(min(data[t_col]), np.nanmax(data[t_col]) + step, step)

        # Return None if too few points
        if len(sample_seq) < 3:
            return None

        # Perform interpolation
        f = np.interp(sample_seq, data[t_col].to_numpy(), data[var].to_numpy())

        return pd.DataFrame({
            'id': [id] * len(sample_seq), 
            t_col: sample_seq, 
            var: f
        })

    def interpolate_linear(self, variable: str, step_size: int, t_column: str = 't') -> "behavpy_core":
        """
        Linearly interpolate time series data to create regularly-spaced observations.

        Creates a new time series with regular intervals by binning the original data and
        then interpolating values between points. The data must be numeric (int/float) and
        approximately linear.

        Args:
            variable (str): Name of column containing values to interpolate
            step_size (int): Time step size in seconds between interpolated points.
                For example, 60 would create points at [0, 60, 120, 180, ...]
            t_column (str, optional): Name of column containing timestamps in seconds.
                Defaults to 't'.

        Returns:
            behavpy_core: New behavpy object containing interpolated data with columns:
                - id: Specimen identifier 
                - t_column: Regular time series
                - variable: Interpolated values

        Raises:
            KeyError: If variable or t_column not found in data
            ValueError: If step_size is not positive
            TypeError: If variable column contains non-numeric data

        Note:
            - Ethoscopy uses numpy's interp function to interpolate, however pandas does 
                have its own implementation of an interpolation method -> .interpolate()
                which can make use of different methods.
                See pandas's documentation on its use if interested.

        Example:
            # Interpolate 'distance' values to 1-minute intervals
            df = df.interpolate_linear('distance', step_size=60)
        """
        # Input validation
        if step_size <= 0:
            raise ValueError("Step size must be positive")

        # Create copy and bin data
        data = self.copy(deep=True)
        data = data.bin_time(variable=variable, t_column=t_column, bin_secs=step_size)
        
        # Rename columns to match expected format
        data = data.rename(columns={
            f'{t_column}_bin': t_column,
            f'{variable}_mean': variable
        })
        
        # Reset index and apply interpolation by specimen
        data = data.reset_index()
        return self.__class__(
            data.groupby('id', group_keys=False).apply(
                partial(self._wrapped_interpolate_lin,
                       var=variable,
                       step=step_size,
                       t_col=t_column)
            ),
            data.meta,
            palette=self.attrs['sh_pal'],
            long_palette=self.attrs['lg_pal'],
            check=True
        )

    @staticmethod
    def _wrapped_bin_data(data: pd.DataFrame, column: Union[str, List[str]], bin_column: str, 
                        function: Union[str, Callable], bin_secs: int) -> pd.DataFrame:
        """
        Internal helper method to bin time series data and apply aggregation functions.
        
        Takes a DataFrame containing data for a single specimen and bins the time series data
        into fixed-width intervals, then applies an aggregation function to specified columns.

        Args:
            data (pd.DataFrame): DataFrame containing data for a single specimen
            column (Union[str, List[str]]): Column name(s) to aggregate within bins
            bin_column (str): Name of column containing timestamps to bin
            function (Union[str, Callable]): Aggregation function to apply to binned data.
                Can be string name of pandas aggregation function or custom callable.
            bin_secs (int): Size of time bins in seconds

        Returns:
            pd.DataFrame: Binned and aggregated data with:
                - Index: Specimen ID repeated for each bin
                - Columns: 
                    - {bin_column}_bin: Start time of each bin
                    - {column}_{function}: Aggregated values for each column

        Note:
            This is an internal helper method used by bin_time().
            The input DataFrame must contain data for only one specimen.
        """

        index_name = data['id'].iloc[0]

        # Create bins
        data[bin_column] = data[bin_column].map(lambda t: bin_secs * floor(t / bin_secs))

        # Set up aggregation
        agg_dict = dict.fromkeys(column, function)
        name_dict = {n: f'{n}_{function.__name__ if callable(function) else function}' 
                     for n in column}

        # Perform binning and aggregation
        binned_data = data.groupby(bin_column).agg(agg_dict)

        # Format output
        binned_data = binned_data.rename(name_dict, axis=1)
        bin_parse_name = f'{bin_column}_bin'
        binned_data.rename_axis(bin_parse_name, inplace=True)
        binned_data.reset_index(level=0, inplace=True)
        
        # Restore specimen ID index
        specimen_index = pd.Index([index_name] * len(binned_data.index), name='id')
        binned_data.set_index(specimen_index, inplace=True)

        return binned_data

    def bin_time(self, variable: Union[str, List[str]], bin_secs: int, 
                function: Union[str, Callable] = 'mean', t_column: str = 't') -> "behavpy_core":
        """
        Bin time series data into fixed intervals and apply aggregation functions.

        Groups data points into time bins of specified size and applies an aggregation 
        function to selected columns within each bin. Useful for downsampling data or
        analyzing behavior over fixed time intervals.

        Args:
            variable (Union[str, List[str]]): Name of column(s) to aggregate within bins.
                Can be single column name or list of column names.
            bin_secs (int): Size of time bins in seconds.
                For example, 60 would create one-minute bins.
            function (Union[str, Callable], optional): Aggregation function to apply to binned data.
                Can be:
                - String name of pandas aggregation function (e.g. 'mean', 'sum', 'max')
                - Custom callable that takes a Series and returns a single value
                Defaults to 'mean'.
            t_column (str, optional): Name of column containing timestamps in seconds.
                Defaults to 't'.

        Returns:
            behavpy_core: New behavpy object containing binned data with columns:
                - {t_column}_bin: Start time of each bin
                - {variable}_{function}: Aggregated values for each variable

        Raises:
            KeyError: If any specified variable or t_column not found in data
            ValueError: If bin_secs is not positive or if function is invalid
            TypeError: If variable is not str or list of str

        Example:
            # Bin data into 5-minute intervals and take mean
            df = df.bin_time('distance', bin_secs=300)
            
            # Bin multiple columns with custom function
            df = df.bin_time(['x', 'y'], bin_secs=60, function=lambda x: x.max() - x.min())
        """
        # Validate bin_secs
        if not isinstance(bin_secs, int) or bin_secs <= 0:
            raise ValueError("bin_secs must be a positive number and an integer")
        
        if isinstance(variable, str):
            variable = [variable]

        # Validate column names        
        for var in variable:
            if not isinstance(var, str):
                raise TypeError(f"All variables must be strings, got {type(var)} for variable: {var}")
            if var not in self.columns:
                raise KeyError(f"Column '{var}' not found in data")

        # Validate function
        if isinstance(function, str):
            valid_funcs = ['mean', 'sum', 'max', 'min', 'std', 'var', 'count']
            if function not in valid_funcs:
                raise ValueError(f"Invalid string function. Must be one of: {valid_funcs}")
        elif not callable(function):
            raise ValueError("function must be a string or callable")

        # Create copy and apply binning
        tdf = self.reset_index().copy(deep=True)
        
        try:
            binned = self.__class__(
                tdf.groupby('id', group_keys=False).apply(
                    partial(self._wrapped_bin_data,
                        column=variable,
                        bin_column=t_column,
                        function=function,
                        bin_secs=bin_secs)
                ),
                tdf.meta,
                palette=self.attrs['sh_pal'],
                long_palette=self.attrs['lg_pal'],
                check=True
            )
                
            return binned
            
        except Exception as e:
            raise ValueError(f"Error during binning operation: {str(e)}") from e
        
    def remove_first_last_bout(self, variable: str) -> "behavpy_core":
        """
        Remove the first and last bouts of a value per specimen.

        Used for columns containing continuous runs of categorical integer values, such as bools,
        to remove potentially incomplete bouts at the start and end of recordings. This is useful
        when you are not sure if the starting and ending bouts were cut in two when filtering
        or stopping the experiment.

        Args:
            variable (str): Column containing boolean state data to analyze.
                Must exist in DataFrame and contain boolean values.

        Returns:
            behavpy_core: New behavpy object with first and last bouts removed from each specimen.

        Raises:
            KeyError: If variable column not found in data

        Example:
            # Remove first/last sleep bouts that may be incomplete
            df = df.remove_first_last_bout('asleep')
        """
        
        def _wrapped_remove_first_last_bout(data: pd.DataFrame) -> pd.DataFrame:
            if len(data) == 0:
                return data
                
            v = data[variable].tolist()
            
            # Find indices where state changes
            try:
                change_list = np.where(np.roll(v,1)!=v)[0]
                ind1 = change_list[0]
                ind2 = change_list[-1]
            except IndexError:
                # Return original data if no changes found
                return data
                
            return data.iloc[ind1:ind2]

        tdf = self.reset_index().copy(deep=True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(_wrapped_remove_first_last_bout), 
                tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    @staticmethod
    def _wrapped_motion_detector(data: pd.DataFrame, time_window_length: int, velocity_correction_coef: float, 
                            masking_duration: int, velocity_threshold: float, walk_threshold: float, 
                            ) -> pd.DataFrame:
        """
        Internal wrapper for motion detection processing on individual specimens. 
        See the docstring for motion_detector() for more details.
        
        Returns:
            pd.DataFrame: Processed movement data with specimen ID index
        """
        index_name = data['id'].iloc[0]
        
        df = max_velocity_detector(data,                                   
                            time_window_length=time_window_length, 
                            velocity_correction_coef=velocity_correction_coef, 
                            masking_duration=masking_duration,
                            velocity_threshold=velocity_threshold,
                            walk_threshold=walk_threshold)

        old_index = pd.Index([index_name] * len(df.index), name='id')
        df.set_index(old_index, inplace=True)  

        return df   

    def motion_detector(self, time_window_length: int = 10, velocity_correction_coef: float = 3e-3, 
                    masking_duration: int = 6, velocity_threshold: float = 1.0, 
                    walk_threshold: float = 2.5) -> "behavpy_core":
        """
        Method version of the motion detector for classifying different types of movement in ethoscope experiments.
        Benchmarked against human-generated ground truth.

        Args:
            time_window_length (int, optional): Period of time the data is binned and sampled to. Default is 10.
            velocity_correction_coef (float, optional): Coefficient to correct velocity data. Use 3e-3 for 'small' tubes 
                (20 per ethoscope), 15e-4 for 'long' tubes (10 per ethoscope). Default is 3e-3.
            masking_duration (int, optional): Seconds during which movement is ignored after stimulus. Default is 6.
            velocity_threshold (float, optional): Threshold above which movement is detected. Default is 1.0.
            walk_threshold (float, optional): Threshold above which movement is classified as walking. Default is 2.5.

        Returns:
            behavpy_core: A behavpy object with added columns for movement classifications including:
                - moving: Boolean indicating if movement was detected
                - micro: Boolean for micro-movements (velocity between thresholds)
                - walk: Boolean for walking movement (above walk threshold)
                - beam_crosses: Count of arena center crossings
                - Additional metrics like velocity and distance
        
        Note:
            The given data must contain the following columns:
                - 't
                - 'x'
                - 'y'
                - 'w'
                - 'h'
                - 'phi'
                - 'xy_dist_log10x1000'
            Optional columns:
                - 'has_interacted'
        """

        tdf = self.reset_index().copy(deep=True)
        return self.__class__(tdf.groupby('id', group_keys=False).apply(partial(self._wrapped_motion_detector,
                                                                            time_window_length=time_window_length,
                                                                            velocity_correction_coef=velocity_correction_coef,
                                                                            masking_duration=masking_duration,
                                                                            velocity_threshold=velocity_threshold,
                                                                            walk_threshold=walk_threshold,
        )), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check=True)

    @staticmethod
    def _wrapped_sleep_contiguous(d_small: pd.DataFrame, mov_column: str, t_column: str, time_window_length: int, 
                                  min_time_immobile: int) -> pd.DataFrame:
        """
        Internal wrapper for sleep analysis processing on individual specimens.
        See the docstring for sleep_contiguous() for more details.

        Returns:
            pd.DataFrame: Processed sleep data with specimen ID index
        """
        def _sleep_contiguous(moving: pd.Series, fs: int, min_valid_time: int = 300) -> List[bool]:
            """ 
            Checks if contiguous bouts of immobility are greater than the minimum valid time given

            Args:
                moving (pandas series): series object containing the movement data of individual flies
                fs (int): sampling frequency (Hz) to scale minimum time immobile to number of data points
                min_valid_time (int, optional): min amount immobile time that counts as sleep, default is 300 (i.e 5 mins) 
            
            returns: 
                A list object to be added to a pandas dataframe
            """
            min_len = fs * min_valid_time
            r_sleep =  rle(np.logical_not(moving)) 
            valid_runs = r_sleep[2] >= min_len 
            r_sleep_mod = valid_runs & r_sleep[0]
            r_small = []
            for c, i in enumerate(r_sleep_mod):
                r_small += ([i] * r_sleep[2][c])

            return r_small

        time_map = pd.Series(range(d_small[t_column].iloc[0], 
                            d_small[t_column].iloc[-1] + time_window_length, 
                            time_window_length
                            ), name = t_column)

        missing_values = time_map[~time_map.isin(d_small[t_column].tolist())]
        d_small = d_small.merge(time_map, how = 'right', on = t_column, copy = False).sort_values(by=[t_column])
        d_small['is_interpolated'] = np.where(d_small[t_column].isin(missing_values), True, False)
        d_small[mov_column] = np.where(d_small['is_interpolated'] == True, False, d_small[mov_column])

        d_small['asleep'] = _sleep_contiguous(d_small[mov_column], 1/time_window_length, min_valid_time = min_time_immobile)
        d_small['id'] = [d_small['id'].iloc[0]] * len(d_small)
        d_small.set_index('id', inplace = True)

        return d_small  

    def sleep_contiguous(self, mov_column: str = 'moving', t_column: str = 't', time_window_length: int = 10, 
                         min_time_immobile: int = 300) -> "behavpy_core":
        """
        Analyse movement data to identify sleep periods based on sustained immobility.

        This method processes movement data to identify sleep periods by:
        1. Binning data into fixed time windows
        2. Interpolating any missing time points
        3. Identifying contiguous periods of immobility
        4. Marking periods as sleep if they exceed the minimum duration threshold

        Args:
            mov_column (str, optional): Column containing boolean movement data.
                True indicates movement, False indicates immobility. Defaults to 'moving'.
            t_column (str, optional): Column containing timestamps in seconds.
                Defaults to 't'.
            time_window_length (int, optional): Size of time windows in seconds for binning data.
                Defaults to 10.
            min_time_immobile (int, optional): Minimum duration in seconds of immobility
                required to classify a period as sleep. Defaults to 300 (5 minutes).

        Returns:
            behavpy_core: New behavpy object with additional columns:
                - is_interpolated: Boolean indicating interpolated time points
                - asleep: Boolean indicating sleep state

        Raises:
            KeyError: If mov_column or t_column not found in data
            ValueError: If time_window_length or min_time_immobile is not a positive number

        Example:
            # Identify sleep with 1-minute windows and 5-minute threshold
            df = df.sleep_contiguous(time_window_length=60, min_time_immobile=300)
        """

        # Validate time_window_length
        if not isinstance(time_window_length, int) or time_window_length <= 0:
            raise ValueError("time_window_length must be a positive number")
        # Validate min_time_immobile
        if not isinstance(min_time_immobile, int) or min_time_immobile <= 0:
            raise ValueError("min_time_immobile must be a positive number")
        
        tdf = self.reset_index().copy(deep = True)
        return self.__class__(tdf.groupby('id', group_keys = False).apply(
            partial(self._wrapped_sleep_contiguous,
                        mov_column = mov_column,
                        t_column = t_column,
                        time_window_length = time_window_length,
                        min_time_immobile = min_time_immobile
            )
        ), tdf.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    def feeding(self, food_position: str, dist_from_food: float = 0.05, micro_mov: str = 'micro', 
                left_rois: List[int] = [1,2,3,4,5,6,7,8,9,10], right_rois: List[int] = [11,12,13,14,15,16,17,18,19,20], 
                add_walk: bool = False, x_position: str = 'x') -> "behavpy_core":
        """
        Estimates feeding behavior based on micro-movements near food sources in ethoscope data.

        Analyses position and movement data to identify potential feeding events, defined as 
        micro-movements occurring within a specified distance from food sources. The method 
        handles asymmetric arena layouts by normalizing x-positions relative to food location.

        Args:
            food_position (str): Position of food relative to arena center. Must be either:
                - "outside": Food is at the ends of the tubes
                - "inside": Food is near the center of the tubes
            dist_from_food (float, optional): Distance threshold from food source, as fraction 
                of tube length (0-1). Defaults to 0.05 (5% of tube length).
            micro_mov (str, optional): Column name containing boolean micro-movement data.
                Defaults to 'micro'.
            left_rois (List[int], optional): ROI IDs for left side of ethoscope. For standard 
                20-tube setup, defaults to [1-10].
            right_rois (List[int], optional): ROI IDs for right side of ethoscope. For standard
                20-tube setup, defaults to [11-20].
            add_walk (bool, optional): If True, uses 'moving' column instead of micro-movements
                to detect feeding. Use when data resolution may miss brief micro-movements.
                Defaults to False.
            x_position (str, optional): Column name containing x-position data.
                Defaults to 'x'.

        Returns:
            behavpy_core: New behavpy object with additional 'feeding' column containing boolean
                values (True = predicted feeding event).

        Raises:
            ValueError: If food_position is not 'outside' or 'inside'
            KeyError: If required columns are missing from data
            ValueError: If dist_from_food is not between 0 and 1
            ValueError: If ROI lists are empty or contain invalid IDs

        Notes:
            - This method modifies x-position values for right-side ROIs by flipping them (1-x).
              Only call once on a dataset.
            - Feeding events are inferred from proximity to food AND micro-movements/movement.
            - Not ground-truthed - predictions are based on behavioral assumptions.
            - May underestimate feeding when using binned data (e.g., 10s or 60s bins) due to
              loss of brief micro-movement sequences.
        """

        # Validate food_position input
        if food_position != 'outside' and food_position != 'inside':
            raise ValueError("Argument for food_position must be 'outside' or 'inside'")

        # Validate dist_from_food range
        if not 0 <= dist_from_food <= 1:
            raise ValueError("dist_from_food must be between 0 and 1")

        # Validate ROI lists
        if not left_rois or not right_rois:
            raise ValueError("ROI lists cannot be empty")
        if not all(isinstance(x, int) for x in left_rois + right_rois):
            raise ValueError("ROI IDs must be integers")

        ds = self.copy(deep = True)
        
        # normalise x values on the right hand side of the ethoscope
        roi_list = set(ds.meta['region_id'])
        # get all rois in the metadata that match the given lists
        l_roi =  [elem for elem in roi_list if elem in left_rois]
        r_roi = [elem for elem in roi_list if elem in right_rois]
        ds_l = ds.xmv('region_id', l_roi)
        ds_r = ds.xmv('region_id', r_roi)
        # flip x values so that both have x=0 on the left handside
        ds_r[x_position] = 1 - ds_r[x_position]
        ds = concat(ds_l, ds_r)
        
        if add_walk is True:
            micro_mov = 'moving'

        def find_feed(d: pd.DataFrame) -> pd.DataFrame:
            """
            Internal function to find feeding events in a single specimen.
            """
            # if there's only 1 data point then add the column and fill with nan value
            if len(d) < 2:
                d['feeding'] = [np.nan]
                return d
            
            # remove x position values that are greater than 3 SD from the mean with zscore analysis and get max and min values
            x_array = d[x_position].to_numpy()
            mask = np.abs(zscore(x_array, nan_policy='omit')) < 3
            x_array = x_array[mask]
            # if the zscore reduces the number below 2, fill with NaNs
            if len(x_array) <= 2:
                d['feeding'] = [np.nan] * len(d)
            x_min = np.min(x_array)
            x_max = np.max(x_array) 
            
            # if the fly is near to the food and mirco moving, then they are assumed to be feeding
            if food_position == 'outside':
                d['feeding'] = np.where((d[x_position] < x_min+dist_from_food) & (d[micro_mov] == True), True, False)
            elif food_position == 'inside':
                d['feeding'] = np.where((d[x_position] > x_max-dist_from_food) & (d[micro_mov] == True), True, False)
            return d
            
        ds.reset_index(inplace = True)   
        ds_meta = ds.meta
        return self.__class__(ds.groupby('id', group_keys = False).apply(find_feed), ds_meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    # HMM SECTION

    @staticmethod
    def _hmm_decode(d: pd.DataFrame, h: Any, b: int, var: str, fun: Union[str, Callable], 
                    t: str, return_type: str = 'array') -> Union[Tuple[List[np.ndarray], List[pd.Series]], pd.DataFrame]:
        """
        Internal method to decode hidden states from time series data using a trained HMM.

        Takes a DataFrame containing behavioral data for a single specimen and uses a trained
        Hidden Markov Model to decode the most likely sequence of hidden states. The data is
        first binned into fixed time intervals before decoding.

        Args:
            d (pd.DataFrame): DataFrame containing behavioral data for a single specimen
            h (Any): Trained HMM object (typically hmmlearn.hmm.CategoricalHMM)
            b (int): Size of time bins in seconds for aggregating data
            var (str): Name of column containing behavioral state data
            fun (Union[str, Callable]): Aggregation function to apply when binning data.
                Can be string name of pandas function or custom callable.
            t (str): Name of column containing timestamps in seconds
            return_type (str, optional): Format of returned data. Must be either:
                - 'array': Returns tuple of (states_list, time_list)
                - 'table': Returns DataFrame with decoded states
                Defaults to 'array'.

        Returns:
            Either:
                - If return_type='array': Tuple containing:
                    - states_list: List of numpy arrays containing decoded states
                    - time_list: List of pandas Series containing timestamps
                - If return_type='table': DataFrame with columns:
                    - id: Specimen identifier
                    - bin: Time bin start time
                    - state: Decoded state
                    - previous_state: Previous decoded state (NaN for first point)
                    - {var}: Original behavioral state
                    - previous_{var}: Previous behavioral state (NaN for first point)

        Raises:
            TypeError: If input DataFrame is empty or HMM model has no .decode() attribute
            ValueError: If return_type is not 'array' or 'table'
        Notes:
            - For 'moving' or 'asleep' columns, True/False values are converted to 1/0
            - This is an internal helper method used by get_hmm_raw()
            - The input DataFrame must contain data for only one specimen
        """
        # Validate inputs
        if return_type not in ['array', 'table']:
            raise ValueError("return_type must be either 'array' or 'table'")
            
        if not hasattr(h, 'decode'):
            raise TypeError("HMM object must have decode() method")

        # Check for empty DataFrame
        if len(d) == 0:
            raise TypeError('Given dataframe has no values')

        # Convert boolean columns to integers
        if var in ['moving', 'asleep']:
            d[var] = np.where(d[var] == True, 1, 0)
        
        # bin the data to X second intervals with a selected column and function on that column
        try:
            bin_df = d.bin_time(var, b, t_column=t, function=fun)
        except Exception as e:
            raise ValueError(f"Error during data binning: {str(e)}") from e
        gb = bin_df.groupby(bin_df.index, sort=False)[f'{var}_{fun}'].apply(list)
        time_list = bin_df.groupby(bin_df.index, sort=False)[f'{t}_bin'].apply(list)

        # logprob_list = []
        states_list = []
        df_list = []

        for i, t, id in zip(gb, time_list, time_list.index):
            seq_o = np.array(i)
            seq = seq_o.reshape(-1, 1)
            logprob, states = h.decode(seq)
            #logprob_list.append(logprob)
            if return_type == 'array':
                states_list.append(states)
            if return_type == 'table':
                label = [id] * len(t)
                previous_state = np.array(states[:-1], dtype = float)
                previous_state = np.insert(previous_state, 0, np.nan)
                previous_var = np.array(seq_o[:-1], dtype = float)
                previous_var = np.insert(previous_var, 0, np.nan)
                all_ar = zip(label, t, states, previous_state, seq_o, previous_var)
                df_list.append(pd.DataFrame(data = all_ar))

        if return_type == 'array':
            return states_list, time_list #, logprob_list
        if return_type == 'table':
            df = pd.concat(df_list)
            df.columns = ['id', 'bin', 'state', 'previous_state', var, f'previous_{var}']
            return df

    @staticmethod
    def hmm_display(hmm: Any, states: List[str], observables: List[str]) -> None:
        """
        Display probability tables for a trained Hidden Markov Model.

        Prints formatted tables showing the starting probabilities, transition probabilities,
        and emission probabilities for a trained HMM. Tables are formatted using the tabulate
        package with GitHub-style formatting.

        Args:
            hmm (Any): Trained HMM object (typically hmmlearn.hmm.CategoricalHMM) containing:
                - startprob_: Starting state probabilities
                - transmat_: State transition probabilities 
                - emissionprob_: Emission probabilities
            states (List[str]): Names of hidden states in order. Length must match number
                of states in HMM.
            observables (List[str]): Names of observable states in order. Length must match
                number of possible emissions in HMM.

        Returns:
            None: Prints three tables to screen:
                - Starting probability table
                - Transition probability table 
                - Emission probability table

        Raises:
            AttributeError: If HMM object missing required probability matrices
            ValueError: If lengths of states/observables don't match HMM dimensions

        Example:
            # Print probability tables for trained HMM
            states = ['rest', 'active']
            observables = ['still', 'moving'] 
            behavpy.hmm_display(trained_hmm, states, observables)
        """

        df_s = pd.DataFrame(hmm.startprob_)
        df_s = df_s.T
        df_s.columns = states
        print("Starting probabilty table: ")
        print(tabulate(df_s, headers = 'keys', tablefmt = "github") + "\n")
        print("Transition probabilty table: ")
        df_t = pd.DataFrame(hmm.transmat_, index = states, columns = states)
        print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
        print("Emission probabilty table: ")
        df_e = pd.DataFrame(hmm.emissionprob_, index = states, columns = observables)
        print(tabulate(df_e, headers = 'keys', tablefmt = "github") + "\n")

    def hmm_train(self, states: List[str], observables: List[str], var_column: str, file_name: str, 
                  trans_probs: Optional[np.ndarray] = None, emiss_probs: Optional[np.ndarray] = None, 
                  start_probs: Optional[np.ndarray] = None, iterations: int = 10, hmm_iterations: int = 100, 
                  tol: float = 10.0, t_column: str = 't', bin_time: int = 60, test_size: int = 10, verbose: bool = False) -> "hmm.CategoricalHMM":
        """
        Train a Hidden Markov Model on behavioral time series data using hmmlearn.

        Uses the hmmlearn.hmm.CategoricalHMM implementation to learn state transitions and emission 
        probabilities from categorical observation sequences. Performs multiple training iterations 
        with different random initializations and keeps the best performing model based on test set 
        log-likelihood.

        Args:
            states (List[str]): Names of hidden states to learn (e.g. ['rest', 'active']).
                Number of states determines n_components in the HMM.
            observables (List[str]): Names of observable states (e.g. ['still', 'moving']).
                Must match the number of unique values in var_column after preprocessing.
            var_column (str): Column containing categorical observations to train on.
                Special handling for 'moving' and 'beam_crosses' boolean values which are converted to 0/1.
            file_name (str): Path to save trained model (.pkl extension required).
                Best performing model across iterations will be saved here.
            trans_probs (Optional[np.ndarray], optional): Initial transition probability matrix (n_states x n_states).
                Use 0 to restrict transitions, 'rand' for random initialization.
                If None, randomly initialized. Default None.
            emiss_probs (Optional[np.ndarray], optional): Initial emission probability matrix (n_states x n_observables).
                Use 0 to restrict emissions, 'rand' for random initialization.
                If None, randomly initialized. Default None.
            start_probs (Optional[np.ndarray], optional): Initial starting probability vector (n_states).
                Use 0 to restrict starts, 'rand' for random initialization.
                If None, randomly initialized. Default None.
            iterations (int, optional): Number of training runs with different random initializations.
                Best model across iterations is kept. Default 10.
            hmm_iterations (int, optional): Maximum number of EM iterations per training run.
                Passed to hmmlearn as n_iter. Default 100.
            tol (float, optional): Log-likelihood convergence threshold for EM algorithm.
                Training stops when gain is below this value. Default 10.
            t_column (str, optional): Column containing timestamps in seconds. 
                Default 't'.
            bin_time (int, optional): Time bin size in seconds for aggregating observations.
                Data is binned before training. 
                Default 60.
            test_size (int, optional): Percentage of sequences to use for test set.
                Used to evaluate models across iterations. Default 10.
            verbose (bool, optional): Whether to print convergence information. 
                Default False.

        Returns:
            hmmlearn.hmm.CategoricalHMM: Trained HMM with best test set performance.
                Model is also saved to file_name.

        Raises:
            TypeError: If file_name doesn't end with .pkl
            KeyError: If required columns not found in data
            ValueError: If probability matrices have invalid shapes/values

        Notes:
            - Data is binned and converted to sequences before training
            - Test/train split is done at the sequence level
            - NaN values must be handled before training
            - Uses CategoricalHMM which expects discrete observations
            - Probability matrices are normalized if provided
            - Best model selected using test set log-likelihood
        """
        
        if file_name.endswith('.pkl') is False:
            raise TypeError('enter a file name and type (.pkl) for the hmm object to be saved under')

        n_states = len(states)
        n_obs = len(observables)

        # Validate 
        if trans_probs is not None and trans_probs.shape != (n_states, n_states):
            raise ValueError(f"trans_probs must have shape ({n_states}, {n_states})")
        if emiss_probs is not None and emiss_probs.shape != (n_states, n_obs):
            raise ValueError(f"emiss_probs must have shape ({n_states}, {n_obs})")
        if start_probs is not None and start_probs.shape != (n_states,):
            raise ValueError(f"start_probs must have shape ({n_states},)")

        hmm_df = self.copy(deep = True)

        def bin_to_list(data, t_column, mov_var, bin):
            """ 
            Bins the time to the given integer and creates a nested list of the movement column by id
            """
            stat = 'max'
            data = data.reset_index()
            t_delta = data[t_column].iloc[1] - data[t_column].iloc[0]
            if t_delta != bin:
                data[t_column] = data[t_column].map(lambda t: bin * floor(t / bin))
                bin_gb = data.groupby(['id', t_column]).agg(**{
                    mov_var : (var_column, stat)
                })
                bin_gb.reset_index(level = 1, inplace = True)
                gb = bin_gb.groupby('id')[mov_var].apply(np.array)
            else:
                gb = data.groupby('id')[mov_var].apply(np.array)
            return gb

        if var_column == 'beam_crosses':
            hmm_df['active'] = np.where(hmm_df[var_column] == 0, 0, 1)
            gb = bin_to_list(hmm_df, t_column = t_column, mov_var = var_column, bin = bin_time)

        elif var_column == 'moving':
            hmm_df[var_column] = np.where(hmm_df[var_column] == True, 1, 0)
            gb = bin_to_list(hmm_df, t_column = t_column, mov_var = var_column, bin = bin_time)

        else:
            gb = bin_to_list(hmm_df, t_column = t_column, mov_var = var_column, bin = bin_time)

        # split runs into test and train lists
        test_train_split = round(len(gb) * (test_size/100))
        rand_runs = np.random.permutation(gb)
        train = rand_runs[test_train_split:]
        test = rand_runs[:test_train_split]

        len_seq_train = [len(ar) for ar in train]
        len_seq_test = [len(ar) for ar in test]

        # Augmenting shape to be accepted by hmmlearn
        seq_train = np.concatenate(train, 0)
        seq_train = seq_train.reshape(-1, 1)
        seq_test = np.concatenate(test, 0)
        seq_test = seq_test.reshape(-1, 1)

        for i in range(iterations):
            print(f"Iteration {i+1} of {iterations}")
            
            init_params = ''
            # h = hmm.MultinomialHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', verbose = verbose)
            h = hmm.CategoricalHMM(n_components = n_states, n_iter = hmm_iterations, tol = tol, params = 'ste', verbose = verbose)

            if start_probs is None:
                init_params += 's'
            else:
                s_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in start_probs], dtype = np.float64)
                s_prob = np.array([[y / sum(x) for y in x] for x in s_prob], dtype = np.float64)
                h.startprob_ = s_prob

            if trans_probs is None:
                init_params += 't'
            else:
                # replace 'rand' with a new random number being 0-1
                t_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in trans_probs], dtype = np.float64)
                t_prob = np.array([[y / sum(x) for y in x] for x in t_prob], dtype = np.float64)
                h.transmat_ = t_prob

            if emiss_probs is None:
                init_params += 'e'
            else:
                # replace 'rand' with a new random number being 0-1
                em_prob = np.array([[np.random.random() if y == 'rand' else y for y in x] for x in emiss_probs], dtype = np.float64)
                em_prob = np.array([[y / sum(x) for y in x] for x in em_prob], dtype = np.float64)
                h.emissionprob_ = em_prob

            h.init_params = init_params
            h.n_features = n_obs # number of emission states

            # call the fit function on the dataset input
            h.fit(seq_train, len_seq_train)

            # Boolean output of if the number of runs convererged on set of appropriate probabilites for s, t, an e
            print("True Convergence:" + str(h.monitor_.history[-1] - h.monitor_.history[-2] < h.monitor_.tol))
            print("Final log liklihood score:" + str(h.score(seq_train, len_seq_train)))

            # On first iteration save and print the first trained matrix
            if i == 0:
                try:
                    h_old = pickle.load(open(file_name, "rb"))
                    if h.score(seq_test, len_seq_test) > h_old.score(seq_test, len_seq_test):
                        print('New Matrix:')
                        df_t = pd.DataFrame(h.transmat_, index = states, columns = states)
                        print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
                        
                        with open(file_name, "wb") as file: pickle.dump(h, file)
                except OSError as e:
                    with open(file_name, "wb") as file: pickle.dump(h, file)

            # After see score on the test portion and only save if better
            else:
                h_old = pickle.load(open(file_name, "rb"))
                if h.score(seq_test, len_seq_test) > h_old.score(seq_test, len_seq_test):
                    print('New Matrix:')
                    df_t = pd.DataFrame(h.transmat_, index = states, columns = states)
                    print(tabulate(df_t, headers = 'keys', tablefmt = "github") + "\n")
                    with open(file_name, "wb") as file: pickle.dump(h, file)

            # Final iteration, load the best model and print its values
            if i+1 == iterations:
                h = pickle.load(open(file_name, "rb"))
                #print tables of trained emission probabilties, not accessible as objects for the user
                self.hmm_display(hmm = h, states = states, observables = observables)
                return h

    def get_hmm_raw(self, hmm: Any, variable: str = 'moving', t_bin: int = 60, 
                    func: str = 'max', t_column: str = 't') -> "behavpy_core":
        """
        Decode time series data using a trained Hidden Markov Model.

        Takes a trained HMM and uses it to decode the most likely sequence of hidden states
        from behavioral time series data. The data is first binned into fixed time intervals
        before decoding.

        Args:
            hmm (Any): Trained HMM object (typically hmmlearn.hmm.CategoricalHMM) containing:
                - startprob_: Starting state probabilities
                - transmat_: State transition probabilities 
                - emissionprob_: Emission probabilities            
            variable (str, optional): Column containing behavioral state data to decode.
                Defaults to 'moving'.
            t_bin (int, optional): Size of time bins in seconds for aggregating data.
                Defaults to 60.
            func (str, optional): Aggregation function to apply when binning data.
                Defaults to 'max'.
            t_column (str, optional): Column containing timestamps in seconds.
                Defaults to 't'.

        Returns:
            behavpy_core: New behavpy object with columns:
                - bin: Time bin start time
                - state: Decoded hidden state
                - previous_state: Previous decoded state (NaN for first point)
                - {variable}: Original behavioral state

        Example:
            # Decode movement data using trained HMM with 5-minute bins
            df = df.get_hmm_raw(trained_hmm, t_bin=60)
        """
        tdf = self.copy(deep=True)
        return self.__class__(
            self._hmm_decode(
                tdf, hmm, t_bin, variable, func, t_column, 
                return_type='table'
            ), 
            tdf.meta, 
            palette=self.attrs['sh_pal'], 
            long_palette=self.attrs['lg_pal'], 
            check=True
        ).drop(columns=[f'previous_{variable}'])

    # PERIODOGRAM SECTION

    def anticipation_score(self, variable: str, day_length: int = 24, lights_off: int = 12, 
                           t_column: str = 't') -> "behavpy_core":
        """
        Calculate anticipation scores to measure circadian rhythm strength.

        Computes an anticipation score by comparing activity levels in the 3 hours 
        immediately before a transition (lights on/off) to activity in the 6 hours before.
        The score is calculated as: (activity in last 3hrs / activity in last 6hrs) * 100.
        Higher scores indicate stronger anticipatory behavior.

        Args:
            variable (str): Column name containing activity measurements to analyze
            day_length (int, optional): Length of experimental day in hours. Defaults to 24.
            lights_off (int, optional): Hour when lights turn off, measured from lights on (0).
                Must be between 0 and day_length. Defaults to 12.
            t_column (str, optional): Column containing timestamps in seconds.
                Defaults to 't'.

        Returns:
            behavpy_core: New behavpy object with addtional columns:
                - anticipation_score: Percentage of 6-hour activity in final 3 hours
                - phase: Either 'Lights On' or 'Lights Off'

        Raises:
            KeyError: If variable or t_column not found in data
            ValueError: If lights_off not between 0 and day_length

        Example:
            # Calculate anticipation scores for movement data
            df = df.anticipation_score('moving')

            # Custom day length and lights-off time
            df = df.anticipation_score('moving', day_length=12, lights_off=6)

        Notes:
            - Data is first wrapped to day_length hours using wrap_time()
            - NaN values are dropped before calculation
            - Scores are calculated separately for lights-on and lights-off transitions
            - Used internally by plot_anticipation_score()
        """

        # Validate lights_off is between 0 and day_length
        if not 0 <= lights_off <= day_length:
            raise ValueError(f"lights_off ({lights_off}) must be between 0 and day_length ({day_length})")

        # Validate we have enough hours before transitions for calculation
        if lights_off < 6:
            raise ValueError("lights_off must be at least 6 hours after lights on for anticipation calculation")
        if day_length - lights_off < 6:
            raise ValueError("lights_off must be at least 6 hours before end of day for anticipation calculation")

        def _ap_score(total, small):
            try:
                return (small / total) * 100
            except ZeroDivisionError:
                return 0

        # drop NaN's and wrap everything to 24 hours
        data = self.dropna(subset=[variable]).copy(deep=True)
        data = data.wrap_time()

        filter_dict = {'Lights Off' : [lights_off - 6, lights_off - 3, lights_off],
                            'Lights On' : [day_length - 6, day_length - 3, day_length]}

        ant_df = pd.DataFrame()

        for phase in [ 'Lights On', 'Lights Off']:

            d = data.t_filter(start_time = filter_dict[phase][0], end_time = filter_dict[phase][2], t_column=t_column)
            total = d.analyse_column(column = variable, function = 'sum')
            
            d = data.t_filter(start_time = filter_dict[phase][1], end_time = filter_dict[phase][2], t_column=t_column)
            small = d.analyse_column(column = variable, function = 'sum')
            d = total.join(small, rsuffix = '_small')
            d = d.dropna()
            d = pd.DataFrame(d[[f'{variable}_sum', f'{variable}_sum_small']].apply(lambda x: _ap_score(*x), axis = 1), columns = ['anticipation_score']).reset_index()
            d['phase'] = phase
            ant_df = pd.concat([ant_df, d])

        return self.__class__(ant_df, self.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    def _validate(self) -> None:
        """ Validator to check further periodogram methods if the data is produced from the periodogram method """
        if any([i not in self.columns.tolist() for i in ['period', 'power']]):
            raise AttributeError('This method is for the computed periodogram data only, please run the periodogram method on your data first')
        

    def _check_periodogram_input(self, v: str, per: str, per_range: Union[List[int], np.ndarray], 
                                 t_col: str, wavelet_type: bool = False) -> Callable:
        """
        Internal method to validate inputs for periodogram analysis and return the appropriate function.

        Validates column names exist in data, periodogram type is supported, and period range is valid.
        Returns the corresponding periodogram function for analysis.

        Args:
            v (str): Name of column containing values to analyze
            per (str): Name of periodogram function to use
            per_range (Union[List[int], np.ndarray]): Two-element list/array containing 
                [min_period, max_period] in hours
            t_col (str): Name of column containing timestamps
            wavelet_type (bool, optional): If True, return wavelet function without additional
                validation. Defaults to False.

        Returns:
            Callable: Selected periodogram function for analysis

        Raises:
            KeyError: If v or t_col not found in data columns
            AttributeError: If per is not a supported periodogram type
            TypeError: If per_range is not a list/array or doesn't have exactly 2 elements
            ValueError: If per_range contains negative values

        Notes:
            Supported periodogram types are:
            - chi_squared
            - lomb_scargle  
            - fourier
        """

        periodogram_list = ['chi_squared', 'lomb_scargle', 'fourier']#, 'welch'] ## remove welch for now

        if v not in self.columns.tolist():
            raise KeyError(f"Variable column {v} is not a column title in your given dataset")

        if t_col not in self.columns.tolist():
            raise KeyError(f"Time column {t_col} is not a column title in your given dataset")

        if wavelet_type is not False:
            fun = eval(per)
            return fun

        if per in periodogram_list:
            fun = eval(per)
        else:
            raise AttributeError(f"Unknown periodogram type, please use one of {*periodogram_list,}")

        if isinstance(per_range, list) is False and isinstance(per_range, np.array) is False:
            raise TypeError(f"per_range should be a list or nummpy array, please change")

        if isinstance(per_range, list) or isinstance(per_range, np.array):

            if len(per_range) != 2:
                raise TypeError("The period range can only be a tuple/array of length 2, please amend")

            if per_range[0] < 0 or per_range[1] < 0:
                raise ValueError(f"One or both of the values of the period_range given are negative, please amend")

        return fun

    def periodogram(self, mov_variable: str, periodogram: str, period_range: List[int] = [10, 32], 
                    sampling_rate: int = 15, alpha: float = 0.01, t_column: str = 't', **kwargs) -> "behavpy_core":
        """ 
        Apply a periodogram analysis to behavioral data, typically movement data.

        This method creates an analysed dataset that can be plotted with other periodogram methods.
        Supported periodogram types include 'chi_squared', 'lomb_scargle', 'fourier', and 'welch'.

        Args:
            mov_variable (str): The name of the column in the data containing the movement data.
            periodogram (str): The name of the function to analyze the dataset with.
                Choose one of ['chi_squared', 'lomb_scargle', 'fourier', 'welch'].
            period_range (List[int], optional): A list containing the minimum and maximum values to find the 
                frequency power (in hours). Default is [10, 32].
            sampling_rate (int, optional): The frequency to resample the data at (in seconds). Default is 15.
            alpha (float, optional): The significance level. Default is 0.01.
            **kwargs: Additional keyword arguments for the periodogram function.

        Returns:
            behavpy_core: An amended behavpy dataframe with new columns 'period' and 'power'.

        Raises:
            AttributeError: If an unknown periodogram function is given. If you want to add your own implementation, e
            dit the base code or raise an issue on GitHub.
        """

        # Validate period_range is between 0 and max, and validate alpha
        if not 0 <= period_range[0] <= period_range[1]:
            raise ValueError(f"period_range ({period_range}) must be greater than 0 and period_range[0] < period_range[1]")
        if not alpha > 0:
            raise ValueError(f"alpha ({alpha}) must be greater than 0")

        fun = self._check_periodogram_input(mov_variable, periodogram, period_range, t_column)
        sampling_rate = 1 / (sampling_rate * 60)  # Converts to frequency
        step_size = int(1 / sampling_rate) # get the new t-diff to interpolate to

        data = self.copy(deep = True)
        # populate sample data 
        sampled_data = data.interpolate_linear(variable = mov_variable, step_size = step_size)
        sampled_data = sampled_data.reset_index()

        return self.__class__(sampled_data.groupby('id', group_keys = False)[[t_column, mov_variable]].apply(
            partial(fun, 
                    var = mov_variable, 
                    t_col = t_column, 
                    period_range = period_range, 
                    freq = sampling_rate,
                    alpha = alpha, 
                    **kwargs)), 
            data.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check = True)

    @staticmethod
    def wavelet_types() -> List[str]:
        """
        Retrieve a list of supported wavelet types for wavelet analysis.
        See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for more information.

        Returns:
            List[str]: A list of strings representing the names of available wavelet types.
        """
        wave_types = ['morl', 'cmor', 'mexh', 'shan', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 
                      'gaus6', 'gaus7', 'gaus8', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8']
        return wave_types

    def _format_wavelet(self, mov_variable: str, sampling_rate: int = 15, wavelet_type: str = 'morl', 
                        t_col: str = 't') -> Tuple[Callable, pd.DataFrame]:
        """
        Prepare data for wavelet analysis by resampling and averaging across specimens.

        This internal method checks the validity of the specified wavelet type, resamples the data 
        at the given sampling rate, and computes the average of the specified movement variable 
        across all specimens. It is recommended to filter the dataset before applying this method 
        to focus on specific experimental groups or individual specimens.

        Args:
            mov_variable (str): Name of the column containing movement data to analyze.
            sampling_rate (int, optional): Frequency to resample the data at (in seconds). 
                Defaults to 15.
            wavelet_type (str, optional): Type of wavelet to use for analysis. Defaults to 'morl'.
            t_col (str, optional): Name of the column containing timestamps. Defaults to 't'.

        Returns:
            Tuple[Callable, pd.DataFrame]: A tuple containing:
                - Callable: The wavelet function corresponding to the specified wavelet type.
                - pd.DataFrame: DataFrame with averaged movement data across specimens.

        Notes:
            - For more information on wavelet types, refer to the PyWavelets documentation:
                - https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html 
        """
        # check input and return the wavelet function with given wave type
        fun = self._check_periodogram_input(v = mov_variable, per = 'wavelet', per_range = None, t_col = t_col, wavelet_type = wavelet_type)
        sampling_rate = 1 / (sampling_rate * 60)
        step_size = int(1 /  sampling_rate)
        # re-sample the data at the given rate and interpolate 
        data = self.copy(deep = True)
        sampled_data = data.interpolate_linear(variable = mov_variable, step_size = step_size)
        # average across the dataset
        avg_data = sampled_data.groupby(t_col).agg(**{
                        mov_variable : (mov_variable, 'mean')
        })
        avg_data = avg_data.reset_index()

        return fun, avg_data

    @staticmethod
    def _wrapped_find_peaks(data: pd.DataFrame, num: int, height: Optional[bool] = False) -> pd.DataFrame:
        """
        Identify peaks in a computed periodogram.

        This internal method uses the `find_peaks` function from `scipy.signal` to detect peaks 
        in the power spectrum of the provided data. It ranks the detected peaks based on their 
        power values and assigns a rank to each peak. The method also allows for the use of a 
        significance threshold for peak detection.

        Args:
            data (pd.DataFrame): DataFrame containing the periodogram data with columns:
                - 'power': The power values corresponding to each period.
                - 'period': The periods associated with the power values.
                - 'sig_threshold' (optional): A threshold for peak significance, used if height is True.
            num (int): The maximum rank number for peaks to be considered significant. 
                Peaks with a rank greater than this value will be marked as False.
            height (bool, optional): If True, uses the 'sig_threshold' column to filter peaks 
                based on their height. If None, no height filtering is applied. Defaults to None.

        Returns:
            pd.DataFrame: The input DataFrame with an additional column 'peak' indicating the rank of 
            each period. If a period is not a peak or exceeds the specified rank limit, it will be marked as False.

        Notes:
            - Ensure that the input DataFrame contains the required columns before calling this method.
        """

        if height is True:
            peak_ind, _ = find_peaks(x = data['power'].to_numpy(), height = data['sig_threshold'].to_numpy())
        else:
            peak_ind, _ = find_peaks(x = data['power'].to_numpy())

        peaks = data['period'].to_numpy()[peak_ind]
        peak_power = data['power'].to_numpy()[peak_ind]
        order = peak_power.argsort()[::-1]
        ranks = order.argsort() + 1

        # Create dictionary with default value of False
        rank_dict = {k: int(v) for k,v in zip(peaks, ranks)}
        
        # Use map with a lambda that returns False for values not in dictionary
        data['peak'] = data['period'].map(lambda x: rank_dict.get(x, False))
        data['peak'] = np.where(data['peak'] > num, False, data['peak'])

        return data
    
    def find_peaks(self, num_peaks: int) -> "behavpy_core":
        """
        Identify significant peaks in a computed periodogram.

        This method uses the `find_peaks` function from `scipy.signal` to detect peaks in the power spectrum
        of the provided data. It ranks the detected peaks based on their power values and assigns a rank to each peak.
        The method can filter peaks based on a significance threshold if the 'sig_threshold' column is present.

        Args:
            num_peaks (int): The maximum rank number for peaks to be considered significant. 
                             Peaks with a rank greater than this value will be marked as False.

        Returns:
            behavpy_core: A new behavpy object containing the original data with an additional column 'peak' 
                          indicating the rank of each period. If a period is not a peak or exceeds the specified 
                          rank limit, it will be marked as False.

        Raises:
            AttributeError: If the data does not contain the required columns for peak detection.
            ValueError: If `num_peaks` is not a positive integer.

        Example:
            # Find the top 3 peaks in the periodogram
            peaks_df = df.find_peaks(num_peaks=3)
        """
        # Validate num_peaks
        if not isinstance(num_peaks, int) or num_peaks <= 0:
            raise ValueError("num_peaks must be a positive integer")    
            
        self._validate()
        data = self.copy(deep=True)
        data = data.reset_index()

        if 'sig_threshold' in data.columns.tolist():
            return self.__class__(data.groupby('id', group_keys=False).apply(
                partial(self._wrapped_find_peaks, num=num_peaks, height=True)), 
                data.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check=True)
        else:
            return self.__class__(data.groupby('id', group_keys=False).apply(
                partial(self._wrapped_find_peaks, num=num_peaks)), 
                data.meta, palette=self.attrs['sh_pal'], long_palette=self.attrs['lg_pal'], check=True)
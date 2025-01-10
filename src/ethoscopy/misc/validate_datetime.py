from datetime import datetime
import pandas as pd

def validate_datetime(data: pd.DataFrame) -> pd.DataFrame:
    """ 
    Validate and standardize date formats in DataFrame.
    
    Converts various date formats to YYYY-MM-DD standard format.

    Args:
        data (pd.DataFrame): DataFrame containing a 'date' column

    Returns:
        pd.DataFrame: DataFrame with standardized dates

    Raises:
        ValueError: If date format cannot be converted to YYYY-MM-DD
    """
    date_list = data['date'].values.tolist()
    new_date_list = []
    for i, date in enumerate(date_list):
            try:
                date == datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError:
                try:
                    if date == datetime.strptime(date, '%d-%m-%Y').strftime('%d-%m-%Y'):
                        date = datetime.strptime(date, '%d-%m-%Y').strftime('%Y-%m-%d')
                        new_date_list.append(date)
                except ValueError:
                    try:
                        if date == datetime.strptime(date, '%d/%m/%Y').strftime('%d/%m/%Y'):
                            date = datetime.strptime(date, '%d/%m/%Y').strftime('%Y-%m-%d')
                            new_date_list.append(date)
                    except ValueError:
                        raise ValueError("Incorrect data format, should be YYYY-MM-DD for row: " + str(i+1))

    if len(new_date_list) == 0:
        return data
    else:
        data['date'] = new_date_list   
        return data

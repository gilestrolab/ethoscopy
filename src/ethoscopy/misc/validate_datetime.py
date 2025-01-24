from datetime import datetime
import pandas as pd

def validate_datetime(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardize date formats in DataFrame.
    
    Converts various date formats to YYYY-MM-DD standard format.
    Supported input formats: YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY

    Args:
        data (pd.DataFrame): DataFrame containing a 'date' column

    Returns:
        pd.DataFrame: DataFrame with standardized dates

    Raises:
        ValueError: If date format cannot be converted to YYYY-MM-DD
    """
    # Define supported date formats
    date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%d/%m']
    
    def convert_date(date_str, row_idx):
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        raise ValueError(f"Incorrect date format in row {row_idx + 1}. "
                        f"Supported formats: YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY")

    # Create a copy to avoid modifying the original DataFrame
    result = data.copy()
    
    # Convert dates using vectorized operations when possible
    try:
        result['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    except ValueError:
        # Fallback to row-by-row processing if pandas can't automatically parse
        result['date'] = [convert_date(date, idx) 
                         for idx, date in enumerate(data['date'])]
    
    return result

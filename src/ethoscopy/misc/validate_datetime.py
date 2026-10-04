import re
from datetime import datetime

import pandas as pd

# Day/month/year strings: 05/01/2024, 5-1-2024, 05.01.24 (same separator twice).
_DAY_MONTH_YEAR = re.compile(r"^\d{1,2}([/.-])\d{1,2}\1(\d{2}|\d{4})$")


def validate_datetime(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardize date formats in DataFrame.

    Converts the 'date' column to the YYYY-MM-DD standard format.
    Day/month strings always follow the lab convention dd/mm/yyyy, whatever
    the other rows contain: '05/01/2024' is 5 January, and '01/25/2024' is an
    error rather than a month-first date. Separators may be '/', '-' or '.',
    and two-digit years are accepted. Other values (YYYY-MM-DD strings,
    datetime cells) are parsed by pandas.

    Args:
        data (pd.DataFrame): DataFrame containing a 'date' column

    Returns:
        pd.DataFrame: DataFrame with standardized dates

    Raises:
        ValueError: If a date cannot be converted to YYYY-MM-DD
    """

    def convert_date(value, row_idx):
        if not isinstance(value, str) and pd.isna(value):
            return value
        error = ValueError(
            f"Incorrect date format in row {row_idx + 1}: {value!r}. "
            f"Supported formats: YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY"
        )
        if isinstance(value, str):
            text = value.strip()
            match = _DAY_MONTH_YEAR.match(text)
            if match:
                sep = match.group(1)
                year = "%Y" if len(match.group(2)) == 4 else "%y"
                try:
                    parsed = datetime.strptime(text, f"%d{sep}%m{sep}{year}")
                except ValueError:
                    raise error from None
                return parsed.strftime("%Y-%m-%d")
            value = text
        try:
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            raise error from None

    # Create a copy to avoid modifying the original DataFrame
    result = data.copy()
    # Row by row: parsing the whole column with pandas would infer month-first
    # from the first value whenever every day in the column is <= 12.
    result["date"] = [convert_date(date, idx) for idx, date in enumerate(data["date"])]
    return result

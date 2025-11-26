# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 24-11-2025

"""
This module loads raw data and cleans it. The imported data should be cleaned and
ready for data preprocessing steps such as imputing, scaling, and encoding.
NaN values are present and preserved for sklearn imputation later.
"""


from .preprocess_data import basic_cleaning
import pandas as pd


# ------------------------------------------------------------
# Main cleaning function used by your pipeline or scripts
# ------------------------------------------------------------
def load_and_clean_raw(path: str) -> pd.DataFrame:
    """
    Convenience function:
    Load raw CSV → run basic_cleaning → return cleaned df.

    Parameters
    ----------
    path : str
        Path to the raw CSV (e.g., data/raw/immo_raw.csv)

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    df = basic_cleaning(df)
    
    return df
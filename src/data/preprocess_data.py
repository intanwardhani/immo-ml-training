# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 25-11-2025

"""
Basic data cleaning function to be applied to raw dataset. This is NOT the same as
full preprocessing (imputing, scaling, encoding) done later. This module focuses on:

- Removing unused columns
- Cleaning column names
- Renaming specific columns
- Removing rows with missing or obviously incorrect prices
- Removing duplicates
- Categorising build year into decades/centuries
"""


from typing import List, Optional
import logging
import pandas as pd
import numpy as np

# Set up module-level logger
logger = logging.getLogger(__name__)

UNUSED_COLUMNS: List[str] = [
    "property_url",
    "property_id"
]
UNREALISTIC_PRICE_THRESHOLD = 1000


def basic_cleaning(df: pd.DataFrame,
                   extra_drop: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Perform basic cleaning on the raw dataset.

    Logs information about rows removed and errors encountered.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for train-test split.
    """

    original_rows = len(df)
    logger.info(f"Starting basic_cleaning() with {original_rows} rows")

    df = df.copy()

    # ----- Drop unnecessary columns -----
    drop_cols = UNUSED_COLUMNS.copy()
    if extra_drop:
        drop_cols.extend(extra_drop)

    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info(f"Dropped unused columns: {drop_cols}")

    # ----- Clean column names (consistent snake_case) -----
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # ----- Rename specific columns -----
    df = df.rename(columns={"state": "building_state",
                            "number_rooms": "number_bedrooms"})

    # ----- Track price-related cleaning ----- 
    if "price" in df.columns:
        before = len(df)

        # Remove NaN price rows
        df = df[df["price"].notna()]
        logger.info(f"Removed {before - len(df)} rows with missing price")

        before = len(df)
        # Remove obviously incorrect placeholder prices (<= 1)
        df = df[df["price"] > 1]
        logger.info(f"Removed {before - len(df)} rows with price <= 1")

        before = len(df)
        # Remove unrealistically low prices
        df = df[df["price"] >= UNREALISTIC_PRICE_THRESHOLD]
        logger.info(f"Removed {before - len(df)} rows with price < {UNREALISTIC_PRICE_THRESHOLD}")

    # ----- Remove duplicates -----
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate rows")

    # ----- Basic data type fixes -----
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()
        
    # ----- Remove invalid bedroom counts -----
    # Allowed types with 0 bedrooms
    allowed_zero_bedrooms = {"studio", "student", "loft", "commercial", "office",
                             "garage", "business", "land"}

    if "number_bedrooms" in df.columns and "property_type" in df.columns:
        # Normalize property_type for matching
        df["property_type"] = df["property_type"].astype(str).str.lower().str.strip()

        # Mask for rows where 0 bedrooms is acceptable
        valid_zero = df["property_type"].isin(allowed_zero_bedrooms)

        # Remove rows where bedrooms == 0 but NOT allowed
        invalid_zero_rows = (df["number_bedrooms"] == 0) & (~valid_zero)

        # Report how many rows are removed
        removed_count = invalid_zero_rows.sum()
        if removed_count > 0:
            logger.info(f"Removing {removed_count} rows with invalid 0 bedrooms.")

        df = df[~invalid_zero_rows]


    # ----- Categorising build year -----
    def categorise_build_year(year):
        if pd.isna(year):
            return np.nan

        try:
            year = int(year)
        except Exception as e:
            # Log specific errors for debugging
            logger.warning(f"Could not parse build_year '{year}': {e}")
            return np.nan

        if 1500 <= year < 1900:
            century = (year // 100) * 100
            return f"{century}s"
        elif year >= 1900:
            decade = (year // 10) * 10
            return f"{decade}s"
        else:
            return np.nan
        
    if "build_year" in df.columns:
        df["build_year_cat"] = df["build_year"].apply(categorise_build_year)
        logger.info("Added build_year_cat column")

    logger.info(
        f"Finished basic_cleaning(): {len(df)} rows remain "
        f"(removed {original_rows - len(df)} total rows)"
    )

    df = df[sorted(df.columns)]

    return df

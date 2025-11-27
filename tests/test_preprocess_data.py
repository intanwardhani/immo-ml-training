# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 25-11-2025

# run on terminal:
# cd immo-ML-project pytest -v tests/test_preprocess_data.py


import pandas as pd
import numpy as np
from src.data.preprocess_data import basic_cleaning


def test_basic_cleaning_price_removal():
    df = pd.DataFrame({
        "price": [200000, None, 1, 300000],
        "state": ["New", "Excellent", "To renovate", "Normal", "To restore",
                  "Fully renovated", "To demolish", "Under construction", None],
        "build_year": [1990, 2000, 1850, None, 1700],
        "property_url": ["https://immovlan.be/en/detail/residence/for-sale/1410/waterloo/vbd29473",
                         "https://immovlan.be/en/detail/residence/for-sale/2200/herentals/rbu66815",
                         "https://immovlan.be/en/detail/residence/for-sale/3080/tervuren/rbu50909",
                         "https://immovlan.be/en/detail/residence/for-sale/4400/flemalle/vbd44979",
                         "https://immovlan.be/en/detail/villa/for-sale/3080/tervuren/rbu60085"]
    })

    cleaned = basic_cleaning(df)

    # Only valid price rows >= 1000 should remain
    assert cleaned["price"].min() >= 1000
    assert len(cleaned) == 2  # rows: 200000, 300000


def test_valid_bedrooms():
    # Create a small sample dataset
    data = {
        "property_type": ["House", "Studio", "Apartment", "Office", "Villa"],
        "number_bedrooms": [0, 0, 2, 0, 0],
        "price": [200000, 150000, 300000, 100000, 500000],
        "build_year": [1990, 2010, 1980, 2000, 1965],
    }

    df = pd.DataFrame(data)

    cleaned = basic_cleaning(df)

    # ---- Expected behavior ----
    # House with 0 bedrooms -> removed
    # Studio with 0 bedrooms -> kept
    # Apartment with 2 bedrooms -> kept
    # Office with 0 bedrooms -> kept
    # Villa with 0 bedrooms -> removed

    expected_property_types = ["studio", "apartment", "office"]  # lowercased in cleaning

    assert list(cleaned["property_type"]) == expected_property_types

    # number of rows left after cleaning: 3
    assert cleaned.shape[0] == 3

    # Ensure no invalid 0-bedroom properties remain
    assert not (
        (cleaned["number_bedrooms"] == 0)
        & (~cleaned["property_type"].isin(["studio", "student", "commercial", "office"]))
    ).any()

def test_build_year_categorisation():
    df = pd.DataFrame({
        "price": [200000, 300000],
        "state": ["New", "Excellent"],
        "build_year": [1850, 1999]
    })

    cleaned = basic_cleaning(df)

    assert "build_year_cat" in cleaned.columns
    assert cleaned.loc[cleaned["build_year"] == 1850, "build_year_cat"].iloc[0] == "1800s"
    assert cleaned.loc[cleaned["build_year"] == 1999, "build_year_cat"].iloc[0] == "1990s"


def test_invalid_build_year_handling():
    df = pd.DataFrame({
        "price": [200000],
        "state": ["good"],
        "build_year": ["unknown"]
    })

    cleaned = basic_cleaning(df)

    assert pd.isna(cleaned["build_year_cat"].iloc[0])


def test_unused_columns_removed():
    df = pd.DataFrame({
        "price": [200000],
        "state": ["good"],
        "property_url": ["abc"]
    })

    cleaned = basic_cleaning(df)

    assert "property_url" not in cleaned.columns

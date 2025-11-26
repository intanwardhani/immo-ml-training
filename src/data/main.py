# UTF-8 Python 3.13.5
# Author: Intan K. Wardhani
# Last modified: 25-11-2025


"""
Script to load raw CSV, clean data, split into train/test, and save processed CSVs.
"""


from .load_data import load_and_clean_raw
from sklearn.model_selection import train_test_split
import os
import logging


# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,  # log level
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_processing.log"),  # save logs to file
        logging.StreamHandler()  # also print to console
    ]
)

logger = logging.getLogger(__name__)

# ----- Path handling -----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "immo_raw.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")


# ------------------------------------------------------------
# If run as script
# (Manually clean + save processed CSVs)
# ------------------------------------------------------------
if __name__ == "__main__":

    logger.info("Loading raw data...")
    df = load_and_clean_raw(RAW_PATH)

    logger.info("Splitting into train/test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    logger.info("Saving cleaned datasets...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    logger.info("Done! Cleaned data saved to:")
    logger.info(f"  - {TRAIN_PATH}")
    logger.info(f"  - {TEST_PATH}")
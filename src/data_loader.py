# this file handles loading the loan dataset from csv and validating it
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# these are the columns we expect in the dataset - if any are missing we raise an error
REQUIRED_COLUMNS = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "Loan_Approved",
]


def load_data(filepath: str) -> pd.DataFrame:
    # load the csv dataset and check that it's not empty and has all required columns
    logger.info("Loading data from: %s", filepath)
    try:
        df = pd.read_csv(filepath)  # read the csv into a dataframe
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Data file not found: {filepath}") from e
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        raise
    # make sure the dataframe isn't empty
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
    # check that all required columns are present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    logger.info("Data loaded successfully. Shape: %s", df.shape)
    return df

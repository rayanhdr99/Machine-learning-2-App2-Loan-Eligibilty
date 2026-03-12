"""
data_loader.py - Loads and validates the loan eligibility (credit) dataset.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "Loan_Approved",
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit/loan dataset from a CSV file."""
    logger.info("Loading data from: %s", filepath)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Data file not found: {filepath}") from e
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        raise
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    logger.info("Data loaded successfully. Shape: %s", df.shape)
    return df

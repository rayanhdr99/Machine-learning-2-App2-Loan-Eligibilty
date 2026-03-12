"""
preprocessor.py - Preprocessing pipeline for the Loan Eligibility dataset.
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
DUMMY_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: mode for categoricals, median for LoanAmount."""
    logger.info("Imputing missing values.")
    df = df.copy()
    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    null_count = df.isnull().sum().sum()
    logger.info("Missing values after imputation: %d", null_count)
    return df


def encode_and_prepare(df: pd.DataFrame):
    """Drop Loan_ID, encode target, one-hot encode categoricals.

    Returns:
        Processed DataFrame ready for splitting.
    """
    logger.info("Encoding and preparing features.")
    df = df.copy()
    df = df.drop("Loan_ID", axis=1)
    df["Credit_History"] = df["Credit_History"].astype("object")
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")
    df["Loan_Approved"] = df["Loan_Approved"].map({"Y": 1, "N": 0}).astype(int)
    df = pd.get_dummies(df, columns=DUMMY_COLS, dtype=int)
    logger.info("Encoded. Columns: %s", list(df.columns))
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2):
    """Train/test split and MinMax scaling.

    Returns:
        Tuple (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns).
    """
    logger.info("Splitting and scaling.")
    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]
    feature_columns = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

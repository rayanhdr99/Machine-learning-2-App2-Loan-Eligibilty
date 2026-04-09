# this file handles all the preprocessing: imputing missing values, encoding, splitting, and scaling
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# lists of categorical columns we need to handle
CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
DUMMY_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # fill missing values - mode for categorical columns, median for LoanAmount
    logger.info("Imputing missing values.")
    df = df.copy()  # don't modify the original dataframe
    # fill categorical columns with their most frequent value (mode)
    for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    # fill loan term and credit history with mode too
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    # fill loan amount with median since it's numerical
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    # log how many missing values are left (should be 0)
    null_count = df.isnull().sum().sum()
    logger.info("Missing values after imputation: %d", null_count)
    return df


def encode_and_prepare(df: pd.DataFrame):
    # drop Loan_ID, encode the target column, and one-hot encode categorical features
    logger.info("Encoding and preparing features.")
    df = df.copy()
    df = df.drop("Loan_ID", axis=1)  # loan ID isn't useful for prediction
    # convert these to object type so pandas treats them as categorical for get_dummies
    df["Credit_History"] = df["Credit_History"].astype("object")
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")
    # map Y/N to 1/0 for the target variable
    df["Loan_Approved"] = df["Loan_Approved"].map({"Y": 1, "N": 0}).astype(int)
    # one-hot encode the categorical columns so the model can use them
    df = pd.get_dummies(df, columns=DUMMY_COLS, dtype=int)
    logger.info("Encoded. Columns: %s", list(df.columns))
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2):
    # split 80/20 into training and testing sets, then scale features to 0-1 range
    logger.info("Splitting and scaling.")
    # separate features (X) from the target (y)
    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]
    feature_columns = list(X.columns)  # save column names for later use in prediction
    # stratify makes sure the train/test split has the same ratio of approved/denied
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    # scale features to 0-1 range using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on training data only
    X_test_scaled = scaler.transform(X_test)  # transform test data using same scaler
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

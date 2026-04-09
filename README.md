# Loan Eligibility Predictor

A production-quality machine learning web application that predicts whether a loan application will be **approved** or **denied**, built with Python, scikit-learn, and Streamlit.

---

## Project Description

This project solves a binary classification problem: given a set of applicant features, predict whether a loan should be approved (`Y`) or denied (`N`). The dataset is the **German Credit / Loan Eligibility dataset** containing **614 records** with 13 columns covering personal, financial, and property information.

Three supervised learning classifiers are trained, evaluated, and exposed through an interactive Streamlit web interface. Users can explore the raw dataset, compare model performance metrics, and enter new applicant details to receive a real-time prediction with confidence scores.

---

## Project Structure

```
loan_eligibility_predictor/
├── app.py                  # Streamlit web application (main entry point)
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation (this file)
├── data/
│   └── credit.csv          # Loan eligibility dataset (614 rows, 13 columns)
└── src/
    ├── __init__.py         # Marks src/ as a Python package
    ├── data_loader.py      # load_data() — reads and validates the CSV file
    ├── preprocessor.py     # impute_missing(), encode_and_prepare(), split_and_scale()
    ├── model.py            # train_logistic_regression(), train_decision_tree(), train_random_forest()
    └── evaluator.py        # evaluate_model(), cross_validate()
```

---

## Dataset

| Property       | Detail                                      |
|----------------|---------------------------------------------|
| File           | `data/credit.csv`                           |
| Rows           | 614                                         |
| Target column  | `Loan_Approved` (Y = approved, N = denied)  |

### Columns

| Column              | Type    | Description                                      |
|---------------------|---------|--------------------------------------------------|
| `Loan_ID`           | object  | Unique identifier — dropped before training      |
| `Gender`            | object  | Male / Female                                    |
| `Married`           | object  | Yes / No                                         |
| `Dependents`        | object  | 0 / 1 / 2 / 3+                                   |
| `Education`         | object  | Graduate / Not Graduate                          |
| `Self_Employed`     | object  | Yes / No                                         |
| `ApplicantIncome`   | int     | Monthly income of the primary applicant          |
| `CoapplicantIncome` | float   | Monthly income of the co-applicant               |
| `LoanAmount`        | float   | Requested loan amount (in thousands)             |
| `Loan_Amount_Term`  | float   | Loan repayment term in months                    |
| `Credit_History`    | float   | 1.0 = history met, 0.0 = not met                 |
| `Property_Area`     | object  | Urban / Semiurban / Rural                        |
| `Loan_Approved`     | object  | **Target** — Y (approved) or N (denied)          |

---

## Setup and Running Instructions

### 1. Navigate to the project directory

```bash
cd "machine learning project/loan_eligibility_predictor"
```

### 2. Ensure the dataset is in place

```
loan_eligibility_predictor/
    data/
        credit.csv      <-- required
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit application

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Preprocessing Pipeline

```
credit.csv
    └─► load_data()              validates CSV and required columns
        └─► impute_missing()     mode imputation for categoricals,
                                 median imputation for LoanAmount
            └─► encode_and_prepare()
                    drop Loan_ID
                    Credit_History / Loan_Amount_Term → object dtype
                    pd.get_dummies (dtype=int) on 6 categorical columns
                    Loan_Approved: Y → 1, N → 0
                └─► split_and_scale()
                        train_test_split(test_size=0.2, stratify=y, random_state=42)
                        MinMaxScaler.fit_transform (train only)
                        MinMaxScaler.transform (test)
```

---

## Models

Three classifiers are trained on the preprocessed dataset:

| Model                    | Key Parameters                                      |
|--------------------------|-----------------------------------------------------|
| **Logistic Regression**  | `max_iter=1000`, `random_state=42`                  |
| **Decision Tree**        | `random_state=42`                                   |
| **Random Forest**        | `n_estimators=100`, `max_depth=5`, `random_state=42`|

---

## Evaluation Metrics

Each model is assessed using three metrics:

| Metric                 | Description                                                      |
|------------------------|------------------------------------------------------------------|
| **Accuracy Score**     | Fraction of correct predictions on the held-out test set         |
| **Confusion Matrix**   | 2x2 matrix showing true/false positives and negatives            |
| **K-Fold Cross-Validation** | 5-fold CV mean and standard deviation of accuracy on training set |

---

## Typical Results

| Model                | Test Accuracy | CV Mean Accuracy | Notes              |
|----------------------|---------------|------------------|--------------------|
| Logistic Regression  | ~83%          | ~82%             | Best overall model |
| Decision Tree        | ~76%          | ~77%             | Tends to overfit   |
| Random Forest        | ~80%          | ~81%             | Robust ensemble    |

The best model achieves approximately **83% accuracy**, with Logistic Regression consistently performing best on this dataset due to the relatively small number of features and the near-linear decision boundary.

---

## Application Pages

The Streamlit app has three navigation pages accessible from the sidebar:

1. **Dataset Overview** — Dataset shape, approval rate, sample rows, bar chart, missing value report.
2. **Model Performance** — Side-by-side accuracy metrics, confusion matrix heatmaps, accuracy comparison bar chart.
3. **Predict Eligibility** — Form-based input for a new applicant; returns Approved/Denied verdict with confidence percentages from the selected model.

---

## Dependencies

| Package        | Version   | Purpose                              |
|----------------|-----------|--------------------------------------|
| `streamlit`    | >=1.32.0  | Web application framework            |
| `pandas`       | >=2.0.0   | Data loading and manipulation        |
| `numpy`        | >=1.24.0  | Numerical operations                 |
| `scikit-learn` | >=1.3.0   | ML models, preprocessing, evaluation |
| `matplotlib`   | >=3.7.0   | Plotting                             |
| `seaborn`      | >=0.12.0  | Confusion matrix heatmaps            |

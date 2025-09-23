"""Data preprocessing script."""

import argparse

import pandas as pd


def preprocess(loan_applications: pd.DataFrame) -> pd.DataFrame:
    clean = loan_applications.copy()

    # Remove rows where loan interest rates are not provided
    clean = clean.dropna(subset=["loan_int_rate"])

    # Drop employment lengths greater than 60 years
    THRESHOLD_EMP_LENGTH = 60
    clean = clean[clean["person_emp_length"] <= THRESHOLD_EMP_LENGTH]

    # Cap income at 1 million
    CAP_INCOME = 1_000_000
    clean["person_income"] = clean["person_income"].clip(upper=CAP_INCOME)

    # Standardize education labels
    education_standardizer = {
        "BELOW_SECONDARY": "Secondary",
        "GRADUATE": "Graduate",
        "POSTGRAD": "Post Graduate",
    }
    clean["person_education"] = (
        clean["person_education"].map(education_standardizer).fillna("Other")
    )

    # Turn CB default on file check into an integer
    clean["cb_person_default_on_file"] = clean["cb_person_default_on_file"].map(
        {"Y": 1, "N": 0}
    )

    return clean


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path")
    parser.add_argument("--preprocessed-data-path")

    return parser.parse_args()


def main():
    args = _get_arguments()

    loan_data = pd.read_csv(args.raw_data_path)
    loan_data = preprocess(loan_data)
    loan_data.to_csv(args.preprocessed_data_path, index=False)

"""Data preprocessing script."""

import argparse

import pandas as pd


def preprocess(df: pd.DataFrame):
    clean = df.copy()
    # Remove rows where loan interest rates are not provided
    clean = clean.dropna(subset=["loan_int_rate"])
    # Cap employment lengths at 60 years
    clean = clean[clean["person_emp_length"] <= 60]
    # Cap income at 1 million
    clean["person_income"] = clean["person_income"].clip(upper=1_000_000)
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

"""Data preprocessing script."""


import pandas as pd
import argparse

from sklearn.compose import ColumnTransformer
import numpy as np
def preprocess(df):
    clean = df.copy()
    # Remove rows where loan interest rates are not provided
    bAdRows=[]
    for i,row in clean.iterrows():
        if row["loan_int_rate"]==None or row["loan_int_rate"]==np.nan:
            bAdRows.append(i)
    clean=clean.drop(bAdRows)
    # Cap employment lengths at 60 years
    outlier_filter = clean["person_emp_length"] > 60
    indices= clean[outlier_filter].index
    clean = clean.drop(index=indices)
    # Cap income at 1 million
    cap = 3_000_000 # former cap
    for i,row in clean.iterrows():
        if row["person_income"] > 1_000_000:
            clean.at[i, "person_income"]=1_000_000
    # Standardize education labels
    for i, row in clean.iterrows():
        if row["person_education"] in ["BELOW_SECONDARY", "Secondary"]:
            clean.at[i, "person_education"] = "Secondary"
        elif row["person_education"] in ["GRADUATE", "Graduate"]:
            clean.at[i, "person_education"] = "Graduate"
        elif row["person_education"] in ["POSTGRAD", "Post Graduate"]:
            clean.at[i, "person_education"] = "Post Graduate"
        else:
            clean.at[i, "person_education"] = "Other"
    # Turn CB default on file check into an integer
    for i,row in clean.iterrows():
        if row["cb_person_default_on_file"] == "Y":
            clean.at[i, "cb_person_default_on_file"] = 1
        elif row["cb_person_default_on_file"] == "N":
            clean.at[i, "cb_person_default_on_file"] = 0

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
import os

import click
import pandas as pd

from credit_default_prediction.data_preprocessing import preprocess
from credit_default_prediction.training import train


class TrainingPipeline:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def train(self, loan_applications: pd.DataFrame):
        clean = preprocess(loan_applications)
        X = clean.drop("loan_status")
        y = clean["loan_status"]
        loan_classifier = self.algorithm(X, y)

        return loan_classifier


@click.command()
@click.option("--dataset-path", help="Path to the training dataset.")
def cli(dataset_path: os.PathLike):
    loan_applications = pd.read_csv(dataset_path)
    training_pipeline = TrainingPipeline(algorithm=train)
    training_pipeline.train(loan_applications)

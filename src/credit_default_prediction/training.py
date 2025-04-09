"""Model training script."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import joblib
import pandas as pd


class IClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()


@dataclass
class HyperParams:
    learning_rate: float
    max_depth: 4
    min_child_weight: 1


def train(X: pd.Series, y: pd.Series, loan_default_classifier: IClassifier):
    loan_default_classifier.fit(X, y)

    return loan_default_classifier


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)

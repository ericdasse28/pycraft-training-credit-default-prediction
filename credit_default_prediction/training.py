import pandas as pd
import xgboost as xgb


def train(X: pd.DataFrame, y: pd.Series | pd.DataFrame, **hyperparameters):
    loan_default_classifier = xgb.XGBClassifier(**hyperparameters)
    loan_default_classifier.fit(X, y)
    return loan_default_classifier

import pandas as pd
from sklearn.base import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "ROC_AUC": roc_auc_score(y, model.predict_proba(X)[:, 1]),
    }

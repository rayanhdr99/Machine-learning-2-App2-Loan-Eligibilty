"""
evaluator.py - Evaluates classification models.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Compute accuracy, confusion matrix, and classification report."""
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", model_name, e)
        raise
    logger.info("%s accuracy: %.4f", model_name, acc)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report, "predictions": y_pred}


def cross_validate(model, X_train, y_train, n_splits: int = 5) -> dict:
    """Run K-Fold cross-validation on training data."""
    logger.info("Running %d-fold cross-validation.", n_splits)
    kfold = KFold(n_splits=n_splits)
    scores = cross_val_score(model, X_train, y_train, cv=kfold)
    logger.info("CV mean: %.4f  std: %.4f", scores.mean(), scores.std())
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}

# this file evaluates how well our models perform using accuracy, confusion matrix, and cross-validation
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    # get predictions on test data and calculate accuracy, confusion matrix, and classification report
    try:
        y_pred = model.predict(X_test)  # predict on test set
        acc = accuracy_score(y_test, y_pred)  # compare predictions to actual labels
        cm = confusion_matrix(y_test, y_pred)  # shows true/false positives and negatives
        report = classification_report(y_test, y_pred, output_dict=True)  # precision, recall, f1-score
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", model_name, e)
        raise
    logger.info("%s accuracy: %.4f", model_name, acc)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report, "predictions": y_pred}


def cross_validate(model, X_train, y_train, n_splits: int = 5) -> dict:
    # run k-fold cross-validation to check how stable the model's accuracy is
    logger.info("Running %d-fold cross-validation.", n_splits)
    kfold = KFold(n_splits=n_splits)  # split data into k folds
    scores = cross_val_score(model, X_train, y_train, cv=kfold)  # get accuracy for each fold
    logger.info("CV mean: %.4f  std: %.4f", scores.mean(), scores.std())
    # return the scores along with mean and standard deviation
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}

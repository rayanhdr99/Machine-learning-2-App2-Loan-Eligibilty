# this file trains three different classification models: logistic regression, decision tree, and random forest
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    # train a logistic regression model on the training data
    logger.info("Training Logistic Regression.")
    try:
        model = LogisticRegression(max_iter=1000, random_state=42)  # max_iter=1000 to make sure it converges
        model.fit(X_train, y_train)  # fit the model on training data
    except Exception as e:
        logger.error("Logistic Regression training failed: %s", e)
        raise
    logger.info("Logistic Regression training complete.")
    return model


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    # train a decision tree classifier on the training data
    logger.info("Training Decision Tree.")
    try:
        model = DecisionTreeClassifier(random_state=42)  # default hyperparameters
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error("Decision Tree training failed: %s", e)
        raise
    logger.info("Decision Tree training complete.")
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    # train a random forest with 100 trees and max depth of 5
    logger.info("Training Random Forest.")
    try:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)  # tuned hyperparameters
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error("Random Forest training failed: %s", e)
        raise
    logger.info("Random Forest training complete.")
    return model

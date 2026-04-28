"""Random Forest modeling and validation."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest(random_state: int = 42) -> Pipeline:
    """Build a reusable Random Forest classification pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and evaluate the Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_random_forest(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def shuffled_label_validation(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train the same model on shuffled labels as a sanity check."""
    y_shuffled = y.sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_reset = X.reset_index(drop=True)

    return train_and_evaluate(
        X_reset,
        y_shuffled,
        test_size=test_size,
        random_state=random_state,
    )

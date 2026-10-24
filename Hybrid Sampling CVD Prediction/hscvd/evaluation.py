"""Metrics (Eqs. 1-5) and k-fold cross-validation (Tables 5 and 7)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import N_FOLDS, RANDOM_STATE, VALIDATION_SIZE
from .data import split_xy
from .sampling import resample


def compute_metrics(y_true, y_pred, y_score):
    """Accuracy, precision, recall, F1 and AUC, in percent."""
    return {
        "Accuracy (%)": 100 * accuracy_score(y_true, y_pred),
        "Precision (%)": 100 * precision_score(y_true, y_pred, zero_division=0),
        "Recall (%)": 100 * recall_score(y_true, y_pred, zero_division=0),
        "F1 Score (%)": 100 * f1_score(y_true, y_pred, zero_division=0),
        "AUC (%)": 100 * roc_auc_score(y_true, y_score),
    }


def tune_threshold(model, X, y):
    """Set model.threshold to the cut-off that maximises Youden's J
    (sensitivity + specificity - 1) on (X, y), usually the validation set."""
    fpr, tpr, thresholds = roc_curve(y, model.predict_scores(X))
    best = np.argmax(tpr - fpr)
    model.threshold = float(min(thresholds[best], 1.0))
    return model.threshold


def evaluate(model, df):
    """Evaluate a fitted model on a DataFrame; returns (metrics, y_true, y_score, cm)."""
    X, y = split_xy(df)
    scores = model.predict_scores(X)
    preds = model.predict(X)
    metrics = compute_metrics(y, preds, scores)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    return metrics, y.to_numpy(), scores, cm


def cross_validate(model_factory, pool, n_folds=N_FOLDS, resample_method=None,
                   tune=False, random_state=RANDOM_STATE):
    """Stratified k-fold CV; a fresh model is trained on every fold.

    model_factory(y_train) returns an unfitted model. If resample_method is
    given, only the training portion of each fold is resampled (used by the
    strict protocol, where the pool is real data). If tune is True, 20% of
    each training portion is set aside as a validation set for early
    stopping and for tuning the decision threshold.
    """
    X, y = split_xy(pool)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True,
                            random_state=random_state)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(folds.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val = y_val = None
        if tune:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr, y_tr, test_size=VALIDATION_SIZE, stratify=y_tr,
                random_state=random_state)
        if resample_method:
            X_tr, y_tr = resample(X_tr, y_tr, resample_method, random_state)
        model = model_factory(y_tr).fit(X_tr, y_tr, X_val, y_val)
        if tune:
            tune_threshold(model, X_val, y_val)
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        metrics = compute_metrics(y_te, model.predict(X_te),
                                  model.predict_scores(X_te))
        rows.append({"Fold": f"Fold {fold}", **metrics})
        print(f"    fold {fold}: " + ", ".join(
            f"{k.split(' ')[0]}={v:.2f}" for k, v in metrics.items()))
    df = pd.DataFrame(rows)
    mean = df.drop(columns="Fold").mean(numeric_only=True)
    std = df.drop(columns="Fold").std(numeric_only=True)
    df = pd.concat([df, pd.DataFrame([{"Fold": "Mean", **mean},
                                      {"Fold": "Std", **std}])],
                   ignore_index=True)
    return df


def format_table(df):
    return df.to_string(index=False, float_format=lambda v: f"{v:.2f}")

"""ML (Section 3.5) and DL (Section 4) models.

Every model exposes the same interface:
    fit(X_train, y_train, X_val=None, y_val=None)
    predict_scores(X) -> continuous score for the positive class (used for ROC/AUC)
    predict(X)        -> 0/1 labels

Setting model.threshold (see evaluation.tune_threshold) makes predict() use
predict_scores(X) >= threshold instead of the default decision rule.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import (AUTOENCODER_DIM, AUTOENCODER_EPOCHS, BATCH_SIZE,
                     DL_EPOCHS, RANDOM_STATE)


# ---------------------------------------------------------------- ML models

class SklearnModel:
    def __init__(self, estimator):
        self.estimator = estimator
        self.history = None
        self.threshold = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        kwargs = {}
        early_stopping = getattr(self.estimator, "early_stopping_rounds", None)
        if early_stopping and X_val is not None:
            kwargs = {"eval_set": [(np.asarray(X_val), np.asarray(y_val))],
                      "verbose": False}
        self.estimator.fit(np.asarray(X_train), np.asarray(y_train), **kwargs)
        return self

    def predict_scores(self, X):
        X = np.asarray(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)[:, 1]
        return self.estimator.decision_function(X)

    def predict(self, X):
        if self.threshold is not None:
            return (self.predict_scores(X) >= self.threshold).astype(int)
        return self.estimator.predict(np.asarray(X)).astype(int)


def build_ml_model(name, random_state=RANDOM_STATE):
    if name == "DT":
        # Fig. 9 labels the decision tree C4.5, i.e. entropy-based splits.
        return SklearnModel(DecisionTreeClassifier(
            criterion="entropy", random_state=random_state))
    if name == "SVM":
        # Linear-kernel SVM (Section 3.5.2). LinearSVC solves the same problem
        # as SVC(kernel="linear") but scales to hundreds of thousands of rows.
        return SklearnModel(make_pipeline(
            MinMaxScaler(), LinearSVC(dual=False, random_state=random_state)))
    if name == "XGBoost":
        return SklearnModel(XGBClassifier(
            eval_metric="logloss", random_state=random_state, n_jobs=-1))
    raise ValueError(f"Unknown ML model: {name}")


# ---------------------------------------------------------------- DL models

def _keras():
    # Imported lazily so the ML-only path does not pay TensorFlow start-up.
    from tensorflow import keras
    return keras


def build_autoencoder(n_inputs, encoding_dim=AUTOENCODER_DIM):
    """Single-layer autoencoder used for feature extraction; returns (autoencoder, encoder)."""
    keras = _keras()
    inputs = keras.Input(shape=(n_inputs,))
    encoded = keras.layers.Dense(encoding_dim, activation="relu")(inputs)
    decoded = keras.layers.Dense(n_inputs, activation="sigmoid")(encoded)
    autoencoder = keras.Model(inputs, decoded, name="autoencoder")
    encoder = keras.Model(inputs, encoded, name="encoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def build_classifier(name, n_inputs):
    keras = _keras()
    layers = keras.layers
    if name == "CNN":
        model = keras.Sequential([
            keras.Input(shape=(n_inputs, 1)),
            layers.Conv1D(filters=8, kernel_size=2, activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ], name="CNN")
    elif name == "NN":
        model = keras.Sequential([
            keras.Input(shape=(n_inputs,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ], name="NN")
    elif name == "MLP":
        model = keras.Sequential([
            keras.Input(shape=(n_inputs,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ], name="MLP")
    else:
        raise ValueError(f"Unknown DL model: {name}")
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


class DLModel:
    """MinMax scaling -> autoencoder feature extraction -> CNN / NN / MLP."""

    def __init__(self, name, epochs=None, batch_size=BATCH_SIZE,
                 ae_epochs=AUTOENCODER_EPOCHS, use_autoencoder=True,
                 balanced=False, early_stopping=False, verbose=0):
        self.name = name
        self.epochs = epochs or DL_EPOCHS[name]
        self.batch_size = batch_size
        self.ae_epochs = ae_epochs
        self.use_autoencoder = use_autoencoder
        # balanced: weight classes inversely to their frequency instead of
        # relying on resampled data. early_stopping: keep the epoch with the
        # lowest validation loss.
        self.balanced = balanced
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.history = None
        self.threshold = None

    def _features(self, X):
        X = self.scaler.transform(np.asarray(X, dtype="float32"))
        if self.use_autoencoder:
            X = self.encoder.predict(X, batch_size=4096, verbose=0)
        if self.name == "CNN":
            X = np.expand_dims(X, axis=2)
        return X

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(np.asarray(X_train, dtype="float32"))
        if self.use_autoencoder:
            autoencoder, self.encoder = build_autoencoder(X_scaled.shape[1])
            autoencoder.fit(X_scaled, X_scaled, epochs=self.ae_epochs,
                            batch_size=self.batch_size, validation_split=0.2,
                            verbose=self.verbose)

        X_tr = self._features(X_train)
        n_inputs = X_tr.shape[1]
        self.model = build_classifier(self.name, n_inputs)
        validation = None
        if X_val is not None:
            validation = (self._features(X_val), np.asarray(y_val))
        y_train = np.asarray(y_train)
        class_weight = None
        if self.balanced:
            counts = np.bincount(y_train, minlength=2)
            class_weight = {c: len(y_train) / (2 * counts[c]) for c in (0, 1)}
        callbacks = []
        if self.early_stopping and validation is not None:
            callbacks.append(_keras().callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True))
        self.history = self.model.fit(
            X_tr, y_train, epochs=self.epochs,
            batch_size=self.batch_size, validation_data=validation,
            class_weight=class_weight, callbacks=callbacks,
            verbose=self.verbose).history
        return self

    def predict_scores(self, X):
        return self.model.predict(self._features(X), batch_size=4096,
                                  verbose=0).ravel()

    def predict(self, X):
        threshold = 0.5 if self.threshold is None else self.threshold
        return (self.predict_scores(X) >= threshold).astype(int)


def build_model(name, **dl_kwargs):
    if name in ("CNN", "NN", "MLP"):
        return DLModel(name, **dl_kwargs)
    return build_ml_model(name)


# ----------------------------------------------------------- improved models

def build_improved_model(name, y_train, **dl_kwargs):
    """Models for the improved pipeline (run_improved.py).

    They are trained on the real, imbalanced training data and handle the
    imbalance with class weights; the decision threshold is tuned afterwards
    on the validation set. Tree models are regularised so they cannot
    memorise individual records.
    """
    y_train = np.asarray(y_train)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    if name == "DT":
        return SklearnModel(DecisionTreeClassifier(
            criterion="entropy", max_depth=8, min_samples_leaf=50,
            class_weight="balanced", random_state=RANDOM_STATE))
    if name == "SVM":
        return SklearnModel(make_pipeline(
            MinMaxScaler(), LinearSVC(dual=False, class_weight="balanced",
                                      random_state=RANDOM_STATE)))
    if name == "LogReg":
        return SklearnModel(make_pipeline(
            StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced")))
    if name == "XGBoost":
        return SklearnModel(XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=5,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="auc",
            early_stopping_rounds=100, random_state=RANDOM_STATE, n_jobs=-1))
    if name in ("CNN", "NN", "MLP"):
        return DLModel(name, balanced=True, early_stopping=True, **dl_kwargs)
    raise ValueError(f"Unknown model: {name}")

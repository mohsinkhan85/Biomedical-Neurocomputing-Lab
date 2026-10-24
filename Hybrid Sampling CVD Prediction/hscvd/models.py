"""ML (Section 3.5) and DL (Section 4) models.

Every model exposes the same interface:
    fit(X_train, y_train, X_val=None, y_val=None)
    predict_scores(X) -> continuous score for the positive class (used for ROC/AUC)
    predict(X)        -> 0/1 labels
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
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

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.estimator.fit(np.asarray(X_train), np.asarray(y_train))
        return self

    def predict_scores(self, X):
        X = np.asarray(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)[:, 1]
        return self.estimator.decision_function(X)

    def predict(self, X):
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
                 ae_epochs=AUTOENCODER_EPOCHS, use_autoencoder=True, verbose=0):
        self.name = name
        self.epochs = epochs or DL_EPOCHS[name]
        self.batch_size = batch_size
        self.ae_epochs = ae_epochs
        self.use_autoencoder = use_autoencoder
        self.verbose = verbose
        self.history = None

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
        self.history = self.model.fit(
            X_tr, np.asarray(y_train), epochs=self.epochs,
            batch_size=self.batch_size, validation_data=validation,
            verbose=self.verbose).history
        return self

    def predict_scores(self, X):
        return self.model.predict(self._features(X), batch_size=4096,
                                  verbose=0).ravel()

    def predict(self, X):
        return (self.predict_scores(X) > 0.5).astype(int)


def build_model(name, **dl_kwargs):
    if name in ("CNN", "NN", "MLP"):
        return DLModel(name, **dl_kwargs)
    return build_ml_model(name)

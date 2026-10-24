"""Data collection, preprocessing and feature selection (Sections 3.1-3.3)."""

import pandas as pd

from .config import SELECTED_FEATURES, TARGET


def load_dataset(path):
    """Load the BRFSS 2015 CSV, drop duplicate rows and cast to integers."""
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df.astype(int)


def describe(df):
    """Print the structure checks described in Section 3.2."""
    print(f"Dataset shape: {df.shape}")
    missing = int(df.isnull().sum().sum())
    print(f"Missing values: {missing}")
    counts = df[TARGET].value_counts().sort_index()
    print(f"Target distribution: no heart disease = {counts.get(0, 0)}, "
          f"heart disease = {counts.get(1, 0)}")


def correlation_ranking(df):
    """Correlation of every feature with the target, in descending order."""
    corr = df.corr()[TARGET].drop(TARGET)
    return corr.sort_values(ascending=False)


def select_features(df, features=SELECTED_FEATURES):
    """Keep the selected features plus the target column."""
    return df[features + [TARGET]].copy()


def split_xy(df):
    return df.drop(columns=TARGET), df[TARGET]

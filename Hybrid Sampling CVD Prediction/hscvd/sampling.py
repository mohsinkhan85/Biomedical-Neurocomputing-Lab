"""Hybrid sampling and dataset partitioning (Section 3.4)."""

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TARGET, TEST_PER_CLASS, VALIDATION_SIZE
from .data import split_xy


def resample(X, y, method, random_state=RANDOM_STATE):
    """Balance (X, y) with SMOTEENN, ROS or hybrid sampling (SMOTEENN then ROS).

    method="none" returns the data unchanged.
    """
    if method == "none":
        return X, y
    if method == "smoteenn":
        return SMOTEENN(random_state=random_state).fit_resample(X, y)
    if method == "ros":
        return RandomOverSampler(random_state=random_state).fit_resample(X, y)
    if method == "hs":
        X_se, y_se = SMOTEENN(random_state=random_state).fit_resample(X, y)
        return RandomOverSampler(random_state=random_state).fit_resample(X_se, y_se)
    raise ValueError(f"Unknown sampling method: {method}")


def resample_df(df, method, random_state=RANDOM_STATE):
    X, y = split_xy(df)
    X_res, y_res = resample(X, y, method, random_state)
    return pd.concat([X_res, y_res], axis=1).reset_index(drop=True)


def hold_out_per_class(df, n_per_class=TEST_PER_CLASS, random_state=RANDOM_STATE):
    """Draw n_per_class rows of each class as a test set; return (test, remaining)."""
    smallest = df[TARGET].value_counts().min()
    if smallest < n_per_class:
        raise ValueError(
            f"Cannot hold out {n_per_class} rows per class: the smallest class has "
            f"{smallest} rows. Lower --test-per-class.")
    picked = [
        df[df[TARGET] == label].sample(n=n_per_class, random_state=random_state)
        for label in (0, 1)
    ]
    test = pd.concat(picked)
    remaining = df.drop(test.index)
    return test.reset_index(drop=True), remaining.reset_index(drop=True)


def build_splits(df, method, protocol="paper", n_per_class=TEST_PER_CLASS,
                 random_state=RANDOM_STATE):
    """Create the train / validation / test sets for one sampling method.

    protocol="paper" follows the paper: the whole dataset is resampled first,
    then n_per_class rows per class are held out as the test set and the rest
    is split 80/20 into training and validation. Synthetic and duplicated
    rows can therefore appear in more than one split.

    protocol="strict" holds out the test set (n_per_class real rows per
    class) and the 20% validation set before sampling, and resamples only the
    training set, so validation and test contain only real, unseen records.

    Returns a dict with "train", "val", "test", "pool" and "resampled".
    "pool" is the data used for cross-validation (resampled in the paper
    protocol; real in the strict protocol, where each fold is resampled
    separately). "resampled" is the balanced data shown in Figs. 4 and 5.
    """
    if protocol == "paper":
        resampled = resample_df(df, method, random_state)
        test, pool = hold_out_per_class(resampled, n_per_class, random_state)
        train, val = train_test_split(
            pool, test_size=VALIDATION_SIZE, random_state=random_state)
    elif protocol == "strict":
        test, pool = hold_out_per_class(df, n_per_class, random_state)
        train_real, val = train_test_split(
            pool, test_size=VALIDATION_SIZE, random_state=random_state,
            stratify=pool[TARGET])
        train = resample_df(train_real, method, random_state)
        resampled = train
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test,
        "pool": pool,
        "resampled": resampled,
    }

"""Data-only steps shared by the public analysis scripts and regression tests."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def common_dates(etf_dates, customer_dates):
    """Return shared dates in chronological order for trailing-window selection."""
    return np.array(sorted(set(etf_dates) & set(customer_dates)))


def standardize_features(features):
    values = StandardScaler().fit_transform(features)
    return pd.DataFrame(values, columns=features.columns, index=features.index)


def fit_kmeans(features, n_clusters):
    return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(features)

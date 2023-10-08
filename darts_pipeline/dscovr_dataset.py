import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def create_dscovr_dataset(dataset, lookback, num_classes):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []

    size = len(dataset) - lookback - 1
    size = min(10**4, len(dataset) - lookback - 1)

    for i in tqdm(range(size)):
        feature = dataset[i:i + lookback, :-1]

        target = max(dataset[i + lookback - 1, -1:], dataset[i + lookback, -1:])
        if target[0] > 0:
            target = F.one_hot(torch.tensor(target).to(torch.int64), num_classes).tolist()
            X.append(feature)
            y.append(target)
    N = len(X)
    X, y = np.array(X).reshape(-1), np.array(y).reshape(-1)

    return torch.reshape(torch.tensor(X), (N, lookback, -1)), torch.reshape(torch.tensor(y), (-1, num_classes))

def mock_dscovr_dataset(dataset, lookback, num_classes):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []

    size = len(dataset) - lookback
    size = 10**4

    for i in tqdm(range(size)):
        feature = dataset[i:i + lookback, :-1]

        target = dataset[i + lookback, -1:]

        target = F.one_hot(torch.tensor([0]).to(torch.int64), num_classes).tolist()
        X.append(feature)
        y.append(target)
    N = len(X)
    X, y = np.array(X).reshape(-1), np.array(y).reshape(-1)

    return torch.reshape(torch.tensor(X), (N, lookback, -1)), torch.reshape(torch.tensor(y), (-1, num_classes))


def anomaly_indices(anomalies_list, df):
    df['row_number'] = df.reset_index().index
    indices = df.loc[anomalies_list]["row_number"].to_list()
    df.drop(columns='row_number', inplace=True)
    return indices

def anomalous_dscovr_dataset(anomalies_list, df_full, lookback, num_classes):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []

    anomaly_entries = df_full.loc[df_full.index.isin(anomalies_list)]
    for e in anomaly_entries:
        feature = df_full[i - lookback:i, :-1]
        target = df_full[i, -1:]
        target = F.one_hot(torch.tensor(target).to(torch.int64), num_classes).tolist()
        X.append(feature)
        y.append(target)

    N = len(X)
    X, y = np.array(X).reshape(-1), np.array(y).reshape(-1)

    return torch.reshape(torch.tensor(X), (N, lookback, -1)), torch.reshape(torch.tensor(y), (-1, num_classes))


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

    size = len(dataset) - lookback
    size = 10**4

    for i in tqdm(range(size)):
        feature = dataset[i:i + lookback, :-1]

        target = dataset[i + lookback, -1:]
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

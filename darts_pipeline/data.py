from darts import TimeSeries
from matplotlib import pyplot as plt
from torchmetrics import SymmetricMeanAbsolutePercentageError
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.models import KalmanFilter, KalmanForecaster
import pandas as pd
import os



def load_time_series(filename, cols=None):
    df = pd.read_csv(os.path.join(os.path.curdir, "data", filename), header=None)
    return TimeSeries.from_dataframe(df=df, value_cols=cols, time_col=0)


def train_valid_split(filenames):
    for filename in filenames:
        return zip(*(ts.split_after(0.8) for ts in load_time_series(filename)))


def model_train_prediction(model, train, valid, is_deterministic=False):
    if not is_deterministic:
        model.fit(series=train, val_series=valid, epochs=10, num_loader_workers=4)
    else:
        model.fit(series=train, val_series=valid)
    return model


def model_train_filtering(model, train, valid, is_deterministic=False):
    if not is_deterministic:
        model.fit(series=train, val_series=valid, epochs=10, num_loader_workers=4)
    else:
        model.fit(series=train, val_series=valid)
    return model


def data_preprocessing(time_series):
    transformer = MissingValuesFiller()
    return transformer.transform(time_series)


def model_anomaly_filtering():
    train = data_preprocessing(load_time_series("data/dsc_2016.csv")[:1 * 60 * 60])
    time_index = train.time_index
    target_series = TimeSeries.from_times_and_values(time_index, train.values()[:, 0])
    covariates = TimeSeries.from_times_and_values(time_index, train.values()[:, 1:3])

    print(target_series.values().shape)
    print(covariates.values().shape)

    kf = KalmanFilter(dim_x=1)
    kf.fit(target_series, covariates)
    filtered_series = kf.filter(target_series, covariates, num_samples=1000)

    plt.figure(figsize=[12, 8])
    target_series.plot(color="red", label="Noisy observations")
    filtered_series.plot(color="green", label="Filtered observations")
    plt.legend()
    plt.show()

model_anomaly_filtering()

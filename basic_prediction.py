import pandas as pd
import numpy as np
import sktime
from sktime.forecasting.arima import ARIMA
from sktime.utils import plotting

def process(df):
  df.columns = ["sensor_"+str(x) for x in range(len(df.columns))]
  df = df.rename({'sensor_0': 't'}, axis=1)
  df = df.set_index('t')
  return df

if __name__ == "__main__":

  window_size = 60

  df = pd.read_csv("dsc_fc_summed_spectra_2023_v01.csv", \
  delimiter = ',', parse_dates=[0], \
  infer_datetime_format=True, na_values='0', \
  header = None)

  df = process(df)
  df = df[df.columns[53:]]

  df_1h_MA = pd.DataFrame(index=pd.date_range(start=df.index[0], end=df.index[-1], freq="1h"))

  for x in df.columns:
    df[f"{x}_1h_MA"] = df[x].fillna(0).rolling(window=window_size).mean()

  y = df_1h_MA['sensor_1_1h_MA'].fillna(0)

  # Tutaj dajemy jaki zakres nas interesuje(w godzinach)
  fh = np.arange(1, 30)

  # step 3: specifying the forecasting algorithm
  forecaster = ARIMA()
  # step 4: fitting the forecaster
  forecaster.fit(y, fh=fh)
  # step 5: querying predictions
  y_pred = forecaster.predict()

  # for probabilistic forecasting:
  #   call a probabilistic forecasting method after or instead of step 5
  y_pred_int = forecaster.predict_interval(coverage=0.9)

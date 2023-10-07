import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import xgboost
# import sktime
# from sktime.forecasting.arima import ARIMA
# from sktime.utils import plotting

def process(df):
  df.columns = ["sensor_" + str(x) for x in range(len(df.columns))]
  df = df.rename({'sensor_0': 't'}, axis=1)
  df = df.set_index('t')
  # df = df[df.columns[53:]]
  return df

if __name__ == "__main__":
  kp_data = pd.read_csv("data/kp-data-new.csv")
  df = pd.read_csv("dsc_fc_summed_spectra_2023_v01.csv", \
                    delimiter = ',',
                    parse_dates=[0], \
                    infer_datetime_format=True,
                    na_values='0', \
                    header = None)

  df = process(df)

  # 1h MA
  df_1h_MA = pd.DataFrame(index=pd.date_range(start=df.index[0],
                                              end=df.index[-1],
                                              freq="1h"))

  for x in df.columns:
    df_1h_MA[x] = df[x].fillna(0).rolling(window=60).mean()


  cols = [f"{x}_1h {x}_2h {x}_3h " for x in df_1h_MA.columns]
  cols = " ".join(x for x in cols)
  cols = cols.split(" ")
  cols = list(filter(None, cols))

  print(f"cols {cols}")

  df_new = pd.DataFrame(columns=cols)


  print(f"df_1h_MA.columns: {df_1h_MA.columns}")


  df_new["sensor_1_1h"] = [el[0] for el in np.array(df_1h_MA['sensor_1'].fillna(0)).reshape(-1, 3)]
  df_new["sensor_1_2h"] = [el[1] for el in np.array(df_1h_MA['sensor_1'].fillna(0)).reshape(-1, 3)]
  df_new["sensor_1_3h"] = [el[2] for el in np.array(df_1h_MA['sensor_1'].fillna(0)).reshape(-1, 3)]

  df_new["sensor_2_1h"] = [el[0] for el in np.array(df_1h_MA['sensor_2'].fillna(0)).reshape(-1, 3)]
  df_new["sensor_2_2h"] = [el[1] for el in np.array(df_1h_MA['sensor_2'].fillna(0)).reshape(-1, 3)]
  df_new["sensor_2_3h"] = [el[2] for el in np.array(df_1h_MA['sensor_2'].fillna(0)).reshape(-1, 3)]

  df_new["sensor_3_1h"] = [el[0] for el in np.array(df_1h_MA['sensor_3'].fillna(0)).reshape(-1, 3)]
  df_new["sensor_3_2h"] = [el[1] for el in np.array(df_1h_MA['sensor_3'].fillna(0)).reshape(-1, 3)]
  df_new["sensor_3_3h"] = [el[2] for el in np.array(df_1h_MA['sensor_3'].fillna(0)).reshape(-1, 3)]

  # Bierzemy tylko pierwsze 3 sensory do predykcji na ten moment
  df_new = df_new[df_new.columns[:9]]

  # Remove last day from df
  # Why? Kp_data jest do 2023-05-01
  # Bedzie trzeba pewnie potem usunac
  df_new = df_new[:-8]

  df_new['y'] =  np.array(kp_data[kp_data.datetime > "2023-01-01"][['datetime', 'Kp']][kp_data.datetime <= str(df_1h_MA.index[-1]).split(" ")[0]]['Kp'].values).reshape(-1, 1)

  X = df_new[df_new.columns[:-1]]
  y = df_new['y']
  xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.2)

  reg = xgboost.XGBRegressor()

  reg.fit(
    xtrain, ytrain,
    eval_set=[(xtrain, ytrain), (xvalid, yvalid)],
    eval_metric='logloss',
    early_stopping_rounds=50,
    verbose=100
  )

  y_pred = reg.predict(xvalid)

  print(mse(yvalid, y_pred))


import pandas as pd
import numpy as np
import os
from datetime import timedelta

kp = pd.read_csv(
    "kp-data-new.csv",
)
kp["k_window"] = pd.to_datetime(kp["datetime"], utc=True)
kp["datetime"] = pd.to_datetime(kp.datetime)
kp.set_index("datetime", inplace=True)
first_rows = kp.iloc[[0]]
kp["Rounded value"] = kp[::-1].rolling(3).max()[::-1]["Rounded value"]
kp.iloc[[0]] = first_rows
kp["is_storm"] = (kp["Rounded value"] > 5).astype(bool)
kp["time_diff_prev_storm"] = (
    kp.index.to_series().diff().dt.total_seconds() * kp["is_storm"]
)
y = kp.time_diff_prev_storm
kp["storm_reading"] = (
    y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1) // (60 * 60 * 3)
)
kp.fillna(0, inplace=True)
storms = kp.loc[(kp["Rounded value"] > 5) & (kp.storm_reading <= 1)]
storms["storm_id"] = storms.index.astype(str)
kp["storm_id"] = storms.storm_id.reindex(kp.index, method="nearest") * kp["is_storm"]
kp.storm_id.replace("", np.nan, inplace=True)

for year_shift in range(8):
    year = year_shift + 2016
    print(year)
    name = f"dsc_fc_summed_spectra_{year}_v01"
    df = pd.read_csv(
        f"{name}.csv",
        delimiter=",",
        parse_dates=[0],
        header=None,
    )
    df["k_window"] = pd.to_datetime(
        df[0].apply(
            lambda x: x.replace(
                hour=x.hour if not x.hour % 3 else x.hour - x.hour % 3,
                minute=0,
                second=0,
            )
        ),
        utc=True,
    )
    df2 = df.merge(kp, how="left", on=["k_window"])[df.columns.tolist() + ["Kp"]]
    df2.rename(columns={"Kp": "k_index_current"}, inplace=True)
    df2.k_window = pd.to_datetime(
        df2.k_window.apply(lambda x: x + timedelta(hours=3)), utc=True
    )
    df3 = df2.merge(kp, how="left", on=["k_window"])[
        df2.columns.tolist() + ["Kp", "storm_id"]
    ]
    df3.drop("k_window", axis=1, inplace=True)
    df3.rename(columns={"Kp": "k_index_target"}, inplace=True)
    df3.to_csv(f"tmp_data/data_{year}.csv")

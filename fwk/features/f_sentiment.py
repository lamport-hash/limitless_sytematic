import pandas as pd
import numpy as np

from dataframe_utils.df_utils import (
    g_close_col,
    g_open_col,
    g_high_col,
    g_low_col,
    g_volume_col,
    g_mid_col,
    g_mid2_col,
    g_close_time_col,
    g_open_time_col,
    g_index_col,
    g_qa_vol_col,
    g_nt_col,
    g_la_vol_col,
    g_lqa_vol_col,
)
import features.features_utils as features_utils


# from https://api.alternative.me/fng/?limit=10&format=csv
def load_sentiment(data_folder, filename="sentiment/feargreedindex_alternativeme.csv", txt_csv=""):
    # Using StringIO to simulate a file‑like object.
    if len(txt_csv) > 0:
        df = pd.read_csv(StringIO(txt_csv))
    else:
        df = pd.read_csv(data_folder + filename)

    # ------------------------------------------------------------------
    # 2. Parse the date column as datetime (day-month-year).
    #    Set it as the DataFrame index for resampling.
    # ------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df.set_index("date", inplace=True)

    print("Original DataFrame:")
    print(df, "\n")

    # ------------------------------------------------------------------
    # 3. Resample to 1‑minute frequency.
    #    Because the original data is daily, we will forward‑fill
    #    each value across all minutes in that day.
    # ------------------------------------------------------------------
    resampled = (
        df.resample("1min").ffill()  # '1T' == 1 minute  # carry last known value forward
    )
    return resampled


from io import StringIO

csv_text = """date,fng_value,fng_classification
01-10-2025,49,Neutral
30-09-2025,50,Neutral
29-09-2025,50,Neutral
28-09-2025,37,Fear
27-09-2025,33,Fear
26-09-2025,28,Fear"""


def test_load_sentiment():
    load_sentiment("", "", csv_text)

"""
Unit tests for firstrate data normalization.

Tests the normalization of firstrate ETF data files to standard candle columns.
"""

import pandas as pd

from core.data_org import (
    get_import_file,
    get_normalised_file,
    create_normalised_dirs,
    ProductType,
)
from core.enums import (
    MktDataFred,
    ExchangeNAME,
    g_index_col,
    g_open_col,
    g_high_col,
    g_low_col,
    g_close_col,
    g_volume_col,
    g_open_time_col,
    g_close_time_col,
)
from norm.norm_firstdata import norm_firstdata


def test_normalize_firstrate_file():
    """
    Test normalization of a firstrate ETF 1h data file.

    This test:
    1. Loads a firstrate file from data/import/firstrate/etf
    2. Normalizes it to standard candle columns
    3. Displays the first 5 and last 5 rows of the normalized DataFrame
    4. Verifies the expected columns are present
    """
    symbol = "AAA"
    subtype = "etf_1h"

    input_file = get_import_file(ExchangeNAME.FIRSTRATE, symbol, subtype)
    output_file = get_normalised_file(
        MktDataFred.CANDLE_1HOUR,
        ExchangeNAME.FIRSTRATE,
        ProductType.ETF,
        symbol,
    )

    create_normalised_dirs(
        MktDataFred.CANDLE_1HOUR,
        ExchangeNAME.FIRSTRATE,
        ProductType.ETF,
        symbol,
    )

    print(f"\n{'='*60}")
    print(f"Testing normalization of: {input_file.name}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    df = norm_firstdata(str(input_file), str(output_file))

    expected_columns = [
        g_index_col,
        g_open_col,
        g_high_col,
        g_low_col,
        g_close_col,
        g_volume_col,
        g_open_time_col,
        g_close_time_col,
    ]

    print("Expected columns:", expected_columns)
    print("Actual columns:  ", list(df.columns))
    print(f"\nDataFrame shape: {df.shape}")
    print(f"DataFrame dtypes:\n{df.dtypes}")

    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

    print(f"\n{'='*60}")
    print("FIRST 2 NORMALIZED ROWS:")
    print(f"{'='*60}")
    print(df.head(2).to_string())

    print(f"\n{'='*60}")
    print("LAST 2 NORMALIZED ROWS:")
    print(f"{'='*60}")
    print(df.tail(2).to_string())

    print(f"\n{'='*60}")
    print("TEST PASSED!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_normalize_firstrate_file()

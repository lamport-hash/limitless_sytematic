import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple, Union, Any, Hashable
from pathlib import Path
from tqdm import tqdm
import psutil

import logging


logger_s = logging.getLogger(__name__)
logger_s.setLevel(logging.INFO)


from core.enums import (
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
    g_precision,
)


g_yaml_t_classification = "t_classification"
g_yaml_t_regression = "t_regression"


# because of parquet not knowing float_16
global_float_64_colnames_list = [
    "Volume",
    "RITA",
    "acc_dist_index",
    "Quote asset volume",
    "Taker buy quote asset volume",
    "Taker buy base asset volume",
]
global_float_32_colnames_list = []


def read_parquet_to_f16(
    p_filename,
    P_col_exception_list=[],
    p_float_precision: str = "16",
):
    loaded_df = pd.read_parquet(p_filename)
    return convert_df_cols_float_to_float(
        loaded_df,
        p_colnames=[],
        p_float_size=p_float_precision,
        p_exception_list=global_float_64_colnames_list
        + global_float_32_colnames_list
        + P_col_exception_list,
        p_test_overflow=True,
    )


def convert_df_cols_float_to_float(
    p_df: pd.DataFrame,
    p_colnames: List[str] = [],
    p_float_size: str = "32",
    p_exception_list: List[str] = [],
    p_test_overflow: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """
    Converts specified DataFrame columns to float16 data type.

    Args:
        p_df (pd.DataFrame): The input DataFrame.
        p_colnames (List[str]): Optional list of column names to convert. If empty, all float columns are converted.

    Returns:
        Tuple[pd.DataFrame, Dict[str, bool]]: A tuple containing the modified DataFrame and a dictionary indicating which columns overflowed during conversion.
    """

    types_to_convert = "float64"

    if p_float_size == g_precision:
        types_to_convert = "float"  # in case of 16 we convert all cols if they are not listed

    if not p_colnames:
        all_to_convert = p_df.select_dtypes(include=types_to_convert).columns.tolist()
        # here we filter out all the col names that are terminated by either _f32 _f64 _i _b _c _t
        col_list = [
            item
            for item in all_to_convert
            if not any(item.endswith(suffix) for suffix in ["_f32", "_f64", "_i", "_b", "_c", "_t"])
        ]
        p_colnames = [x for x in col_list if x not in p_exception_list]

    float_limits = {
        "16": 65504.0,
        "32": 3.4e38,
    }
    max_limit = float_limits.get(p_float_size, 65504.0)

    for col in p_colnames:
        if p_test_overflow:
            max_val = p_df[col].max()
            min_val = p_df[col].min()

            if max_val > max_limit or min_val < -max_limit:
                print(" that col :" + col + " will likely overflow ")
        p_df[col] = p_df[col].astype("float" + p_float_size)

    return p_df


def scale_dataframe_columns(
    p_df: pd.DataFrame, p_colnames: List[str], p_factor: float
) -> pd.DataFrame:
    """
    Multiplies the specified columns in a DataFrame by a given factor.

    Args:
        p_df (pd.DataFrame): The input DataFrame.
        p_colnames (List[str]): The list of column names to be scaled.
        p_factor (float): The scaling factor.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns scaled by the given factor.
    """
    df_scaled = p_df.copy()
    for col in p_colnames:
        if col in df_scaled.columns:
            df_scaled[col] *= p_factor
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    return df_scaled


def filter_df_by_column_match(
    p_df: pd.DataFrame, p_other_df: pd.DataFrame, p_colname: Hashable
) -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows where the values in a specified column match
    those found in the same column of another DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame to filter.
        p_other_df (pd.DataFrame): The other DataFrame containing the matching values.
        p_colname (Hashable): The column name to check for matches.

    Returns:
        pd.DataFrame: A filtered version of the original DataFrame with only matching rows.
    """
    # Ensure the specified column exists in both DataFrames
    if p_colname not in p_df.columns or p_colname not in p_other_df.columns:
        raise ValueError(f"Column '{p_colname}' does not exist in one or both DataFrames.")

    # Extract unique values from the other DataFrame's column
    matching_values = set(p_other_df[p_colname])

    # Filter the original DataFrame to include only rows with matching column values
    filtered_df = p_df[p_df[p_colname].isin(matching_values)]

    return filtered_df


def get_all_features_from_df(assets, df):
    feature_list = []
    for asset in assets:
        for col in df.columns:
            if col[:6] == asset + "_F_":
                feature_list.append(col)
    return feature_list


def gen_features_colname_for_coins(
    p_coin_list: List[str], p_list_cols: List[str]
) -> List[List[str]]:
    """
    Generate feature column names for multiple coins.

    Args:
        p_coin_list: List of coin identifiers
        p_list_cols: List of base column features

    Returns:
        List of lists containing combined feature columns for each coin
    """
    if not p_coin_list:
        raise ValueError("p_coin_list must contain at least one coin")

    if not p_list_cols:
        raise ValueError("p_list_cols must contain at least one column name")

    total_list_cols = []

    for coin in p_coin_list:
        list_cols = [f"{coin}_{col}" for col in p_list_cols]
        total_list_cols += list_cols

    return total_list_cols


def get_system_memory():
    """Get system memory information in GB."""
    mem = psutil.virtual_memory()
    sysmem = {"total": mem.total / 1024**3, "used": mem.used / 1024**3, "free": mem.free / 1024**3}
    print("System Memory:")
    for key, value in sysmem.items():
        print(f"{key}: {value:.2f} GB")

    return sysmem


def measure_parquet_memory(parquet_files):
    """
    Load and measure memory usage of multiple Parquet files.

    Args:
        parquet_files (list): List of paths to Parquet (.parquet) files

    Returns:
        list: Memory usage in MB for each Parquet file
    """
    total_size_in_memory_gb = 0

    try:
        # Load and measure each Parquet file
        for i, file_path in enumerate(tqdm(parquet_files)):
            df = pd.read_parquet(file_path)
            # loaded_dfs.append(df)
            size_in_memory_bytes = df.memory_usage(deep=True).sum()
            size_in_memory_gb = size_in_memory_bytes / (1024**3)
            print(f"Size in memory: {size_in_memory_gb:.6f} GB")
            total_size_in_memory_gb += size_in_memory_gb
            # Calculate memory usage
            del df

    except Exception as e:
        print(f"Error processing file: {file_path}. Error: {str(e)}")
        raise

    return total_size_in_memory_gb


def prefix_and_select_columns(
    df1: pd.DataFrame, p_name: str, p_cols: List[str], p_id_col: str
) -> pd.DataFrame:
    """
    Args:
        df1 (pd.DataFrame): The first dataframe to be cleaned and renamed.
        p_name (str): The prefix to add to the columns of the dataframe.
        p_cols (List[str]): List of column names in the dataframe that need to be prefixed.
        p_id_col (str): The column name to keep identical.

    Returns:
        pd.DataFrame: The cleaned dataframe
    """
    prefixed_cols = {col: f"{p_name}_{col}" for col in p_cols}
    list_cols = [f"{p_name}_{col}" for col in p_cols]
    df1_prefixed = df1.rename(columns=prefixed_cols)
    list_cols.append(p_id_col)
    df1_prefixed = df1_prefixed[list_cols]  # clean before merging

    return df1_prefixed


def merge_and_prefix_columns(
    df1: pd.DataFrame, df2: pd.DataFrame, p_name: str, p_cols: List[str], p_id_col: str
) -> pd.DataFrame:
    """
    Merges two dataframes and prefixes columns from the second dataframe with a given name.

    Args:
        df1 (pd.DataFrame): The first dataframe to be merged.
        df2 (pd.DataFrame): The second dataframe to be merged and whose columns will be prefixed.
        p_name (str): The prefix to add to the columns of the second dataframe.
        p_cols (List[str]): List of column names in the second dataframe that need to be prefixed.
        p_id_col (str): The column name used for merging the two dataframes.

    Returns:
        pd.DataFrame: The merged dataframe with prefixed columns from the second dataframe.
    """
    # Prefix the specified columns in df2
    prefixed_cols = {col: f"{p_name}_{col}" for col in p_cols}
    list_cols = [f"{p_name}_{col}" for col in p_cols]
    df2_prefixed = df2.rename(columns=prefixed_cols)
    list_cols.append(p_id_col)
    df2_prefixed = df2_prefixed[list_cols]  # clean before merging

    # Merge df1 and df2 based on the id column
    merged_df = pd.merge(df1, df2_prefixed, on=p_id_col, sort=False)

    return merged_df


def merge_multiple_dataframes(
    df_list: List[pd.DataFrame], p_names: List[str], p_cols_list: List[List[str]], p_id_col: str
) -> pd.DataFrame:
    """
    Merges multiple dataframes with prefixed columns based on a list of names and columns.

    Args:
        df_list (List[pd.DataFrame]): A list of dataframes to be merged.
        p_names (List[str]): List of prefixes for the columns of each dataframe.
        p_cols_list (List[List[str]]): List of lists of column names in each dataframe that need to be prefixed.
        p_id_col (str): The column name used for merging all the dataframes.

    Returns:
        pd.DataFrame: The final merged dataframe with all prefixed columns from the input dataframes.
    """

    # Start with the first dataframe
    result_df = df_list[0]

    # Iterate over the remaining dataframes, names, and corresponding columns
    for i in range(1, len(df_list)):
        df = df_list[i]
        name = p_names[i - 1]  # Since we start merging from the second dataframe
        cols = p_cols_list[i - 1]

        result_df = merge_and_prefix_columns(result_df, df, name, cols, p_id_col)

    return result_df


def merge_multiple_dataframes_from_parquet(
    file_paths: List[Union[str, Path]],
    p_names: List[str],
    p_cols_list: List[str],
    p_id_col: str,
    p_float_16: bool = False,
    p_list_mode: bool = False,
    p_list_df: List[pd.DataFrame] = [],
) -> pd.DataFrame:
    """
    Merges multiple dataframes read from Parquet files with prefixed columns based on a list of names and columns.

    Args:
        file_paths (List[Union[str, Path]]): A list of file paths to Parquet files.
        p_names (List[str]): List of prefixes for the columns of each dataframe.
        p_cols_list (List[List[str]]): List of lists of column names in each dataframe that need to be prefixed.
        p_id_col (str): The column name used for merging all the dataframes.

    Returns:
        pd.DataFrame: The final merged dataframe with all prefixed columns from the input Parquet files.

    Raises:
        FileNotFoundError: If any file path is invalid or file not found.
        pd.errors.EmptyDataError: If any Parquet file is empty.
        Exception: For other errors during reading or merging, logged appropriately.
    """
    try:
        # Read the first dataframe from the first Parquet file
        if p_list_mode:
            result_df = p_list_df[0]
            # logger_s.info('list mode ', result_df.shape())
        else:
            if p_float_16:
                result_df = read_parquet_to_f16(file_paths[0])
            else:
                result_df = pd.read_parquet(file_paths[0])

        if len(p_cols_list) == 0:
            p_cols_list = list(result_df.columns)
            p_cols_list.remove(g_index_col)

        result_df = prefix_and_select_columns(result_df, p_names[0], p_cols_list, p_id_col)

        for i in range(1, len(file_paths)):
            file_path = file_paths[i]
            name = p_names[i]

            try:
                # Read each Parquet file incrementally
                if p_list_mode:
                    df = p_list_df[i]
                else:
                    if p_float_16:
                        df = read_parquet_to_f16(file_path)
                    else:
                        df = pd.read_parquet(file_path)

                # Validate that required columns exist before merging
                if p_id_col not in result_df.columns or p_id_col not in df.columns:
                    raise ValueError(f"Missing id column {p_id_col}")

                logger_s.info(f"Merging dataframe {i} with prefix {name}")
                if len(p_cols_list) == 0:
                    p_cols_list = list(df.columns)
                    p_cols_list.remove(g_index_col)
                result_df = merge_and_prefix_columns(result_df, df, name, p_cols_list, p_id_col)

            except (IOError, pd.errors.EmptyDataError) as e:
                logger_s.error(f"Failed to read or process file {file_path}: {str(e)}")
                raise

        return result_df

    except IndexError:
        raise ValueError("List of names and columns must be same length as file_paths list")

    except Exception as e:
        logger_s.error(f"Error during merging: {str(e)}", exc_info=True)
        raise


# Example use case
if __name__ == "__main__":
    data1 = {"id": [1, 2, 3], "value1": [10, 20, 30]}
    data2 = {"id": [1, 2, 4], "col1": ["a", "b", "c"], "col2": [1, 2, 3]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    result_df = merge_and_prefix_columns(df1, df2, "prefix", ["col1", "col2"], "id")
    print(result_df)

    data1 = {"id": [1, 2, 3], "value1": [10, 20, 30]}
    data2 = {"id": [1, 2, 4], "col1": ["a", "b", "c"], "col2": [1, 2, 3]}
    data3 = {"id": [1, 2, 5], "colA": [True, False, True], "colB": [0.1, 0.2, 0.3]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)

    result_df = merge_multiple_dataframes(
        [df1, df2, df3], ["prefix2", "prefix3"], [["col1", "col2"], ["colA", "colB"]], "id"
    )
    print(result_df)

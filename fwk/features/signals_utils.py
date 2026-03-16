import pandas as pd
import numpy as np

def map_P_to_N(df, column):
    """
    Map 'P' values to 'N' in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    column (str): Column name to transform
    
    Returns:
    pd.DataFrame: Dataframe with transformed values
    """
    df = df.copy()
    df[column] = df[column].replace('P', 'N')
    return df

def map_minus1_0_1_to_0_1_2(df, column):
    """
    Map values -1, 0, 1 to 0, 1, 2 respectively.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    column (str): Column name to transform
    
    Returns:
    pd.DataFrame: Dataframe with transformed values
    """
    df = df.copy()
    mapping = {-1: 0, 0: 1, 1: 2}
    df[column] = df[column].map(mapping)
    return df

def logical_and(df, columns, result_col='and_result'):
    """
    Perform logical AND operation across specified columns.
    True if all values are True/1, False otherwise.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to apply AND operation
    result_col (str): Name of result column
    
    Returns:
    pd.DataFrame: Dataframe with AND result column
    """
    df = df.copy()
    # Convert to boolean (consider 1, True, 'True' as True)
    bool_df = df[columns].applymap(lambda x: bool(x) if isinstance(x, (bool, int)) 
                                   else str(x).lower() == 'true' or x == 1)
    df[result_col] = bool_df.all(axis=1)
    return df

def logical_or(df, columns, result_col='or_result'):
    """
    Perform logical OR operation across specified columns.
    True if at least one value is True/1.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to apply OR operation
    result_col (str): Name of result column
    
    Returns:
    pd.DataFrame: Dataframe with OR result column
    """
    df = df.copy()
    bool_df = df[columns].applymap(lambda x: bool(x) if isinstance(x, (bool, int)) 
                                   else str(x).lower() == 'true' or x == 1)
    df[result_col] = bool_df.any(axis=1)
    return df

def logical_nor(df, columns, result_col='nor_result'):
    """
    Perform logical NOR operation across specified columns.
    True if all values are False/0.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to apply NOR operation
    result_col (str): Name of result column
    
    Returns:
    pd.DataFrame: Dataframe with NOR result column
    """
    df = df.copy()
    bool_df = df[columns].applymap(lambda x: bool(x) if isinstance(x, (bool, int)) 
                                   else str(x).lower() == 'true' or x == 1)
    df[result_col] = ~bool_df.any(axis=1)
    return df

def logical_nand(df, columns, result_col='nand_result'):
    """
    Perform logical NAND operation across specified columns.
    True if at least one value is False/0.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to apply NAND operation
    result_col (str): Name of result column
    
    Returns:
    pd.DataFrame: Dataframe with NAND result column
    """
    df = df.copy()
    bool_df = df[columns].applymap(lambda x: bool(x) if isinstance(x, (bool, int)) 
                                   else str(x).lower() == 'true' or x == 1)
    df[result_col] = ~bool_df.all(axis=1)
    return df

def logical_xor(df, columns, result_col='xor_result'):
    """
    Perform logical XOR operation across specified columns.
    True if an odd number of values are True.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to apply XOR operation
    result_col (str): Name of result column
    
    Returns:
    pd.DataFrame: Dataframe with XOR result column
    """
    df = df.copy()
    bool_df = df[columns].applymap(lambda x: bool(x) if isinstance(x, (bool, int)) 
                                   else str(x).lower() == 'true' or x == 1)
    df[result_col] = bool_df.sum(axis=1) % 2 == 1
    return df

def boolean_filter(df, condition_col, value=True):
    """
    Filter dataframe based on boolean condition.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    condition_col (str): Column name with boolean values
    value (bool): Value to filter on (True or False)
    
    Returns:
    pd.DataFrame: Filtered dataframe
    """
    return df[df[condition_col] == value].copy()

def convert_to_boolean(df, columns, true_values=None, false_values=None):
    """
    Convert specified columns to boolean values.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to convert
    true_values (list): Values to consider as True (default: [1, '1', 'True', 'true', True])
    false_values (list): Values to consider as False (default: [0, '0', 'False', 'false', False])
    
    Returns:
    pd.DataFrame: Dataframe with converted boolean columns
    """
    if true_values is None:
        true_values = [1, '1', 'True', 'true', True]
    if false_values is None:
        false_values = [0, '0', 'False', 'false', False]
    
    df = df.copy()
    for col in columns:
        df[col] = df[col].apply(lambda x: True if x in true_values 
                                else False if x in false_values 
                                else bool(x))
    return df

def conditional_set(df, condition_col, target_col, condition_value, set_value):
    """
    Set values in target column based on condition in another column.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    condition_col (str): Column to check condition
    target_col (str): Column to modify
    condition_value: Value to check for in condition column
    set_value: Value to set in target column when condition is met
    
    Returns:
    pd.DataFrame: Dataframe with modified values
    """
    df = df.copy()
    df.loc[df[condition_col] == condition_value, target_col] = set_value
    return df

# Example usage and testing
if __name__ == "__main__":
    # Create sample dataframe
    data = {
        'col1': ['P', 'N', 'P', 'N', 'P'],
        'col2': [-1, 0, 1, -1, 0],
        'col3': [True, False, True, False, True],
        'col4': [1, 0, 1, 0, 1],
        'col5': ['True', 'False', 'True', 'False', 'True']
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Test map_P_to_N
    df1 = map_P_to_N(df, 'col1')
    print("After mapping 'P' to 'N':")
    print(df1[['col1']])
    print()
    
    # Test map_minus1_0_1_to_0_1_2
    df2 = map_minus1_0_1_to_0_1_2(df, 'col2')
    print("After mapping -1,0,1 to 0,1,2:")
    print(df2[['col2']])
    print()
    
    # Test logical operations
    df3 = logical_and(df, ['col3', 'col4'], 'and_result')
    df3 = logical_or(df3, ['col3', 'col4'], 'or_result')
    df3 = logical_nor(df3, ['col3', 'col4'], 'nor_result')
    df3 = logical_nand(df3, ['col3', 'col4'], 'nand_result')
    df3 = logical_xor(df3, ['col3', 'col4'], 'xor_result')
    
    print("Logical operations on col3 and col4:")
    print(df3[['col3', 'col4', 'and_result', 'or_result', 'nor_result', 'nand_result', 'xor_result']])
    print()
    
    # Test boolean filter
    filtered_df = boolean_filter(df3, 'and_result', True)
    print("Filtered where AND result is True:")
    print(filtered_df[['col3', 'col4']])
    print()
    
    # Test convert to boolean
    df4 = convert_to_boolean(df, ['col1', 'col5'])
    print("After converting to boolean:")
    print(df4[['col1', 'col5']])
    print()
    
    # Test conditional set
    df5 = conditional_set(df, 'col2', 'col1', -1, 'CONDITION_MET')
    print("After conditional set (col2=-1 sets col1 to 'CONDITION_MET'):")
    print(df5[['col1', 'col2']])
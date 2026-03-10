

from typing import Dict, Any



def add_stop_loss_signal_yaml(
    yaml_conf_dict: Dict[str, Any],
    asset: str,
    close_col: str = "S_close_f32",
    high_col: str = "S_high_f32",
    low_col: str = "S_low_f32",
    up_objective: float = 0.012,
    up_stoploss: float = -0.004,
    down_objective: float = -0.012,
    down_stoploss: float = 0.004,
    N_periods: int = 240
) -> Dict[str, Any]:
    """
    Add a stop-loss signal specification to the 'targets_classification' section of a YAML configuration dictionary.

    Parameters:
        yaml_conf_dict (Dict[str, Any]): The existing YAML configuration dictionary.
        asset (str): The asset symbol (e.g., 'BTC', 'ETH')
        close_col (str): Name of the close price column
        high_col (str): Name of the high price column
        low_col (str): Name of the low price column
        up_objective (float): Target upside gain threshold
        up_stoploss (float): Stop-loss level for upward trades
        down_objective (float): Target downside loss threshold
        down_stoploss (float): Stop-loss level for downward trades
        N_periods (int): Number of periods to look back

    Returns:
        Dict[str, Any]: The updated configuration dictionary (modified in-place and returned).
    """
    # Ensure 'targets_classification' exists and is a dict
    if "targets_classification" not in yaml_conf_dict:
        yaml_conf_dict["targets_classification"] = {}

    # Ensure 'targets_classification' is a dictionary
    if not isinstance(yaml_conf_dict["targets_classification"], dict):
        raise TypeError("'targets_classification' must be a dictionary")

    # Define the new signal configuration
    new_signal = {
        f"{asset}_stop_loss_signal_class_{N_periods}": {
            "type": "single_asset",
            "asset": asset,
            "function": "gen_perfect_stoploss_signal_class",
            "params": {
                "close_col": close_col,
                "high_col": high_col,
                "low_col": low_col,
                "up_objective": up_objective,
                "up_stoploss": up_stoploss,
                "down_objective": down_objective,
                "down_stoploss": down_stoploss,
                "N_periods": N_periods
            }
        }
    }

    # Merge the new signal into targets_classification
    yaml_conf_dict["targets_classification"].update(new_signal)

    return yaml_conf_dict 
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Union

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)



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


def load_yaml_from_md(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration from a markdown file containing a YAML code block.

    Args:
        filepath: Path to .md file with YAML configuration.

    Returns:
        Dict[str, Any]: Parsed YAML configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no YAML block found in the file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8")

    yaml_block_match = re.search(
        r"```yaml\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE
    )
    if not yaml_block_match:
        raise ValueError(f"No YAML code block found in {filepath}")

    yaml_content = yaml_block_match.group(1)

    yaml = YAML()
    config = yaml.load(yaml_content)

    if not config:
        raise ValueError(f"Invalid YAML content in {filepath}")

    return config


def add_perfect_signal_class_yaml(
    yaml_conf_dict: Dict[str, Any],
    asset: str,
    close_col: str = "S_close_f32",
    high_col: str = "S_high_f32",
    low_col: str = "S_low_f32",
    upstrong_val: float = 0.02,
    downstrong_val: float = -0.02,
    flat_val: float = 0.005,
    N_periods: int = 60,
) -> Dict[str, Any]:
    """
    Add a perfect signal class specification to the 'targets_classification' section.

    Args:
        yaml_conf_dict: The existing YAML configuration dictionary.
        asset: The asset symbol (e.g., 'BTC', 'ETH')
        close_col: Name of the close price column
        high_col: Name of the high price column
        low_col: Name of the low price column
        upstrong_val: Threshold for strong upward class
        downstrong_val: Threshold for strong downward class
        flat_val: Threshold between neutral and weak movement
        N_periods: Number of periods to look ahead

    Returns:
        The updated configuration dictionary.
    """
    if "targets_classification" not in yaml_conf_dict:
        yaml_conf_dict["targets_classification"] = {}

    if not isinstance(yaml_conf_dict["targets_classification"], dict):
        raise TypeError("'targets_classification' must be a dictionary")

    new_signal = {
        f"{asset}_perfect_signal_class_{N_periods}": {
            "type": "single_asset",
            "asset": asset,
            "function": "gen_perfect_signal_class",
            "params": {
                "close_col": close_col,
                "high_col": high_col,
                "low_col": low_col,
                "upstrong_val": upstrong_val,
                "downstrong_val": downstrong_val,
                "flat_val": flat_val,
                "N_periods": N_periods,
            },
        }
    }

    yaml_conf_dict["targets_classification"].update(new_signal)

    return yaml_conf_dict


def add_spread_signal_yaml(
    yaml_conf_dict: Dict[str, Any],
    id_spread: str,
    asset1: str,
    factor1: float,
    asset2: str,
    factor2: float,
    function: str = "gen_perfect_signal_class",
    close_col: str = "S_close_f32",
    high_col: str = "S_high_f32",
    low_col: str = "S_low_f32",
    N_periods: int = 60,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Add a spread signal specification to the 'targets_classification' section.

    Args:
        yaml_conf_dict: The existing YAML configuration dictionary.
        id_spread: Identifier for the spread target.
        asset1: First asset symbol.
        factor1: Multiplier for first asset.
        asset2: Second asset symbol.
        factor2: Multiplier for second asset.
        function: Target generation function name.
        close_col: Name of the close price column.
        high_col: Name of the high price column.
        low_col: Name of the low price column.
        N_periods: Number of periods to look ahead.
        **kwargs: Additional parameters for the function.

    Returns:
        The updated configuration dictionary.
    """
    if "targets_classification" not in yaml_conf_dict:
        yaml_conf_dict["targets_classification"] = {}

    if not isinstance(yaml_conf_dict["targets_classification"], dict):
        raise TypeError("'targets_classification' must be a dictionary")

    params = {
        "close_col": close_col,
        "high_col": high_col,
        "low_col": low_col,
        "N_periods": N_periods,
        **kwargs,
    }

    new_signal = {
        f"{id_spread}_signal": {
            "type": "spread_asset",
            "asset1": asset1,
            "factor1": factor1,
            "asset2": asset2,
            "factor2": factor2,
            "function": function,
            "params": params,
        }
    }

    yaml_conf_dict["targets_classification"].update(new_signal)

    return yaml_conf_dict 
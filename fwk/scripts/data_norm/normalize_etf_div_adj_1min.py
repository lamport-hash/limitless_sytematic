"""
Normalize all firstrate ETF dividend-adjusted 1min data files.

This script processes all .txt files in data/import/firstrate/etf_div_adj/
and converts them to normalized parquet format in data/normalised/.
"""

import sys
from pathlib import Path
from tqdm import tqdm

from core.data_org import (
    get_import_file,
    get_normalised_file,
    create_normalised_dirs,
    ProductType,
)
from core.enums import MktDataTFreq, ExchangeNAME
from norm.norm_firstrate_etf_div_adj import norm_firstrate_etf_div_adj


def normalize_all_etf_div_adj_1min(p_test_single: bool = False, p_symbol: str = None):
    """Normalize all ETF dividend-adjusted 1min files from firstrate.
    
    Args:
        p_test_single: If True, only process one file for testing.
        p_symbol: If provided, only process this specific symbol (e.g., "QQQ").
    """
    
    subtype = "etf_div_adj"
    source = ExchangeNAME.FIRSTRATE
    data_freq = MktDataTFreq.CANDLE_1MIN
    product_type = ProductType.ETF

    import_dir = Path(__file__).parent.parent.parent / "data" / "import" / "firstrate" / "etf_div_adj"
    
    txt_files = sorted(import_dir.glob("*.txt"))
    total_files = len(txt_files)
    
    if p_symbol:
        txt_files = [f for f in txt_files if f.stem.split("_")[0] == p_symbol.upper()]
        print(f"SYMBOL MODE: Processing only {p_symbol.upper()}")
    elif p_test_single:
        txt_files = txt_files[:1]
        print(f"TEST MODE: Processing only 1 file")
    
    existing_symbols = set()
    output_dir = Path(__file__).parent.parent.parent / "data" / "normalised" / data_freq.value / f"{source.value}_undefined" / product_type.value
    if output_dir.exists():
        existing_symbols = {d.name for d in output_dir.iterdir() if d.is_dir()}
    
    files_to_process = [f for f in txt_files if f.stem.split("_")[0] not in existing_symbols]
    remaining = len(files_to_process)
    
    print(f"Found {total_files} total files")
    print(f"Already processed: {total_files - remaining}")
    print(f"To process: {remaining}")
    print(f"Source: {source.value}")
    print(f"Subtype: {subtype}")
    print(f"Output: {data_freq.value}/{source.value}_undefined/{product_type.value}/")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    for txt_file in tqdm(files_to_process, desc="Normalizing"):
        symbol = txt_file.stem.split("_")[0]
        
        try:
            df, start_date, end_date = norm_firstrate_etf_div_adj(str(txt_file), "/tmp/temp_norm.df.parquet")
            
            output_file = get_normalised_file(
                data_freq,
                source,
                product_type,
                symbol,
                p_start=start_date,
                p_end=end_date,
            )
            
            create_normalised_dirs(
                data_freq,
                source,
                product_type,
                symbol,
            )
            
            df.to_pickle(str(output_file), compression="gzip")
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"\nError processing {symbol}: {e}")
    
    print("-" * 60)
    print(f"Completed: {success_count} succeeded, {error_count} failed")
    print(f"Total processed: {total_files - remaining + success_count}/{total_files}")
    
    return success_count, error_count


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    symbol = None
    
    for i, arg in enumerate(sys.argv):
        if arg == "--symbol" and i + 1 < len(sys.argv):
            symbol = sys.argv[i + 1]
            break
    
    normalize_all_etf_div_adj_1min(p_test_single=test_mode, p_symbol=symbol)

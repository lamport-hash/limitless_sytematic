"""
Normalize all firstrate FX 1h data files.

This script processes all .txt files in data/import/firstrate/fx_1h/
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
    list_instruments,
)
from core.enums import MktDataTFreq, ExchangeNAME
from norm.norm_firstrate_fx import norm_firstrate_fx


def normalize_all_fx_1h(p_test_single: bool = False):
    """Normalize all FX 1h files from firstrate.
    
    Args:
        p_test_single: If True, only process one file for testing.
    """
    
    subtype = "fx_1h"
    source = ExchangeNAME.FIRSTRATE
    data_freq = MktDataTFreq.CANDLE_1HOUR
    product_type = ProductType.SPOT

    import_dir = Path(__file__).parent.parent.parent / "data" / "import" / "firstrate" / "fx_1h"
    
    txt_files = sorted(import_dir.glob("*.txt"))
    total_files = len(txt_files)
    
    if p_test_single:
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
            input_file = get_import_file(source, symbol, subtype)
            
            df, start_date, end_date = norm_firstrate_fx(str(txt_file), "/tmp/temp_norm.df.parquet")
            
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
    normalize_all_fx_1h(p_test_single=test_mode)

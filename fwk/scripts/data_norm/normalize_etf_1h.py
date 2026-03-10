"""
Normalize all firstrate ETF 1h data files.

This script processes all .txt files in data/import/firstrate/etf_1h/
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
from core.enums import MktDataFred, ExchangeNAME
from norm.norm_firstdata import norm_firstdata


def normalize_all_etf_1h():
    """Normalize all ETF 1h files from firstrate."""
    
    subtype = "etf_1h"
    source = ExchangeNAME.FIRSTRATE
    data_freq = MktDataFred.CANDLE_1HOUR
    product_type = ProductType.ETF

    import_dir = Path("/home/brian/sing/data/import/firstrate/etf_1h")
    output_dir = Path("/home/brian/sing/data/normalised/candle_1hour/firstrate_undefined/etf")
    
    txt_files = list(import_dir.glob("*.txt"))
    total_files = len(txt_files)
    
    existing_symbols = set()
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
            output_file = get_normalised_file(
                data_freq,
                source,
                product_type,
                symbol,
            )
            
            create_normalised_dirs(
                data_freq,
                source,
                product_type,
                symbol,
            )
            
            load_and_map_columns_firstrate(str(input_file), str(output_file))
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"\nError processing {symbol}: {e}")
    
    print("-" * 60)
    print(f"Completed: {success_count} succeeded, {error_count} failed")
    print(f"Total processed: {total_files - remaining + success_count}/{total_files}")
    

if __name__ == "__main__":
    normalize_all_etf_1h()

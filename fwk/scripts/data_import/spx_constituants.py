import zipfile
import os
import pandas as pd
from datetime import datetime

def load_constituent_stocks(constituent_file, target_date, zip_folder):
    """
    Load stock data for all constituents on a specific date.
    
    Parameters:
    constituent_file (str): Path to the constituent index file
    target_date (str): Date in format 'YYYY-MM-DD'
    zip_folder (str): Folder containing zip files named stock_A***.zip, stock_B***.zip, etc.
    
    Returns:
    dict: Dictionary with ticker as key and DataFrame of stock data as value
    """
    
    # Read the constituent file
    constituents_df = pd.read_csv(constituent_file, header=None, names=['date', 'tickers'])
    
    # Find the row for the target date
    target_row = constituents_df[constituents_df['date'] == target_date]
    
    if len(target_row) == 0:
        raise ValueError(f"Date {target_date} not found in constituent file")
    
    # Get the tickers for that date and split them
    tickers = target_row.iloc[0]['tickers'].split(',')
    tickers = [ticker.strip() for ticker in tickers]  # Remove any whitespace
    
    print(f"Found {len(tickers)} constituents for date {target_date}")
    
    # Dictionary to store stock data
    stock_data = {}
    
    # Group tickers by their first letter for efficient zip file access
    tickers_by_letter = {}
    for ticker in tickers:
        first_letter = ticker[0].upper()
        if first_letter not in tickers_by_letter:
            tickers_by_letter[first_letter] = []
        tickers_by_letter[first_letter].append(ticker)
    
    # Process each letter group
    for letter, letter_tickers in tickers_by_letter.items():
        # Find the zip file for this letter
        zip_pattern = f"stock_{letter}***.zip"
        zip_file = None
        
        for file in os.listdir(zip_folder):
            if file.startswith(f"stock_{letter}") and file.endswith('.zip'):
                zip_file = os.path.join(zip_folder, file)
                break
        
        if zip_file is None:
            print(f"Warning: No zip file found for letter {letter}")
            continue
        
        print(f"Processing zip file: {zip_file}")
        
        # Open the zip file
        with zipfile.ZipFile(zip_file, 'r') as zf:
            # For each ticker in this group
            for ticker in letter_tickers:
                # Try different possible file name formats
                possible_filenames = [
                    f"{ticker}.csv",
                    f"{ticker}.txt",
                    f"{ticker}.parquet",
                    f"{ticker.lower()}.csv",
                    f"{ticker.upper()}.csv"
                ]
                
                found = False
                for filename in possible_filenames:
                    if filename in zf.namelist():
                        try:
                            # Read the file content
                            with zf.open(filename) as f:
                                # Try to read as CSV first
                                try:
                                    df = pd.read_csv(f)
                                    stock_data[ticker] = df
                                    print(f"  Loaded {ticker}")
                                    found = True
                                    break
                                except:
                                    # If not CSV, try reading as text and parse accordingly
                                    f.seek(0)  # Reset file pointer
                                    content = f.read().decode('utf-8')
                                    # Parse based on your data format
                                    # This is a placeholder - adjust based on actual data format
                                    lines = content.strip().split('\n')
                                    # Process lines as needed
                                    stock_data[ticker] = lines
                                    print(f"  Loaded {ticker} (as text)")
                                    found = True
                                    break
                        except Exception as e:
                            print(f"  Error loading {ticker}: {e}")
                
                if not found:
                    print(f"  Warning: Could not find file for {ticker}")
    
    return stock_data

# Alternative version that returns a combined DataFrame
def load_constituent_stocks_combined(constituent_file, target_date, zip_folder):
    """
    Load stock data for all constituents and combine into a single DataFrame.
    
    Parameters:
    constituent_file (str): Path to the constituent index file
    target_date (str): Date in format 'YYYY-MM-DD'
    zip_folder (str): Folder containing zip files named stock_A***.zip, stock_B***.zip, etc.
    
    Returns:
    pandas.DataFrame: Combined DataFrame with all stock data
    """
    
    stock_data = load_constituent_stocks(constituent_file, target_date, zip_folder)
    
    # Combine all DataFrames (assuming they have the same structure)
    combined_df = pd.concat(stock_data.values(), keys=stock_data.keys())
    
    return combined_df

# Example usage
if __name__ == "__main__":
    try:
        # Example call
        result = load_constituent_stocks(
            constituent_file="constituents.csv",
            target_date="1996-08-19",
            zip_folder="./stock_zips"
        )
        
        print(f"\nSuccessfully loaded {len(result)} stocks")
        
        # Print first few rows of first stock as example
        if result:
            first_ticker = list(result.keys())[0]
            print(f"\nSample data for {first_ticker}:")
            print(result[first_ticker].head())
            
    except Exception as e:
        print(f"Error: {e}")
import os
import pandas as pd

def main():
    # Make sure output directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    print("="*60)
    print("STEP A: Load Datasets")
    print("="*60)
    
    # Paths
    fg_path = 'data/fear_greed_index.csv'
    trades_path = 'data/historical_data.csv'
    
    print(f"Loading Fear & Greed Index from: {fg_path}")
    df_fg = pd.read_csv(fg_path)
    print(f"Loading Historical Trader Data from: {trades_path}")
    df_trades = pd.read_csv(trades_path)
    
    print("\n--- Fear & Greed Index Info ---")
    print(f"Shape: {df_fg.shape}")
    print(f"Columns: {list(df_fg.columns)}")
    print("\nDtypes:")
    print(df_fg.dtypes)
    print("\nHead 5:")
    print(df_fg.head(5))
    print("\nTail 5:")
    print(df_fg.tail(5))
    
    print("\n--- Historical Trader Data Info ---")
    print(f"Shape: {df_trades.shape}")
    print(f"Columns: {list(df_trades.columns)}")
    print("\nDtypes:")
    print(df_trades.dtypes)
    print("\nHead 5:")
    print(df_trades.head(5))
    print("\nTail 5:")
    print(df_trades.tail(5))
    
    print("\n" + "="*60)
    print("STEP B: Inspect for Issues")
    print("="*60)
    
    print("--- Fear & Greed Index Nulls ---")
    print(df_fg.isnull().sum())
    
    print("\n--- Historical Trader Data Nulls ---")
    print(df_trades.isnull().sum())
    
    print(f"\nFear & Greed Index duplicates: {df_fg.duplicated().sum()}")
    print(f"Historical Trader Data duplicates: {df_trades.duplicated().sum()}")
    
    print("\n--- Unique Values in Trader Data ---")
    if 'Side' in df_trades.columns:
        print(f"Unique 'Side' values: {df_trades['Side'].unique()}")
    else:
        print("Side column not found!")
        
    if 'Direction' in df_trades.columns:
        print(f"Unique 'Direction' values: {df_trades['Direction'].unique()}")
    else:
        print("Direction column not found!")
        
    # Check for leverage or margin columns
    leverage_cols = [col for col in df_trades.columns if 'lev' in col.lower() or 'margin' in col.lower()]
    print(f"\nLeverage/Margin columns found: {leverage_cols}")
    print(f"Full column list of trader data:\n{list(df_trades.columns)}")
    
    print("\n--- Fear & Greed Classification Unique Values ---")
    if 'classification' in df_fg.columns:
        print(f"Unique 'classification' values: {df_fg['classification'].unique()}")
    else:
        print("classification column not found!")
        
    if 'date' in df_fg.columns:
        print(f"Fear & Greed original Date Range: {df_fg['date'].min()} to {df_fg['date'].max()}")
    
    print("\n" + "="*60)
    print("STEP C: Clean and Standardize Dates")
    print("="*60)
    
    # Clean Fear & Greed Date
    print("Converting Fear & Greed 'date' to datetime...")
    df_fg['date'] = pd.to_datetime(df_fg['date'])
    print(f"Fear & Greed index 'date' dtype: {df_fg['date'].dtype}")
    
    # Clean Trades Date
    print("Converting Trader Data 'Timestamp IST' to datetime...")
    # Timestamp IST is in format DD-MM-YYYY HH:MM
    df_trades['timestamp'] = pd.to_datetime(df_trades['Timestamp IST'], dayfirst=True)
    # Derive date-only column (no time component) as datetime
    df_trades['date'] = pd.to_datetime(df_trades['timestamp'].dt.date)
    print(f"Trader Data 'timestamp' dtype: {df_trades['timestamp'].dtype}")
    print(f"Trader Data 'date' dtype: {df_trades['date'].dtype}")
    
    # Check ranges and overlap
    fg_min, fg_max = df_fg['date'].min(), df_fg['date'].max()
    trades_min, trades_max = df_trades['date'].min(), df_trades['date'].max()
    
    print(f"\nFear & Greed Date Range: {fg_min.strftime('%Y-%m-%d')} to {fg_max.strftime('%Y-%m-%d')}")
    print(f"Trader Data Date Range: {trades_min.strftime('%Y-%m-%d')} to {trades_max.strftime('%Y-%m-%d')}")
    
    overlap_min = max(fg_min, trades_min)
    overlap_max = min(fg_max, trades_max)
    
    if overlap_min <= overlap_max:
        print(f"Date Overlap: {overlap_min.strftime('%Y-%m-%d')} to {overlap_max.strftime('%Y-%m-%d')}")
        overlap_days = (overlap_max - overlap_min).days + 1
        print(f"Number of overlapping days: {overlap_days}")
    else:
        print("LOUD WARNING: NO DATE OVERLAP DETECTED BETWEEN DATASETS!")
        
    print("\n" + "="*60)
    print("STEP D: Handle Known Data Quirks")
    print("="*60)
    
    # PnL quirks
    if 'Closed PnL' in df_trades.columns:
        pnl_series = df_trades['Closed PnL']
        total_rows = len(df_trades)
        zero_pnl_count = (pnl_series == 0).sum()
        nonzero_pnl_count = (pnl_series != 0).sum()
        null_pnl_count = pnl_series.isnull().sum()
        
        print(f"Closed PnL breakdown:")
        print(f"  Closed PnL == 0: {zero_pnl_count} rows ({zero_pnl_count / total_rows * 100:.2f}%)")
        print(f"  Closed PnL != 0: {nonzero_pnl_count} rows ({nonzero_pnl_count / total_rows * 100:.2f}%)")
        print(f"  Closed PnL is Null: {null_pnl_count} rows ({null_pnl_count / total_rows * 100:.2f}%)")
    else:
        print("Closed PnL column not found!")
        
    # Execution Price, Size Tokens, Size USD numerical inspect
    numeric_cols = ['Execution Price', 'Size Tokens', 'Size USD']
    for col in numeric_cols:
        if col in df_trades.columns:
            print(f"\nNumerical description for '{col}':")
            print(df_trades[col].describe())
            
            # Check for negative or zero values
            neg_count = (df_trades[col] < 0).sum()
            zero_count = (df_trades[col] == 0).sum()
            print(f"  Negative values count: {neg_count}")
            print(f"  Zero values count: {zero_count}")
        else:
            print(f"Column '{col}' not found in Trader Data!")
            
    print("\n" + "="*60)
    print("STEP E: Save Intermediate Output")
    print("="*60)
    
    fg_clean_path = 'data/processed/fear_greed_clean.csv'
    trades_clean_path = 'data/processed/trades_clean.csv'
    
    print(f"Saving cleaned Fear & Greed index to {fg_clean_path}")
    df_fg.to_csv(fg_clean_path, index=False)
    
    print(f"Saving cleaned Trader Data to {trades_clean_path}")
    df_trades.to_csv(trades_clean_path, index=False)
    
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    
    # We will compute findings to print
    summary_findings = []
    
    # Null summary
    fg_nulls = df_fg.isnull().sum().sum()
    trades_nulls = df_trades.isnull().sum().sum()
    if fg_nulls > 0:
        summary_findings.append(f"- Fear & Greed Index contains {fg_nulls} null values.")
    else:
        summary_findings.append("- Fear & Greed Index contains 0 null values.")
        
    if trades_nulls > 0:
        summary_findings.append(f"- Trader Data contains {trades_nulls} null values. Details:\n{df_trades.isnull().sum()[df_trades.isnull().sum() > 0].to_string()}")
    else:
        summary_findings.append("- Trader Data contains 0 null values.")
        
    # Duplicate summary
    if df_fg.duplicated().sum() > 0:
        summary_findings.append(f"- Fear & Greed Index has {df_fg.duplicated().sum()} duplicate rows.")
    if df_trades.duplicated().sum() > 0:
        summary_findings.append(f"- Trader Data has {df_trades.duplicated().sum()} duplicate rows.")
        
    # Leverage summary
    if len(leverage_cols) > 0:
        summary_findings.append(f"- Leverage/Margin columns found: {leverage_cols}")
    else:
        summary_findings.append("- Leverage/Margin columns NOT found in the dataset.")
        
    # Date overlap summary
    if overlap_min <= overlap_max:
        summary_findings.append(f"- Overlapping date range: {overlap_min.strftime('%Y-%m-%d')} to {overlap_max.strftime('%Y-%m-%d')} ({overlap_days} days).")
    else:
        summary_findings.append("- WARNING: No overlapping date range found!")
        
    # PnL zero percentage
    if 'Closed PnL' in df_trades.columns:
        summary_findings.append(f"- Closed PnL: {zero_pnl_count / total_rows * 100:.2f}% of rows are exactly 0 (confirming fill-level data structure).")
        
    # Numeric column anomalies
    for col in numeric_cols:
        if col in df_trades.columns:
            neg_count = (df_trades[col] < 0).sum()
            zero_count = (df_trades[col] == 0).sum()
            if neg_count > 0 or zero_count > 0:
                summary_findings.append(f"- Column '{col}' has anomalies: {neg_count} negatives, {zero_count} zeros.")
                
    for finding in summary_findings:
        print(finding)
    print("="*60)

if __name__ == '__main__':
    main()

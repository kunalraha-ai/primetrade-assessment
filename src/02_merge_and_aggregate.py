import os
import pandas as pd
import numpy as np

def main():
    # Make sure output directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Paths
    fg_clean_path = 'data/processed/fear_greed_clean.csv'
    trades_clean_path = 'data/processed/trades_clean.csv'
    
    # Check if files exist
    if not os.path.exists(fg_clean_path) or not os.path.exists(trades_clean_path):
        print("ERROR: Processed files do not exist. Please run step 1 first.")
        return
        
    print("="*60)
    print("STEP A: Load & Merge Trades with Fear & Greed Index")
    print("="*60)
    
    print(f"Loading cleaned Fear & Greed Index from: {fg_clean_path}")
    df_fg = pd.read_csv(fg_clean_path)
    print(f"Loading cleaned Trader Data from: {trades_clean_path}")
    df_trades = pd.read_csv(trades_clean_path)
    
    # Parse dates to ensure clean joining
    df_fg['date'] = pd.to_datetime(df_fg['date'])
    df_trades['date'] = pd.to_datetime(df_trades['date'])
    
    # Left join trades on date with fear_greed
    print("Performing left-join on 'date'...")
    df_merged = pd.merge(
        df_trades, 
        df_fg[['date', 'classification', 'value']], 
        on='date', 
        how='left'
    )
    
    # Check for failed matches
    unmatched_count = df_merged['classification'].isnull().sum()
    total_rows = len(df_merged)
    print(f"Merge completed. Total trade rows: {total_rows}")
    print(f"Trade rows failing to match sentiment (null classification): {unmatched_count} ({unmatched_count / total_rows * 100:.4f}%)")
    if unmatched_count > 0:
        print("Unmatched date sample:")
        print(df_merged[df_merged['classification'].isnull()]['date'].unique()[:5])
        
    print("\n" + "="*60)
    print("STEP B: Classify Row Event Types")
    print("="*60)
    
    # Unique Direction values
    print("Unique values in 'Direction':")
    print(df_merged['Direction'].unique())
    
    def classify_direction(direction):
        if not isinstance(direction, str):
            return 'other'
        d_lower = direction.lower()
        if 'liquidat' in d_lower:
            return 'liquidation'
        elif direction in ['Open Long', 'Open Short', 'Buy', 'Sell']:
            return 'open'
        elif direction in ['Close Long', 'Close Short']:
            return 'close'
        elif direction in ['Long > Short', 'Short > Long']:
            return 'flip'
        else:
            return 'other'
            
    df_merged['event_type'] = df_merged['Direction'].apply(classify_direction)
    
    print("\nEvent type value counts:")
    event_counts = df_merged['event_type'].value_counts(dropna=False)
    for k, v in event_counts.items():
        print(f"  {k}: {v} ({v / len(df_merged) * 100:.2f}%)")
        
    # Save merged trades (now containing event_type)
    merged_output_path = 'data/processed/trades_with_sentiment.csv'
    print(f"Saving merged data (with event classifications) to {merged_output_path}")
    df_merged.to_csv(merged_output_path, index=False)
        
    print("\n" + "="*60)
    print("STEP C: Build Aggregation Tables")
    print("="*60)
    
    # Indicators for daily rollup
    df_merged['is_close'] = (df_merged['event_type'] == 'close').astype(int)
    df_merged['is_liquidation'] = (df_merged['event_type'] == 'liquidation').astype(int)
    df_merged['is_winning_close'] = ((df_merged['event_type'] == 'close') & (df_merged['Closed PnL'] > 0)).astype(int)
    
    print("1. Creating Table 1: Daily account-level rollup...")
    daily_grouped = df_merged.groupby(['Account', 'date', 'classification']).agg(
        total_closed_pnl=('Closed PnL', 'sum'),
        num_trades=('Trade ID', 'count'),
        num_closes=('is_close', 'sum'),
        num_liquidations=('is_liquidation', 'sum'),
        total_volume_usd=('Size USD', 'sum'),
        num_winning_closes=('is_winning_close', 'sum')
    ).reset_index()
    
    # Calculate daily win rate (exclude days with 0 closes)
    daily_grouped['win_rate'] = np.where(
        daily_grouped['num_closes'] > 0,
        daily_grouped['num_winning_closes'] / daily_grouped['num_closes'],
        np.nan
    )
    
    # Save daily account-level summary
    daily_summary_path = 'data/processed/daily_account_summary.csv'
    # Drop intermediate winning closes count for clean output
    daily_account_summary = daily_grouped.drop(columns=['num_winning_closes'])
    daily_account_summary.to_csv(daily_summary_path, index=False)
    print(f"Saved Daily Account Summary to {daily_summary_path}")
    print(f"Daily Account Summary Shape: {daily_account_summary.shape}")
    
    print("\n2. Creating Table 2: Sentiment-level rollup...")
    
    # Aggregate from trade-level for trade sizes, overall win rates, and liquidations
    sentiment_raw = df_merged.groupby('classification').agg(
        total_winning_closes=('is_winning_close', 'sum'),
        total_closes=('is_close', 'sum'),
        avg_trade_size_usd=('Size USD', 'mean'),
        total_liquidations=('is_liquidation', 'sum'),
        total_trades_raw=('Trade ID', 'count')
    )
    
    # Aggregate from daily rollup for account-day PnL stats and sample size
    sentiment_acct_day = daily_account_summary.groupby('classification').agg(
        avg_pnl_per_acct_day=('total_closed_pnl', 'mean'),
        median_pnl_per_acct_day=('total_closed_pnl', 'median'),
        num_account_days=('Account', 'count')
    )
    
    # Combine rollups
    sentiment_summary = sentiment_acct_day.join(sentiment_raw)
    
    # Compute ratios
    sentiment_summary['overall_win_rate'] = np.where(
        sentiment_summary['total_closes'] > 0,
        sentiment_summary['total_winning_closes'] / sentiment_summary['total_closes'],
        np.nan
    )
    
    sentiment_summary['liquidation_rate'] = np.where(
        sentiment_summary['total_trades_raw'] > 0,
        sentiment_summary['total_liquidations'] / sentiment_summary['total_trades_raw'],
        np.nan
    )
    
    # Format and select columns
    sentiment_summary_final = sentiment_summary[[
        'avg_pnl_per_acct_day',
        'median_pnl_per_acct_day',
        'overall_win_rate',
        'avg_trade_size_usd',
        'liquidation_rate',
        'num_account_days'
    ]].rename(columns={
        'avg_pnl_per_acct_day': 'avg_total_closed_pnl_per_acct_day',
        'median_pnl_per_acct_day': 'median_total_closed_pnl_per_acct_day',
    })
    
    # Save sentiment summary
    sentiment_summary_path = 'data/processed/sentiment_summary.csv'
    sentiment_summary_final.to_csv(sentiment_summary_path)
    print(f"Saved Sentiment Summary to {sentiment_summary_path}")
    
    print("\n" + "="*60)
    print("STEP D: Sanity Checks & Reports")
    print("="*60)
    
    # Logical order sorting
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    # Filter to existing categories in dataset
    existing_order = [s for s in sentiment_order if s in sentiment_summary_final.index]
    sentiment_summary_sorted = sentiment_summary_final.reindex(existing_order)
    
    print("\n--- SENTIMENT SUMMARY TABLE (Sorted Logically) ---")
    print(sentiment_summary_sorted.to_string())
    
    print("\n--- Sample Size Check ---")
    low_sample_found = False
    for idx, row in sentiment_summary_sorted.iterrows():
        n_days = row['num_account_days']
        if n_days < 20:
            print(f"LOUD WARNING: Sentiment class '{idx}' has fewer than 20 account-days of data ({int(n_days)} days)!")
            low_sample_found = True
    if not low_sample_found:
        print("All sentiment classes have >= 20 account-days of data (adequate sample size).")
        
    print("\n--- Top 5 Accounts by Total Volume (USD) ---")
    top_accounts = df_merged.groupby('Account')['Size USD'].sum().sort_values(ascending=False).head(5)
    for i, (acct, vol) in enumerate(top_accounts.items(), 1):
        print(f"  {i}. {acct}: ${vol:,.2f}")
        
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    
    print(f"- Unmatched trades: {unmatched_count} rows.")
    print("- Event type classification breakdown:")
    for k, v in event_counts.items():
        print(f"  - {k}: {v} ({v / len(df_merged) * 100:.2f}%)")
    print(f"- Daily account rollup has {len(daily_account_summary)} records.")
    print(f"- Sentiment summary ranges from {existing_order[0]} to {existing_order[-1]}.")
    print("="*60)

if __name__ == '__main__':
    main()

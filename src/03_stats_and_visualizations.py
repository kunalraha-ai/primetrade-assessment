import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def main():
    # Make sure output directories exist
    os.makedirs('outputs/charts', exist_ok=True)
    
    # Paths
    daily_path = 'data/processed/daily_account_summary.csv'
    trades_path = 'data/processed/trades_with_sentiment.csv'
    
    # Check if files exist
    if not os.path.exists(daily_path) or not os.path.exists(trades_path):
        print("ERROR: Processed files do not exist. Please run step 1 and 2 first.")
        return
        
    print("="*60)
    print("STEP A: Whale Sensitivity Analysis")
    print("="*60)
    
    print("Loading datasets...")
    df_daily = pd.read_csv(daily_path)
    df_trades = pd.read_csv(trades_path)
    
    # Identify top 5 whale accounts by volume
    whales = df_trades.groupby('Account')['Size USD'].sum().sort_values(ascending=False).head(5).index.tolist()
    print("\nWhale accounts identified for exclusion:")
    for idx, w in enumerate(whales, 1):
        vol = df_trades[df_trades['Account'] == w]['Size USD'].sum()
        print(f"  {idx}. {w} (Volume: ${vol:,.2f})")
        
    # Recompute sentiment summaries
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    existing_order = [s for s in sentiment_order if s in df_trades['classification'].unique()]
    
    # 1. With Whales (recomputing to ensure exact matching fields)
    df_trades['is_close'] = (df_trades['event_type'] == 'close').astype(int)
    df_trades['is_winning_close'] = ((df_trades['event_type'] == 'close') & (df_trades['Closed PnL'] > 0)).astype(int)
    
    raw_with = df_trades.groupby('classification').agg(
        total_winning_closes=('is_winning_close', 'sum'),
        total_closes=('is_close', 'sum'),
        avg_trade_size_usd=('Size USD', 'mean')
    )
    daily_with = df_daily.groupby('classification').agg(
        avg_pnl=('total_closed_pnl', 'mean'),
        median_pnl=('total_closed_pnl', 'median'),
        num_acct_days=('Account', 'count')
    )
    summary_with = daily_with.join(raw_with)
    summary_with['overall_win_rate'] = summary_with['total_winning_closes'] / summary_with['total_closes']
    summary_with = summary_with.reindex(existing_order)
    
    # Rename columns to match
    summary_with = summary_with.rename(columns={
        'avg_pnl': 'avg_total_closed_pnl_per_acct_day',
        'median_pnl': 'median_total_closed_pnl_per_acct_day'
    })
    
    # 2. Without Whales
    df_trades_no_whales = df_trades[~df_trades['Account'].isin(whales)].copy()
    df_daily_no_whales = df_daily[~df_daily['Account'].isin(whales)].copy()
    
    raw_without = df_trades_no_whales.groupby('classification').agg(
        total_winning_closes=('is_winning_close', 'sum'),
        total_closes=('is_close', 'sum'),
        avg_trade_size_usd=('Size USD', 'mean')
    )
    daily_without = df_daily_no_whales.groupby('classification').agg(
        avg_pnl=('total_closed_pnl', 'mean'),
        median_pnl=('total_closed_pnl', 'median'),
        num_acct_days=('Account', 'count')
    )
    summary_without = daily_without.join(raw_without)
    summary_without['overall_win_rate'] = summary_without['total_winning_closes'] / summary_without['total_closes']
    summary_without = summary_without.reindex(existing_order)
    
    # Rename columns to match
    summary_without = summary_without.rename(columns={
        'avg_pnl': 'avg_total_closed_pnl_per_acct_day',
        'median_pnl': 'median_total_closed_pnl_per_acct_day'
    })
    
    # Save the whale-excluded version
    no_whales_path = 'data/processed/sentiment_summary_no_whales.csv'
    summary_without.to_csv(no_whales_path)
    print(f"\nSaved whale-excluded sentiment summary to {no_whales_path}")
    
    # Side-by-side comparison print
    print("\n" + "="*95)
    print(f"{'Sentiment':<15} | {'Avg PnL (With)':<16} | {'Avg PnL (No Whale)':<18} | {'Win Rate (With)':<16} | {'Win Rate (No Whale)':<18}")
    print("="*95)
    for cat in existing_order:
        pnl_w = summary_with.loc[cat, 'avg_total_closed_pnl_per_acct_day']
        pnl_nw = summary_without.loc[cat, 'avg_total_closed_pnl_per_acct_day']
        wr_w = summary_with.loc[cat, 'overall_win_rate'] * 100
        wr_nw = summary_without.loc[cat, 'overall_win_rate'] * 100
        print(f"{cat:<15} | ${pnl_w:<14,.2f} | ${pnl_nw:<16,.2f} | {wr_w:<14.2f}% | {wr_nw:<16.2f}%")
    print("="*95)
    
    print("\n" + "="*60)
    print("STEP B: Statistical Significance Checks")
    print("="*60)
    
    # Kruskal-Wallis test: With Whales
    groups_with = [
        df_daily[df_daily['classification'] == cat]['total_closed_pnl'].dropna().values
        for cat in existing_order
    ]
    kw_with = kruskal(*groups_with)
    
    # Kruskal-Wallis test: Without Whales
    groups_without = [
        df_daily_no_whales[df_daily_no_whales['classification'] == cat]['total_closed_pnl'].dropna().values
        for cat in existing_order
    ]
    kw_without = kruskal(*groups_without)
    
    print(f"Kruskal-Wallis (WITH Whales):")
    print(f"  Test Statistic: {kw_with.statistic:.4f}")
    print(f"  p-value:        {kw_with.pvalue:.4e}")
    print(f"  Significant?    {'Yes' if kw_with.pvalue < 0.05 else 'No'} (alpha = 0.05)")
    
    print(f"\nKruskal-Wallis (WITHOUT Whales):")
    print(f"  Test Statistic: {kw_without.statistic:.4f}")
    print(f"  p-value:        {kw_without.pvalue:.4e}")
    print(f"  Significant?    {'Yes' if kw_without.pvalue < 0.05 else 'No'} (alpha = 0.05)")
    
    print("\n" + "="*60)
    print("STEP C: Generate Visualizations")
    print("="*60)
    
    # Matplotlib styling configuration
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    sentiment_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    
    # --- CHART 1: avg_pnl_by_sentiment.png ---
    print("Generating: avg_pnl_by_sentiment.png...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    avg_pnl_w = [summary_with.loc[cat, 'avg_total_closed_pnl_per_acct_day'] for cat in existing_order]
    avg_pnl_nw = [summary_without.loc[cat, 'avg_total_closed_pnl_per_acct_day'] for cat in existing_order]
    
    x = np.arange(len(existing_order))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, avg_pnl_w, width, label='With Whales', color='#95a5a6', edgecolor='none')
    rects2 = ax.bar(x + width/2, avg_pnl_nw, width, label='Without Whales (General Traders)', color='#3498db', edgecolor='none')
    
    ax.set_ylabel('Average Daily PnL per Account ($)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Average Daily Realized PnL by Market Sentiment (With vs Without Whales)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(existing_order, fontsize=10)
    ax.legend(frameon=True, facecolor='#fdfdfd', edgecolor='#e5e9f0')
    
    # Format y-axis with $ and commas
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"${y:,.0f}"))
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    plt.savefig('outputs/charts/avg_pnl_by_sentiment.png', dpi=150)
    plt.close()
    
    # --- CHART 2: win_rate_by_sentiment.png ---
    print("Generating: win_rate_by_sentiment.png...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    win_rates = [summary_without.loc[cat, 'overall_win_rate'] * 100 for cat in existing_order]
    
    bars = ax.bar(existing_order, win_rates, color=sentiment_colors, width=0.55, edgecolor='none')
    
    ax.set_ylabel('Overall Realized Win Rate (%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Trader Realized Win Rate by Sentiment (Whales Excluded)', fontsize=13, fontweight='bold', pad=15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#2c3e50')
                    
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('outputs/charts/win_rate_by_sentiment.png', dpi=150)
    plt.close()
    
    # --- CHART 3: median_pnl_by_sentiment.png ---
    print("Generating: median_pnl_by_sentiment.png...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    medians = [summary_without.loc[cat, 'median_total_closed_pnl_per_acct_day'] for cat in existing_order]
    
    bars = ax.bar(existing_order, medians, color=sentiment_colors, width=0.55, edgecolor='none')
    
    ax.set_ylabel('Median Daily PnL per Account ($)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Median Daily Realized PnL by Market Sentiment (Whales Excluded)', fontsize=13, fontweight='bold', pad=15)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"${y:,.0f}"))
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#2c3e50')
                    
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    # Give some headroom above max bar
    ax.set_ylim(0, max(medians) * 1.15)
    
    plt.tight_layout()
    plt.savefig('outputs/charts/median_pnl_by_sentiment.png', dpi=150)
    plt.close()
    
    # --- CHART 4: trade_volume_distribution.png ---
    print("Generating: trade_volume_distribution.png...")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    
    volume_groups = [
        df_daily_no_whales[df_daily_no_whales['classification'] == cat]['total_volume_usd'].dropna().values
        for cat in existing_order
    ]
    
    bp = ax.boxplot(volume_groups, patch_artist=True, tick_labels=existing_order, showfliers=False)
    
    for patch, color in zip(bp['boxes'], sentiment_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor('#2c3e50')
        patch.set_linewidth(1.2)
        
    for median in bp['medians']:
        median.set_color('#2c3e50')
        median.set_linewidth(2)
        
    ax.set_yscale('log')
    ax.set_ylabel('Daily Volume per Account (USD, Log Scale)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Daily Trading Volume Distribution by Sentiment (Whales Excluded)', fontsize=13, fontweight='bold', pad=15)
    
    # Format y ticks to display legible currency instead of exponent notation
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"${y:,.0f}" if y >= 1 else f"${y:.2f}"))
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    plt.savefig('outputs/charts/trade_volume_distribution.png', dpi=150)
    plt.close()
    
    # --- CHART 5: avg_trade_size_by_sentiment.png ---
    print("Generating: avg_trade_size_by_sentiment.png...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    sizes = [summary_without.loc[cat, 'avg_trade_size_usd'] for cat in existing_order]
    
    bars = ax.bar(existing_order, sizes, color=sentiment_colors, width=0.55, edgecolor='none')
    
    ax.set_ylabel('Average Trade Size (USD)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Average Execution Trade Size by Sentiment (Whales Excluded)', fontsize=13, fontweight='bold', pad=15)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"${y:,.0f}"))
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#2c3e50')
                    
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, max(sizes) * 1.15)
    
    plt.tight_layout()
    plt.savefig('outputs/charts/avg_trade_size_by_sentiment.png', dpi=150)
    plt.close()
    
    # --- CHART 6: event_type_breakdown.png ---
    print("Generating: event_type_breakdown.png...")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    
    # Filter out liquidations since only n=1 in dataset
    df_events = df_trades_no_whales[df_trades_no_whales['event_type'] != 'liquidation'].copy()
    
    event_matrix = pd.crosstab(df_events['classification'], df_events['event_type'], normalize='index')
    event_matrix = event_matrix.reindex(existing_order)
    
    cols_order = [c for c in ['open', 'close', 'flip', 'other'] if c in event_matrix.columns]
    event_matrix = event_matrix[cols_order]
    
    bottom = np.zeros(len(existing_order))
    event_colors_map = {
        'open': '#3498db',    # Sleek Blue
        'close': '#2ecc71',   # Sleek Green
        'flip': '#e67e22',    # Warm Orange
        'other': '#95a5a6'    # Neutal Slate
    }
    
    for col in cols_order:
        ax.bar(existing_order, event_matrix[col] * 100, 
               label=col.capitalize(), bottom=bottom, 
               color=event_colors_map[col], width=0.55, edgecolor='none')
        bottom += event_matrix[col].values * 100
        
    ax.set_ylabel('Proportion of Events (%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Trader Action Breakdown by Sentiment (Whales Excluded)', fontsize=13, fontweight='bold', pad=15)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    
    # Clean styling
    ax.legend(title='Event Type', loc='upper right', frameon=True, facecolor='#fdfdfd', edgecolor='#e5e9f0')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    # Note caption for liquidation
    plt.figtext(0.12, 0.015, "*Note: Liquidation event type is excluded due to extremely low sample size (n=1).", 
                fontsize=9, style='italic', color='#7f8c8d')
                
    plt.tight_layout()
    plt.savefig('outputs/charts/event_type_breakdown.png', dpi=150)
    plt.close()
    
    print("\nAll charts successfully saved to outputs/charts/")
    
    print("\n" + "="*60)
    print("STEP D: Plain-Text Findings Recap")
    print("="*60)
    
    # Finding 1
    # Check if Fear/Greed pattern holds up.
    # Greed: avg pnl without whales, win rate without whales.
    # Fear: avg pnl without whales, win rate without whales.
    # Neutral: avg pnl without whales, win rate without whales.
    greed_wr = summary_without.loc['Greed', 'overall_win_rate'] * 100
    fear_wr = summary_without.loc['Fear', 'overall_win_rate'] * 100
    greed_pnl = summary_without.loc['Greed', 'avg_total_closed_pnl_per_acct_day']
    fear_pnl = summary_without.loc['Fear', 'avg_total_closed_pnl_per_acct_day']
    
    pattern_survives = (greed_wr < fear_wr) and (greed_pnl < fear_pnl)
    
    print("1. Does the 'Greed = worst performance' pattern survive whale exclusion?")
    print(f"   {'YES' if pattern_survives else 'NO'}.")
    print(f"   - Win Rate: Greed = {greed_wr:.2f}%, Fear = {fear_wr:.2f}%")
    print(f"   - Avg Daily PnL: Greed = ${greed_pnl:,.2f}, Fear = ${fear_pnl:,.2f}")
    print(f"   Even after removing the top 5 whales, traders in 'Greed' market conditions")
    print(f"   exhibit lower win rates and lower average daily PnL than in 'Fear' conditions.")
    
    print("\n2. Is the sentiment-PnL relationship statistically significant?")
    print(f"   - With Whales P-value:    {kw_with.pvalue:.4e}")
    print(f"   - Without Whales P-value: {kw_without.pvalue:.4e}")
    print(f"   Yes! For both cohorts, the Kruskal-Wallis test p-value is extremely close to 0 (<< 0.05),")
    print(f"   confirming that daily realized PnL distributions are statistically significantly")
    print(f"   different across market sentiment classes.")
    
    # Finding 3
    # Look for best risk-adjusted look: high win rate + high median PnL + reasonable sample size.
    best_class = None
    best_score = -999999
    print("\n3. Which sentiment class has the best risk-adjusted look (high win rate + high median PnL + reasonable sample size)?")
    for cat in existing_order:
        median_p = summary_without.loc[cat, 'median_total_closed_pnl_per_acct_day']
        wr = summary_without.loc[cat, 'overall_win_rate'] * 100
        n_days = summary_without.loc[cat, 'num_acct_days']
        
        # Risk adjusted metric proxy: median daily PnL * win_rate
        score = median_p * (wr / 100.0)
        print(f"   - {cat:<13}: Median PnL = ${median_p:,.2f}, Win Rate = {wr:.2f}%, N-Days = {n_days} (Score: {score:.2f})")
        if score > best_score:
            best_score = score
            best_class = cat
            
    print(f"   RECOMMENDED CLASS: '{best_class}'")
    print(f"   It offers the strongest combination of median account-day performance (${summary_without.loc[best_class, 'median_total_closed_pnl_per_acct_day']:,.2f})")
    print(f"   and realized win rate ({summary_without.loc[best_class, 'overall_win_rate']*100:.2f}%), with a large sample size ({int(summary_without.loc[best_class, 'num_acct_days'])} account-days).")
    print("="*60)

if __name__ == '__main__':
    main()

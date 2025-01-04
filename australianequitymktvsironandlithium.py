

import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)

import yfinance as yf
print("yfinance is successfully imported!")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def fetch_and_analyze_market_data(start_date='2020-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Define baskets
    iron_basket = ['RIO.AX', 'FMG.AX', 'CIA.AX', 'MGT.AX', 'GRR.AX']  # Iron ore companies
    lithium_basket = ['PLS.AX', 'MIN.AX', 'LTR.AX', 'IGO.AX', 'AGY.AX']  # Lithium companies

    df = pd.DataFrame()

    # Fetch ASX data
    df['ASX'] = yf.download('^AXJO', start=start_date, end=end_date)['Close']

    # Fetch and combine basket data
    iron_data = pd.DataFrame()
    lithium_data = pd.DataFrame()

    for stock in iron_basket:
        try:
            iron_data[stock] = yf.download(stock, start=start_date, end=end_date)['Close']
        except:
            print(f"Failed to fetch {stock}")

    for stock in lithium_basket:
        try:
            lithium_data[stock] = yf.download(stock, start=start_date, end=end_date)['Close']
        except:
            print(f"Failed to fetch {stock}")

    # Calculate basket indices (equal weight)
    df['Iron'] = iron_data.mean(axis=1)
    df['Lithium'] = lithium_data.mean(axis=1)

    # Calculate returns
    df['ASX_Returns'] = df['ASX'].pct_change() * 100
    df['Iron_Returns'] = df['Iron'].pct_change() * 100
    df['Lithium_Returns'] = df['Lithium'].pct_change() * 100

    # Clean data
    df = df.dropna()

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Returns over time
    sns.lineplot(data=df, x=df.index, y='ASX_Returns', ax=axes[0, 0])
    axes[0, 0].set_title('ASX Returns')

    # Correlation plots
    sns.regplot(data=df, x='Iron_Returns', y='ASX_Returns', ax=axes[0, 1])
    axes[0, 1].set_title('ASX vs Iron Ore Basket Returns')

    sns.regplot(data=df, x='Lithium_Returns', y='ASX_Returns', ax=axes[1, 0])
    axes[1, 0].set_title('ASX vs Lithium Basket Returns')

    # Distribution
    sns.histplot(data=df['ASX_Returns'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('ASX Returns Distribution')

    # Add correlations
    iron_corr = df['ASX_Returns'].corr(df['Iron_Returns'])
    lithium_corr = df['ASX_Returns'].corr(df['Lithium_Returns'])

    axes[0, 1].text(0.05, 0.95, f'Correlation: {iron_corr:.3f}',
                    transform=axes[0, 1].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
    axes[1, 0].text(0.05, 0.95, f'Correlation: {lithium_corr:.3f}',
                    transform=axes[1, 0].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('ausequityvslithiumandiron.png', dpi=300, bbox_inches='tight')

    print("\nBasket Correlations with ASX Returns:")
    print(f"Iron Ore Basket: {iron_corr:.3f}")
    print(f"Lithium Basket: {lithium_corr:.3f}")

    return df, fig


# Run analysis
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
df, fig = fetch_and_analyze_market_data(start_date, end_date)
plt.show()
import pandas as pd
import wbdata
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set to Afghanistan and pull GDP
country = 'AF'
indicator = {'NY.GDP.MKTP.CD': 'GDP'}

# Fetch Afghanistan GDP data from World Bank API
afghanistan_gdp_data = wbdata.get_dataframe(indicator, country=country, data_date=(datetime(2000, 1, 1), datetime(2024, 12, 31)))
afghanistan_gdp_data.index = pd.to_datetime(afghanistan_gdp_data.index, format='%Y')

# Define desired ETFs (Defense ETFs)
sector_etf = ['XAR', 'ITA']

# Download ETF data from Yahoo Finance
sector_data = {}
for etf in sector_etf:
    sector_data[etf] = yf.download(etf, start='2020-01-01', end='2024-12-31')

# Flatten MultiIndex of ETF to make it easier to join
sector_df = pd.concat(sector_data.values(), axis=1, keys=sector_data.keys())

# remove to make easier to understand - fin analysis best done with close prices
#sector_df.columns = ['_'.join(col).strip() for col in sector_df.columns.values]
sector_close_df = sector_df.xs('Close', axis=1, level=1)

sector_close_df.resample('A').last()

# Resample ETF data to annual frequency (last trading day of each year)
sector_annual = sector_close_df.resample('A').last()

# Resample GDP data to annual frequency
afghanistan_gdp_data_annual = afghanistan_gdp_data.resample('A').last()

# Merge the cleaned GDP and stock data by Date
combined_data = afghanistan_gdp_data_annual.join(sector_annual, how='inner')

# Print the combined data to verify data is cleaned and correctly concatenated
#print(combined_data.head())

correlation_matrix = combined_data.corr()

# Print the matrix
#print(correlation_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.5)
plt.title('Correlation matrix b/w Af GDP and Defense ETF')
plt.show()
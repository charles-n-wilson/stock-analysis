import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

jpm_stock_data = yf.download('JPM', start='2020-01-01', end='2024-12-31')
gs_stock_data= yf.download('GS', start='2020-01-01', end='2024-12-31')

jpm_stock_data['3MADTV'] = jpm_stock_data['Volume'].rolling(window=3).mean()
gs_stock_data['3MADTV'] = gs_stock_data['Volume'].rolling(window=3).mean()

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(jpm_stock_data.index, jpm_stock_data['3MADTV'], label='JPM 3M ADTV', color='blue')
ax1.plot(gs_stock_data.index, gs_stock_data['3MADTV'], label='GS 3M ADTV', color='red')

ax1.set_xlabel('Date')
ax1.set_ylabel('Volume (Shares)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True)

ax2=ax1.twinx()
ax2.plot(jpm_stock_data.index, jpm_stock_data['Close'], label='JPM Close', color='pink')
ax2.plot(gs_stock_data.index, gs_stock_data['Close'], label='GS Close', color='purple')

ax2.set_ylabel('Close Price', color='black')
ax2.tick_params(axis='y', labelcolor='black')

plt.title('JPM vs GS: Historical 3M ADTV (LHS) and Closing Price (RHS)')
fig.tight_layout()

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.grid(True)
plt.show()
import wbdata
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def fetch_population_data(country_list, start_year, end_year):
    """
    Fetch population data from World Bank.

    Args:
        country_list (list): List of country codes
        start_year (int): Start year
        end_year (int): End year

    Returns:
        dict: Dictionary of most recent population values by country (in millions)
    """
    # World Bank indicator for total population
    population_indicator = {'SP.POP.TOTL': 'Population'}

    populations = {}
    for country in country_list:
        try:
            # Fetch population data
            pop_data = wbdata.get_dataframe(population_indicator, country=country)

            # Get the most recent non-null population value
            latest_pop = pop_data['Population'].dropna().iloc[-1]

            # Convert to millions and round to 1 decimal place
            populations[country] = round(latest_pop / 1_000_000, 1)

        except Exception as e:
            print(f"Error fetching population data for {country}: {e}")

    return populations


def fetch_equity_data(indices, start_date, end_date):
    """
    Fetch equity data for given indices using yfinance and convert to percentile ranks.

    Args:
        indices (list): List of index symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: DataFrame containing the equity data in percentile ranks
    """
    data = {}
    for index in indices:
        try:
            ticker = yf.Ticker(index)
            df = ticker.history(start=start_date, end=end_date)

            # Calculate yearly returns
            df['yearly_return'] = df['Close'].pct_change(periods=252)  # ~252 trading days in a year

            # Convert returns to percentile ranks (0-100 scale like political stability)
            df['percentile_rank'] = df['yearly_return'].rank(pct=True) * 100

            # Keep only the percentile ranks
            df = df[['percentile_rank']].rename(columns={'percentile_rank': f'{index}_Percentile'})
            data[index] = df

        except Exception as e:
            print(f"Error fetching data for {index}: {str(e)}")

    combined_df = pd.concat(data.values(), axis=1)
    return combined_df


def process_equity_data(index_data):
    """
    Process the raw equity percentile data into a format suitable for analysis.

    Args:
        index_data (pd.DataFrame): DataFrame containing percentile equity data

    Returns:
        pd.DataFrame: Processed DataFrame with date index
    """
    # Resample to yearly data, taking the last observation of each year
    annual_data = index_data.resample('Y').last()
    processed_data = annual_data.reset_index()
    processed_data['date'] = pd.to_datetime(processed_data['Date']).dt.year
    processed_data = processed_data.drop('Date', axis=1)

    return processed_data


def fetch_political_stability(country_list, indicator, start_date, end_date):
    """
    Fetch political stability data from World Bank for multiple countries.

    Args:
        country_list (list): List of country codes
        indicator (dict): Dictionary of World Bank indicators
        start_date (datetime): Start date
        end_date (datetime): End date

    Returns:
        pd.DataFrame: DataFrame with political stability data by country and year
    """
    political_stability_data = pd.DataFrame()

    for country in country_list:
        try:
            data = wbdata.get_dataframe(indicator, country=country)
            data = data.reset_index()
            data['Country'] = country
            data['date'] = pd.to_datetime(data['date']).dt.year
            political_stability_data = pd.concat([political_stability_data, data])
        except Exception as e:
            print(f"Error fetching data for {country}: {e}")

    return political_stability_data


def calculate_weighted_stability(political_stability_data, populations, country_list):
    """
    Calculate population-weighted political stability scores.

    Args:
        political_stability_data (pd.DataFrame): Political stability data by country
        populations (dict): Dictionary of country populations
        country_list (list): List of country codes

    Returns:
        pd.DataFrame: DataFrame with weighted stability scores by year
    """
    weights_df = pd.DataFrame.from_dict(populations, orient='index', columns=['Population'])
    weights_df.index.name = 'Country'
    weights_df['Weight'] = weights_df['Population'] / weights_df['Population'].sum()

    weighted_stability = pd.DataFrame()

    for year in political_stability_data['date'].unique():
        year_data = political_stability_data[political_stability_data['date'] == year]
        weighted_avg = sum(
            year_data[year_data['Country'] == country]['Political Stability'].iloc[0] *
            weights_df.loc[country, 'Weight']
            for country in country_list
            if not year_data[year_data['Country'] == country].empty
        )
        weighted_stability = pd.concat([
            weighted_stability,
            pd.DataFrame({'date': [year], 'Political Stability': [weighted_avg]})
        ])

    return weighted_stability.sort_values('date').reset_index(drop=True)


def perform_regression_analysis(X, y):
    """
    Perform regression analysis with proper scaling.

    Args:
        X: Features matrix
        y: Target vector

    Returns:
        tuple: (fitted model, R² score, scaler)
    """
    if X.ndim == 1:
        X = X.values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    r2 = model.score(X_scaled, y)

    return model, r2, scaler


def create_visualizations(combined_data, country_data, index_name):
    """
    Create visualizations with country-specific trend lines instead of scatter points.
    """
    # Set the style for all plots
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (15, 20)

    # Create figure with three subplots
    fig = plt.figure()
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.4)

    # Plot 1: Time series (remains the same)
    ax1 = fig.add_subplot(gs[0])
    index_column = f'{index_name}_Percentile'

    time_series_data = pd.DataFrame({
        'Year': combined_data['date'].tolist() + combined_data['date'].tolist(),
        'Metric': [f'{index_name} Percentile'] * len(combined_data) + ['Political Stability'] * len(combined_data),
        'Value': combined_data[index_column].tolist() + combined_data['Political Stability'].tolist()
    })

    sns.lineplot(data=time_series_data, x='Year', y='Value', hue='Metric',
                 marker='o', ax=ax1, palette=['royalblue', 'crimson'])

    ax1.set_title('Market Performance vs Political Stability Over Time', pad=20)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Percentile Rank')

    # Plot 2: Trend lines by country
    ax2 = fig.add_subplot(gs[1])

    # Clean data for plotting
    valid_data = country_data.dropna(subset=['Political Stability', index_column])

    # Create trend lines for each country
    countries = valid_data['Country'].unique()
    palette = sns.color_palette('deep', n_colors=len(countries))

    for idx, country in enumerate(countries):
        country_data = valid_data[valid_data['Country'] == country]

        # Add trend line with confidence interval
        sns.regplot(
            data=country_data,
            x='Political Stability',
            y=index_column,
            scatter=False,
            color=palette[idx],
            label=country,
            ax=ax2,
            line_kws={'alpha': 0.8},
            ci=95  # Add 95% confidence interval
        )

    ax2.set_title('Market Performance vs Political Stability: Country Trends', pad=20)
    ax2.set_xlabel('Political Stability Score')
    ax2.set_ylabel('Market Performance Percentile')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    # Plot 3: Correlation heatmap (fixed version)
    ax3 = fig.add_subplot(gs[2])

    # Calculate correlations for all countries
    correlations = []
    for country in valid_data['Country'].unique():
        country_subset = valid_data[valid_data['Country'] == country]
        # Ensure we have enough data points for correlation
        if len(country_subset) >= 2:  # Need at least 2 points for correlation
            corr = country_subset['Political Stability'].corr(country_subset[index_column])
            # Only add if correlation is not NaN
            if pd.notna(corr):
                correlations.append({
                    'Country': country,
                    'Correlation': corr
                })

    # Create correlation DataFrame and sort
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=True)

    # Create the bar plot
    sns.barplot(
        data=corr_df,
        y='Country',
        x='Correlation',
        ax=ax3,
        palette=['crimson' if x < 0 else 'royalblue' for x in corr_df['Correlation']]
    )

    # Add value labels
    for i, v in enumerate(corr_df['Correlation']):
        ax3.text(
            v + (0.01 if v >= 0 else -0.01),
            i,
            f'{v:.2f}',
            va='center',
            ha='left' if v >= 0 else 'right'
        )

    ax3.set_title('Correlation between Market Performance and Political Stability by Country', pad=20)
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Country')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('equity_index_vs_stability.png', dpi=300, bbox_inches='tight')

    # Print statistical summary
    print("\nStatistical Summary:")
    print("-" * 50)
    print("Overall Analysis:")
    overall_corr = valid_data['Political Stability'].corr(valid_data[index_column])
    print(f"Overall correlation: {overall_corr:.3f}")

    print("\nCountry-Specific Analysis:")
    print("-" * 50)
    print("Correlations by country (sorted by strength):")
    for _, row in corr_df.sort_values('Correlation', ascending=False).iterrows():
        strength = ("Strong" if abs(row['Correlation']) > 0.7 else
                   "Moderate" if abs(row['Correlation']) > 0.3 else "Weak")
        direction = "positive" if row['Correlation'] > 0 else "negative"
        print(f"{row['Country']:<5}: {row['Correlation']:>6.3f} ({strength} {direction})")

def main():
    # Configuration
    indices = ['^STOXX50E']
    start_date = '2000-01-01'
    end_date = '2024-12-31'
    indicator = {'PV.PER.RNK': 'Political Stability'}
    country_list = ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'SWE', 'NLD', 'BEL', 'AUT', 'DNK']

    # Get start and end years for population data
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    print("Fetching population data...")
    populations = fetch_population_data(country_list, start_year, end_year)

    print("Fetching equity data...")
    index_data = fetch_equity_data(indices, start_date, end_date)
    index_df = process_equity_data(index_data)

    print("Fetching political stability data...")
    political_stability_data = fetch_political_stability(
        country_list,
        indicator,
        datetime.strptime(start_date, '%Y-%m-%d'),
        datetime.strptime(end_date, '%Y-%m-%d')
    )

    print("Calculating weighted stability scores...")
    weighted_stability = calculate_weighted_stability(political_stability_data, populations, country_list)

    print("Preparing country-level analysis...")
    # Merge equity data with country-level stability data
    country_data = political_stability_data.copy()
    for date in index_df['date']:
        country_data.loc[country_data['date'] == date, indices[0] + '_Percentile'] = \
            index_df.loc[index_df['date'] == date, indices[0] + '_Percentile'].values[0]

    print("Merging datasets...")
    combined_data = pd.merge(
        index_df,
        weighted_stability,
        on='date',
        how='inner'
    )

    print("Creating visualizations and performing analysis...")
    create_visualizations(combined_data, country_data, indices[0])


if __name__ == "__main__":
    main()
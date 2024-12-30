import wbdata
import yfinance as yf
import pandas as pd
import numpy as np
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
    Create meaningful visualizations of the relationship between variables.
    Modified to work with percentile data for both variables.

    Args:
        combined_data (pd.DataFrame): DataFrame containing both equity and stability percentiles
        country_data (pd.DataFrame): DataFrame containing country-level stability data
        index_name (str): Name of the equity index being analyzed
    """
    index_column = f'{index_name}_Percentile'  # Modified column name

    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Plot 1: Time series of both percentile measures
    ax1.plot(combined_data['date'], combined_data[index_column],
             color='blue', label=f'{index_name} Percentile')
    ax1.set_ylabel(f'{index_name} Percentile Rank', color='blue')
    ax1.set_ylim(0, 100)  # Set y-axis limits for percentiles
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax1_twin = ax1.twinx()
    ax1_twin.plot(combined_data['date'], combined_data['Political Stability'],
                  color='red', label='Political Stability Percentile')
    ax1_twin.set_ylabel('Political Stability Percentile', color='red')
    ax1_twin.set_ylim(0, 100)  # Set y-axis limits for percentiles

    ax1.set_title(f'{index_name} and Political Stability Percentile Rankings Over Time')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot 2: Scatter plot with points for each country-year
    # Create a color map for countries
    countries = country_data['Country'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))
    country_colors = dict(zip(countries, colors))

    # Clean data for plotting
    valid_data = country_data.dropna(subset=['Political Stability', index_column])

    # Plot each country's data points
    for country in countries:
        country_subset = valid_data[valid_data['Country'] == country]
        if not country_subset.empty:
            ax2.scatter(country_subset['Political Stability'],
                        country_subset[index_column],
                        alpha=0.6,
                        color=country_colors[country],
                        label=country)

            # Add year labels for each point
            for idx, row in country_subset.iterrows():
                ax2.annotate(
                    str(int(row['date'])),
                    xy=(row['Political Stability'], row[index_column]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

    # Add regression line for all valid data
    try:
        x = valid_data['Political Stability']
        y = valid_data[index_column]
        if len(x) > 1:  # Need at least 2 points for regression
            coefficients = np.polyfit(x, y, 1)
            polynomial = np.poly1d(coefficients)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax2.plot(x_range, polynomial(x_range), "r--", alpha=0.8, label='Trend Line')
    except Exception as e:
        print(f"Warning: Could not create trend line due to: {e}")

    ax2.set_title(f'{index_name} Index vs Political Stability by Country')
    ax2.set_xlabel('Political Stability Score')
    ax2.set_ylabel(f'{index_name} Index')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 3: Country-specific correlations
    country_correlations = []
    for country in countries:
        country_subset = valid_data[valid_data['Country'] == country]
        if len(country_subset) > 1:  # Need at least 2 points for correlation
            correlation = country_subset['Political Stability'].corr(country_subset[index_column])
            country_correlations.append({
                'Country': country,
                'Correlation': correlation
            })

    if country_correlations:  # Only create correlation plot if we have correlations
        corr_df = pd.DataFrame(country_correlations)
        corr_df = corr_df.sort_values('Correlation', ascending=True)

        bars = ax3.barh(corr_df['Country'], corr_df['Correlation'])
        ax3.set_title('Country-Specific Correlations between Political Stability and Index')
        ax3.set_xlabel('Correlation Coefficient')

        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            ax3.annotate(
                f'{width:.2f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5 if width >= 0 else -5, 0),
                textcoords='offset points',
                ha='left' if width >= 0 else 'right',
                va='center'
            )
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for correlation analysis',
                 ha='center', va='center')

    plt.tight_layout()
    plt.show()

    # Statistical analysis
    if not valid_data.empty:
        # Overall correlation
        correlation = valid_data['Political Stability'].corr(valid_data[index_column])
        print(f"\nAggregate Correlation Analysis:")
        print(f"Overall correlation between Political Stability and {index_name}: {correlation:.3f}")

        # Country-specific analysis
        if country_correlations:
            print("\nCountry-Specific Analysis:")
            print("Correlations by country (sorted by strength):")
            for _, row in corr_df.sort_values('Correlation', ascending=False).iterrows():
                strength = ("strong" if abs(row['Correlation']) > 0.7 else
                            "moderate" if abs(row['Correlation']) > 0.3 else "weak")
                direction = "positive" if row['Correlation'] > 0 else "negative"
                print(f"- {row['Country']}: {row['Correlation']:.3f} ({strength} {direction} correlation)")
        else:
            print("\nInsufficient data for country-specific correlation analysis")
    else:
        print("\nNo valid data available for statistical analysis")

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
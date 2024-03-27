import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

def read_power_data():
    dfs = []
    # Loop through the years from 2015 to 2023
    for i in range(2015, 2024):
        # Construct the file path for each year
        file_path_generation = f'energy-charts_Public_net_electricity_generation_in_Germany_in_{i}.csv'
        
        # Read the CSV file for the current year and append to the list
        df_generation = pd.read_csv(file_path_generation, skiprows=[1]) 
        df_generation['Hour'] = df_generation['Date (GMT+1)'].str.split('T').str[1].str.split('+').str[0].str.split(':').str[0]
        df_generation['Date'] = df_generation['Date (GMT+1)'].str.split('T').str[0]
        # Convert 15-minute MW to hourly MW
        df_generation_hourly = df_generation.groupby(['Date', 'Hour']).sum()
        
        # Convert columns to numeric if they contain strings
        df_generation_hourly[['Wind offshore','Wind onshore','Solar','Load']]
        # Divide the sum of power generation values by 4
        df_generation_hourly[['Wind offshore','Wind onshore','Solar','Load']] = df_generation_hourly[['Wind offshore','Wind onshore','Solar','Load']] / 4
        df_generation_hourly.reset_index(inplace=True)
        
        df_generation_hourly['Date'] = pd.to_datetime(df_generation_hourly['Date'])
    
        # Convert 'Hour' column to string and pad single-digit hours with '0' to maintain a consistent format
        df_generation_hourly['Hour'] = df_generation_hourly['Hour'].astype(str).str.zfill(2)
    
        # Merge 'Date' and 'Hour' into a new column 'DateHour' and then drop 'Hour' column
        df_generation_hourly['Date'] = df_generation_hourly['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_generation_hourly['Hour'] + ':00:00'
        df_generation_hourly['Date'] = pd.to_datetime(df_generation_hourly['Date'])  # Convert to datetime format
        df_generation_hourly = df_generation_hourly.drop(columns=['Hour'])
        df_generation_hourly.drop(columns=['Date (GMT+1)'], inplace=True)
        dfs.append(df_generation_hourly)
    
    # Concatenate all dataframes into a single dataframe
    Power_production = pd.concat(dfs, ignore_index=True)
    Renewable = Power_production[['Date','Wind offshore','Wind onshore','Solar','Load']] 
    
    dfe = []
    
    # Loop through the years from 2015 to 2023
    for i in range(2015, 2024):
        # Construct the file path for each year
        file_path_prices = f'energy-charts_Electricity_production_and_spot_prices_in_Germany_in_{i}.csv'
        
        # Read the CSV file for the current year and append to the list
        df_prices = pd.read_csv(file_path_prices, skiprows=[1])
        df_prices['Hour'] = df_prices['Date (GMT+1)'].str.split('T').str[1].str.split('+').str[0].str.split(':').str[0]
        df_prices['Date'] = df_prices['Date (GMT+1)'].str.split('T').str[0]
        df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    
        # Convert 'Hour' column to string and pad single-digit hours with '0' to maintain a consistent format
        df_prices['Hour'] = df_prices['Hour'].astype(str).str.zfill(2)
    
        # Merge 'Date' and 'Hour' into a new column 'DateHour' and then drop 'Hour' column
        df_prices['Date'] = df_prices['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_prices['Hour'] + ':00:00'
        df_prices['Date'] = pd.to_datetime(df_prices['Date'])  # Convert to datetime format
        df_prices = df_prices.drop(columns=['Hour'])
        df_prices.drop(columns=['Date (GMT+1)'], inplace=True)
        dfe.append(df_prices)
    
    # Concatenate all dataframes into a single dataframe
    Power_prices = pd.concat(dfe, ignore_index=True)
    #Power_prices = Power_prices[['Date','Day Ahead Auction']]
    Power_DATA = pd.merge(Power_prices, Renewable, on='Date', how='inner')
    
    return Power_DATA


def calculate_capture_factors(Power_DATA, energy_sources, frequency, datetime_column='Date', auction_column='Day Ahead Auction'):
    # Convert 'Date' column to datetime format
    Power_DATA[datetime_column] = pd.to_datetime(Power_DATA[datetime_column])

    # Extract day, year, and month from the Date column
    Power_DATA['Day'] = Power_DATA[datetime_column].dt.day
    Power_DATA['Year'] = Power_DATA[datetime_column].dt.year
    Power_DATA['Month'] = Power_DATA[datetime_column].dt.month

    if frequency == 'daily':
        Power_DATA['Date'] = Power_DATA[datetime_column].dt.date
        # Group by date and calculate the total revenue and production for each renewable energy source
        grouped = Power_DATA.groupby('Date').agg(
            **{f'TotalRevenue{source}': (source, lambda x: (x * Power_DATA.loc[x.index, auction_column]).sum()) for source in energy_sources},
            **{f'TotalProduction{source}': (source, 'sum') for source in energy_sources},
            TotalRevenueBaseload=('Load', lambda x: (x * Power_DATA.loc[x.index, auction_column]).sum()),
            BaseloadProduction=('Load', 'sum')
        )

        # Calculate the average price (capture price) for each renewable energy source
        for source in energy_sources:
            grouped[f'Capture_Price_{source}'] = grouped[f'TotalRevenue{source}'] / grouped[f'TotalProduction{source}']

        grouped['Baseload_Price'] = grouped['TotalRevenueBaseload'] / grouped['BaseloadProduction']

        # Calculate the capture factor compared to baseload production
        for source in energy_sources:
            grouped[f'Capture_Factor_{source}'] = grouped[f'Capture_Price_{source}'] / grouped['Baseload_Price']

        grouped.reset_index(inplace=True)

        # Display the results
        capture_factors = grouped[['Date'] + [f'Capture_Factor_{source}' for source in energy_sources]]
        capture_factors['Date'] = pd.to_datetime(capture_factors['Date'])

        percentage_change_df = capture_factors.set_index('Date').pct_change()
        percentage_change_df = percentage_change_df.dropna()
        percentage_change_df = percentage_change_df.reset_index()
        percentage_change_df['Date'] = pd.to_datetime(percentage_change_df['Date'])

    elif frequency == 'monthly':
        # Group by date and calculate the total revenue and production for each renewable energy source
        grouped = Power_DATA.groupby(['Year', 'Month']).agg(
            **{f'TotalRevenue{source}': (source, lambda x: (x * Power_DATA.loc[x.index, auction_column]).sum()) for source in energy_sources},
            **{f'TotalProduction{source}': (source, 'sum') for source in energy_sources},
            TotalRevenueBaseload=('Load', lambda x: (x * Power_DATA.loc[x.index, auction_column]).sum()),
            BaseloadProduction=('Load', 'sum')
        )

        # Calculate the average price (capture price) for each renewable energy source
        for source in energy_sources:
            grouped[f'Capture_Price_{source}'] = grouped[f'TotalRevenue{source}'] / grouped[f'TotalProduction{source}']

        grouped['Baseload_Price'] = grouped['TotalRevenueBaseload'] / grouped['BaseloadProduction']

        # Calculate the capture factor compared to baseload production
        for source in energy_sources:
            grouped[f'Capture_Factor_{source}'] = grouped[f'Capture_Price_{source}'] / grouped['Baseload_Price']

        grouped.reset_index(inplace=True)

        # Display the results
        capture_factors = grouped[['Year', 'Month'] + [f'Capture_Factor_{source}' for source in energy_sources]]
        capture_factors['Date'] = pd.to_datetime(capture_factors[['Year', 'Month']].assign(day=1))
        capture_factors.reset_index(inplace=True)

        # Calculate percentage change
        percentage_change_df = capture_factors.set_index('Date').pct_change()
        percentage_change_df = percentage_change_df.dropna().reset_index()
        percentage_change_df['Date'] = pd.to_datetime(percentage_change_df['Date'])

    elif frequency == 'yearly':
        # Group by date and calculate the total revenue and production for each renewable energy source
        grouped = Power_DATA.groupby(['Year']).agg(
            **{f'TotalRevenue{source}': (source, lambda x: (x * Power_DATA.loc[x.index, auction_column]).sum()) for source in energy_sources},
            **{f'TotalProduction{source}': (source, 'sum') for source in energy_sources},
            TotalRevenueBaseload=('Load', lambda x: (x * Power_DATA.loc[x.index, auction_column]).sum()),
            BaseloadProduction=('Load', 'sum')
        )

        # Calculate the average price (capture price) for each renewable energy source
        for source in energy_sources:
            grouped[f'Capture_Price_{source}'] = grouped[f'TotalRevenue{source}'] / grouped[f'TotalProduction{source}']

        grouped['Baseload_Price'] = grouped['TotalRevenueBaseload'] / grouped['BaseloadProduction']

        # Calculate the capture factor compared to baseload production
        for source in energy_sources:
            grouped[f'Capture_Factor_{source}'] = grouped[f'Capture_Price_{source}'] / grouped['Baseload_Price']

        grouped.reset_index(inplace=True)

        # Display the results
        capture_factors = grouped[['Year'] + [f'Capture_Factor_{source}' for source in energy_sources]]
        capture_factors['Date'] = pd.to_datetime(capture_factors['Year'], format='%Y', errors='coerce')

        # Reset index to handle calculation issues
        capture_factors.reset_index(drop=True, inplace=True)

        # Calculate percentage change
        percentage_change_df = capture_factors.set_index('Date').pct_change()
        percentage_change_df = percentage_change_df.dropna().reset_index()
        percentage_change_df['Date'] = pd.to_datetime(percentage_change_df['Date'])
        percentage_change_df = percentage_change_df.reset_index()
        percentage_change_df['Date'] = pd.to_datetime(percentage_change_df['Date'])

    return percentage_change_df[['Date', 'Capture_Factor_Wind offshore', 'Capture_Factor_Wind onshore', 'Capture_Factor_Solar']]



def calculate_volatility(data, frequency, date_column='Date', value_column='Day Ahead Auction', weight_column='Load'):
    # Convert Date to datetime format
    data['Date'] = pd.to_datetime(data[date_column])
    
    if frequency == 'yearly':
        # Extract year from the Date column
        data['Year'] = data['Date'].dt.year
        
        # Calculate the standard deviation of 'Day Ahead Auction' grouped by Year
        volatility = data.groupby(['Year'])[value_column].std()

        # Weighted standard deviation if weight_column is provided
        if weight_column:
            weighted_volatility = data.groupby(['Year']).apply(
                lambda x: np.sqrt(np.average((x[value_column] - x[value_column].mean()) ** 2, weights=x[weight_column]))
            )
        else:
            weighted_volatility = None
            
    elif frequency == 'monthly':
        # Extract year and month from the Date column
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        
        # Calculate the standard deviation of 'Day Ahead Auction' grouped by Year and Month
        volatility = data.groupby(['Year', 'Month'])[value_column].std()

        # Weighted standard deviation if weight_column is provided
        if weight_column:
            weighted_volatility = data.groupby(['Year', 'Month']).apply(
                lambda x: np.sqrt(np.average((x[value_column] - x[value_column].mean()) ** 2, weights=x[weight_column]))
            )
        else:
            weighted_volatility = None
            
    else:  # Daily frequency
        # Extract day, year, and month from the Date column
        data['Day'] = data['Date'].dt.day
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        
        # Calculate the standard deviation of 'Day Ahead Auction' grouped by Year, Month, and Day
        volatility = data.groupby(['Year', 'Month', 'Day'])[value_column].std()

        # Weighted standard deviation if weight_column is provided
        if weight_column:
            weighted_volatility = data.groupby(['Year', 'Month', 'Day']).apply(
                lambda x: np.sqrt(np.average((x[value_column] - x[value_column].mean()) ** 2, weights=x[weight_column]))
            )
        else:
            weighted_volatility = None

    
    # Make dataframe for weighted volatility
    if weighted_volatility is not None:
        weighted_volatility_reset = weighted_volatility.reset_index()

        # Create the 'Date' column based on the frequency
        if 'Day' in weighted_volatility_reset.columns:
            weighted_volatility_reset['Date'] = pd.to_datetime(weighted_volatility_reset[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1) + '-01')
            weighted_volatility_reset = weighted_volatility_reset.drop(columns=['Day'])
        elif 'Month' in weighted_volatility_reset.columns:
            weighted_volatility_reset['Date'] = pd.to_datetime(weighted_volatility_reset[['Year', 'Month']].astype(str).agg('-'.join, axis=1) + '-01')
        else:
            weighted_volatility_reset['Date'] = pd.to_datetime(weighted_volatility_reset[['Year']].astype(str).agg('-'.join, axis=1) + '-01')

        # Compute the percentage change for 'Weighted_volatility_adj'
        weighted_volatility_reset['Weighted_volatility_adj'] = weighted_volatility_reset[0].pct_change().dropna()
        weighted_volatility_reset = weighted_volatility_reset[['Date', 'Weighted_volatility_adj']]
        weighted_volatility_reset.dropna(inplace=True)
    else:
        weighted_volatility_reset = None
  
    weighted_volatility_reset['Date'] = pd.to_datetime(weighted_volatility_reset['Date']).dt.date
    weighted_volatility_reset['Date'] = pd.to_datetime(weighted_volatility_reset['Date'])

    return weighted_volatility_reset



def filter_dates_by_interval(data, column_names, interval_start, interval_end, date, print_deleted_dates=False):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[data['Date'] < date]
    
    filtered_dates = data
    for column_name in column_names:
        filtered_dates = filtered_dates[(filtered_dates[column_name] >= interval_start) & (filtered_dates[column_name] <= interval_end)]
    
    deleted_dates = data[~data.index.isin(filtered_dates.index)]  # Get dates not in filtered_dates
    
    if print_deleted_dates:
        print("Deleted Dates:")
        print(deleted_dates['Date'])
    
    return filtered_dates

def fixed_effect_model(data, category_column, capture_column, outcome_column, date_component):
    # Extract the specified component from the datetime column
    if date_component == 'year':
        data[date_component] = data['Date'].dt.year
    elif date_component == 'month':
        data[date_component] = data['Date'].dt.month
    elif date_component == 'day':
        data[date_component] = data['Date'].dt.day

    # Add constant and relevant columns to the design matrix
    X = sm.add_constant(data[[capture_column, date_component]])
    y = data[outcome_column]

    # Fit the model using robust linear regression with fixed effects
    model = sm.RLM(y, X).fit()
    
    # Get the residuals
    residuals = model.resid

    # Perform the Ljung-Box test
    lb_test_statistic = acorr_ljungbox(residuals)

    # Print the test statistic and p-value
    print("Ljung-Box test statistic:", lb_test_statistic)

    # Perform Durbin-Watson test for autocorrelation
    #dw_test = durbin_watson(model.resid)
    #dw_test = print('Autocorrelation',dw_test)
    return model.summary(), #dw_test


def analyze_power_data(frequency='daily', interval_start=-1.0, interval_end=1.0, date=pd.Timestamp(2024, 1, 1), FE='year'):
    # Read power data
    Power_DATA = read_power_data()
    
    # Define energy sources
    energy_sources = ['Wind offshore', 'Wind onshore', 'Solar']
    
    # Calculate capture factors
    percentage_change_df = calculate_capture_factors(Power_DATA, energy_sources, frequency)
    
    # Calculate volatility
    weighted_volatility_reset = calculate_volatility(Power_DATA, frequency=frequency)
    
    # Merge data
    ANALYSE_DATA = pd.merge(percentage_change_df, weighted_volatility_reset, on='Date', how='inner')
    
    # Filter Data
    column_names = ['Capture_Factor_Wind offshore', 'Capture_Factor_Wind onshore', 'Capture_Factor_Solar']
    ANALYSE_DATA_F = filter_dates_by_interval(ANALYSE_DATA, column_names, interval_start, interval_end, date, print_deleted_dates=False)
    
    # The Analysis:
    onshore_summary_year = fixed_effect_model(ANALYSE_DATA_F, 'Month', 'Capture_Factor_Wind onshore', 'Weighted_volatility_adj', FE)
    offshore_summary_month = fixed_effect_model(ANALYSE_DATA_F, 'Month', 'Capture_Factor_Wind offshore', 'Weighted_volatility_adj', FE)
    solar_summary_day = fixed_effect_model(ANALYSE_DATA_F, 'Month', 'Capture_Factor_Solar', 'Weighted_volatility_adj', FE)
    
    # Print the summaries
    print(onshore_summary_year)
    print(offshore_summary_month)
    print(solar_summary_day)


# QuantTools Repository

## Overview

Welcome to the QuantTools repository, a collection of Python scripts for quantitative analysis of financial data. This repository provides tools to extract, analyze, and visualize data related to financial markets.


## Script: corr_matrix.py

### Dependencies

- `requests`: To extract data from websites.
- `yfinance`: For downloading financial data.
- `pandas`: For data manipulation and analysis.
- `seaborn` and `matplotlib.pyplot`: For plotting data and visualizations.
- `logging`: To log information about data downloads and potential issues.


### Purpose

The `corr_matrix.py` script analyzes the correlation among S&P 500 stock returns based on historical data. It performs the following tasks:

- Retrieves S&P 500 ticker symbols from Wikipedia.
- Downloads adjusted closing prices (`Adj Close`) of these stocks from Yahoo Finance.
- Calculates daily percentage returns.
- Computes and visualizes the correlation matrix of daily returns using a cluster map.

### Functions

#### `get_sp500_tickers()`

- **Purpose**: Extracts the list of S&P 500 ticker symbols from a Wikipedia page and cleans the symbols for use with Yahoo Finance.

#### `download_financial_data(tickers, startdate, enddate)`

- **Purpose**: Downloads adjusted closing prices (`Adj Close`) of specified ticker symbols over a defined time period.

#### `calculate_daily_returns(data)`

- **Purpose**: Computes the daily percentage change in stock prices based on the downloaded data.

#### `calculate_corr_matrix(returns)`

- **Purpose**: Generates the correlation matrix of daily returns for S&P 500 stocks.

#### `plot_corr_matrix(corr_matrix, output_file='correlation_matrix.png')`

- **Purpose**: Visualizes the correlation matrix as a cluster map to identify groups of stocks with similar return patterns.

#### `main()`

- **Purpose**: Orchestrates the execution of functions to perform the entire data analysis pipeline.

### Logging

- **Purpose**: Logs information, warnings, and errors during data download and analysis.

## Usage

To run the `corr_matrix.py` script:
1. Ensure all dependencies (`requests`, `yfinance`, `pandas`, `seaborn`, `matplotlib`) are installed.
2. Execute the script using Python:

   ```bash
   python corr_matrix.py

## Script: Linear Regression Analysis

### Dependencies

- `yfinance`: For fetching historical stock prices.
- `pandas`: For data manipulation and analysis.
- `statsmodels`: To perform statistical models and tests.
- `numpy`: For numerical operations.
- `scipy`: For statistical functions.
- `matplotlib.pyplot`: For creating visualizations.

### Purpose

The `linear_regression_analysis.py` script conducts linear regression analysis on stock returns using Fama-French factors. It provides capabilities to:

- Download historical stock prices from Yahoo Finance.
- Fetch Fama-French Five Factor data.
- Prepare data for regression by merging stock returns with Fama-French factors.
- Perform linear regression using both built-in and manual methods.
- Visualize observed vs. predicted excess returns and display regression metrics.

### Functions

#### `download_financial_data(ticker, time_frame='5y', start_date=None, end_date=None)`

- **Purpose**: Downloads adjusted closing prices of a specified stock ticker.
- **Parameters**:
  - `ticker` (str): Stock ticker symbol.
  - `time_frame` (str, optional): Predefined period or custom date range.
  - `start_date` (str, optional): Start date for data retrieval.
  - `end_date` (str, optional): End date for data retrieval.
- **Returns**: Series containing percentage changes in adjusted closing prices.

#### `get_factors()`

- **Purpose**: Fetches and cleans Fama-French Five Factor data from Kenneth French's data library.
- **Returns**: DataFrame with cleaned Fama-French factors.

#### `get_regressor_and_observation(stock_data, ff_factors)`

- **Purpose**: Prepares independent variables (Fama-French factors) and dependent variable (excess returns) for regression.
- **Parameters**:
  - `stock_data` (DataFrame): Historical stock returns.
  - `ff_factors` (DataFrame): Fama-French factors.
- **Returns**: DataFrames `X` (independent variables) and `y` (dependent variable).

#### `linear_regr_builtIn(X, y)`

- **Purpose**: Performs linear regression using built-in functions from `statsmodels`.
- **Parameters**:
  - `X` (DataFrame): Independent variables with intercept added.
  - `y` (Series): Dependent variable (excess returns).
- **Returns**: None (displays regression results summary).

#### `linear_regr_manual(X, y)`

- **Purpose**: Manually computes linear regression coefficients, predictions, residuals, and regression metrics.
- **Parameters**:
  - `X` (DataFrame): Independent variables with intercept added.
  - `y` (Series): Dependent variable (excess returns).
- **Returns**: Dictionary of regression metrics including R-squared, adjusted R-squared, F-statistics, p-values, MSE, and RMSE.

#### `plot_regression_results(results)`

- **Purpose**: Visualizes observed vs. predicted excess returns using a scatter plot.
- **Parameters**:
  - `results` (dict): Dictionary containing regression results.
- **Returns**: None (displays scatter plot with annotations).

#### `main()`

- **Purpose**: Main function to execute the entire regression analysis pipeline.
- **Usage**: Orchestrates data download, factor fetching, regression analysis, and visualization.

### Usage

To execute the `linear_regression_analysis.py` script:
1. Ensure all dependencies (`yfinance`, `pandas`, `statsmodels`, `numpy`, `scipy`, `matplotlib`) are installed.
2. Run the script using Python:

   ```bash
   python linear_regression_analysis.py

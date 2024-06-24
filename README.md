# QuantTools Repository

## Overview

Welcome to the QuantTools repository, a collection of Python scripts for quantitative analysis of financial data. This repository provides tools to extract, analyze, and visualize data related to financial markets.

## Dependencies

- `requests`: To extract data from websites.
- `yfinance`: For downloading financial data.
- `pandas`: For data manipulation and analysis.
- `seaborn` and `matplotlib.pyplot`: For plotting data and visualizations.
- `logging`: To log information about data downloads and potential issues.

## Script: corr_matrix.py

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

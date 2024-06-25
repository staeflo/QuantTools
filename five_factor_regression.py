#Financial data
import yfinance as yf
#For handeling data
import pandas as pd
#To access linear regression functions
import statsmodels.api as sm
import numpy as np
from scipy import stats
#For plotting data
import matplotlib.pyplot as plt


# Function to Download Historical Stock Prices
# This function fetches the adjusted closing prices of a given stock ticker using the yfinance library.
# Users can specify either a custom date range or a predefined time frame for the data.
# 
# Parameters:
# - ticker (str): The stock ticker symbol to download data for.
# - time_frame (str, optional): The predefined period to fetch data for. Defaults to '5y'.
#   Valid options include: {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}.
# - start_date (str, optional): The start date for fetching data in 'yyyy-mm-dd' format. Defaults to None.
# - end_date (str, optional): The end date for fetching data in 'yyyy-mm-dd' format. Defaults to None.
#
# Returns:
# - A series containing the percentage change of adjusted closing prices for the specified period.
#
# Usage:
# download_financial_data('AAPL', time_frame='1y')
# download_financial_data('AAPL', start_date='2023-01-01', end_date='2023-12-31')

def download_financial_data(ticker, time_frame='5y', start_date=None, end_date=None):
    if start_date and end_date:
        stock_returns = yf.download(ticker ,start = start_date, end= end_date)['Adj Close'].pct_change().dropna()
    else:
        stock_returns = yf.download(ticker , period=time_frame)['Adj Close'].pct_change().dropna()
    return stock_returns



# Download Fama-French Five Factor Data
# This function downloads the Fama-French Five Factor data from Kenneth French's data library.
# It reads the CSV data from the specified URL, cleans it by removing missing values,
# converts the index to datetime format, and scales the factor values to decimals.
#
# Returns: 
# - A DataFrame containing the cleaned Fama-French Five Factor data.
def get_factors():
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    ff_factors = pd.read_csv(url, skiprows=3, index_col=0)
    # Clean the data
    ff_factors = ff_factors.dropna()
    ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m%d')  # Convert index to datetime
    ff_factors = ff_factors.astype(float) / 100  # Convert to decimal
    return ff_factors


# Prepare data for regression
# This function merges stock returns data with Fama-French factors and prepares the
# independent and dependent variables for regression analysis.
#
# Parameters:
# - stock_data : DataFrame containing stock returns with date as the index.
# - ff_factors : DataFrame containing Fama-French factors with date as the index.
#
# Returns:
# - X : DataFrame containing the independent variables (Fama-French factors).
# - y : Series containing the dependent variable (excess stock returns).

def get_regressor_and_observation(stock_data, ff_factors):
    # Merge stock data with Fama-French factors on the date index
    data = pd.merge(stock_data, ff_factors, left_index=True, right_index=True)
    
    # Rename columns for clarity
    # Stock_Returns: Returns of the stock
    # Mkt-RF: Market return minus risk-free rate
    # SMB: Small Minus Big (size factor)
    # HML: High Minus Low (value factor)
    # RMW: Robust Minus Weak (profitability factor)
    # CMA: Conservative Minus Aggressive (investment factor)
    # RF: Risk-Free rate
    data.columns = ['Stock_Returns', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    
    # Calculate excess returns (stock returns minus risk-free rate)
    data['Excess_Return'] = data['Stock_Returns'] - data['RF']
    
    # Define independent variables (factors) and dependent variable (excess returns)
    X = data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    y = data['Excess_Return']
    
    return X, y


#Linear regression with built-in function

def linear_regr_builtIn(X, y):
    # Add a constant to the independent variables
    # This adds a column of ones to the DataFrame `X`, which represents the intercept in the regression model.
    X = sm.add_constant(X)
    # The excess returns not captured by the exposure to risk premia is known as alpha
    X = X.rename(columns={'const': 'alpha'})
    # Fit the regression model
    # This creates an Ordinary Least Squares (OLS) regression model and fits it using the provided
    # independent variables `X` and dependent variable `y`.
    model = sm.OLS(y, X).fit()
    
    # Display the regression results
    # This prints a summary of the regression results, including coefficients, p-values, R-squared, etc.
    print(model.summary())





def linear_regr_manual(X, y):
    # Add a column of ones for the intercept term in X
    X = np.column_stack((np.ones(len(X)), X))
    
    # Calculate beta_hat (coefficients) manually
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Calculate predictions
    y_hat = X @ beta_hat
    
    # Calculate residuals
    residuals = y - y_hat
    
    # Number of observations
    n = len(y)
    
    # Number of independent variables (excluding intercept)
    k = X.shape[1] - 1
    
    # Calculate SSR (Sum of Squared Residuals)
    SSR = np.sum(residuals**2)
    
    # Calculate degrees of freedom
    df_full = n - k - 1
    df_reduced = n - 1  # Reduced model has only intercept
    
    # Calculate R-squared for full model
    y_mean = np.mean(y)
    Var_mean = np.sum((y - y_mean)**2)
    R_squared_full = 1 - SSR / Var_mean
    
    # Calculate R-squared for the intercept-only model
    X_reduced = np.ones_like(X[:, 0])
    beta_hat_reduced = np.mean(y)
    y_hat_reduced = X_reduced * beta_hat_reduced
    residuals_reduced = y - y_hat_reduced
    SSR_reduced = np.sum(residuals_reduced**2)
    R_squared_reduced = 1 - SSR_reduced / Var_mean
    
    # Calculate standard error of the regression (MSE and RMSE)
    MSE = SSR / df_full
    RMSE = np.sqrt(MSE)
    
    # Calculate F-statistic and its p-value against intercept-only model
    F_statistic_null = ((R_squared_full - R_squared_reduced) / k) / (SSR_reduced / df_reduced)
    p_value = stats.f.sf(F_statistic_null, k, df_reduced)
    

    # Calculate F-statistic of adjusted R^2
    F_statistic_adj = (R_squared_full / k) / ((1 - R_squared_full) / (n - k - 1))
    # Calculate p-value for F-statistic of adjusted R^2
    p_value_adj = stats.f.sf(F_statistic_adj, k, n - k - 1)


    # Calculate adjusted R-squared for full model
    adj_R_squared_full = 1 - (1 - R_squared_full) * ((n - 1) / df_full)
    


    # Return detailed metrics
    return {
        'y': y,
        'y_hat': y_hat,
        'R_squared_full': R_squared_full,
        'adj_R_squared_full': adj_R_squared_full,
        'F_statistic_null': F_statistic_null,
        'p_value_null': p_value,
        'F_statistic_adj': F_statistic_adj,
        'p_value_adj': p_value_adj,
        'MSE': MSE,
        'RMSE': RMSE
    }

def plot_regression_results(results):
    # Extract necessary data from results dictionary
    y = results['y']
    y_hat = results['y_hat']
    annotation_text = (
        f'R-squared (R^2): {results["R_squared_full"]:.4f}\n'
        f'Adjusted R-squared: {results["adj_R_squared_full"]:.4f}\n'
        f'F-statistic (against intercept-only): {results["F_statistic_null"]:.4f}\n'
        f'p-value (against intercept-only): {results["p_value_null"]:.4f}\n'
        f'F-statistic (adjusted R-squared): {results["F_statistic_adj"]:.4f}\n'
        f'p-value (adjusted R-squared): {results["p_value_adj"]:.4f}'
    )

    # Scatter plot of observed vs. predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_hat, color='blue', alpha=0.6)
    plt.plot(y, y, color='red', linestyle='--')  # Plotting the diagonal line (perfect fit)
    plt.title('Observed vs. Predicted Excess Returns')
    plt.xlabel('Observed Excess Returns')
    plt.ylabel('Predicted Excess Returns')
    plt.grid(True)
    
    # Adding annotations
    plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()



    # Scatter plot of observed vs. predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_hat, color='blue', alpha=0.6)
    plt.plot(y, y, color='red', linestyle='--')  # Plotting the diagonal line (perfect fit)
    plt.title('Observed vs. Predicted Excess Returns')
    plt.xlabel('Observed Excess Returns')
    plt.ylabel('Predicted Excess Returns')
    plt.grid(True)
    plt.show()


def main():
# Define the ticker symbol for the S&P 500 (example)
    ticker = '^GSPC'

    
    try:
        # Download financial data
        data = download_financial_data(ticker)
        
        # Get Fama-French factors
        factors = get_factors()
        
        # Prepare data for regression
        X, y = get_regressor_and_observation(data, factors)
        
        # Perform linear regression using built-in functions
        linear_regr_builtIn(X, y)

        # Perform linear regression using built-in functions
        output = linear_regr_manual(X, y)

        plot_regression_results(output)    

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()




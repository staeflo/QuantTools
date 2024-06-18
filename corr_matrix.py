# To extract data from Website
import requests
from io import StringIO
#Financial data
import yfinance as yf
#For handeling data
import pandas as pd
#For plotting data
import seaborn as sns
import matplotlib.pyplot as plt
#To log missing data from yf
import logging

# Set up logging
logging.basicConfig(filename='data_download.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Function that extracts the S&P500 ticker list from Wikipedia
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = requests.get(url).text
    data = StringIO(html)
    df = pd.read_html(data,header=0)[0]
    tickers = df['Symbol'].tolist()
    #Clean the tickers. Some contain '.', but yf expects '-', e.g. 'BRK.B' -> 'BRK-B'
    clean_tickers = []
    for ticker in tickers:
        ticker = ticker.replace('.','-')
        clean_tickers.append(ticker)
    return clean_tickers


#Function that downloads financial data
def download_financial_data(tickers, startdate,enddate):
    data = yf.download(tickers, start = startdate, end= enddate)['Adj Close']
    logging.info('Initial downloaded data:')
    #logging.info(data.head())

    #Identify tickers that failed to download and contain only NaN values
    all_nan = data.columns[data.isna().all()].tolist()
    if all_nan:
        logging.warning(f'Data is missing for tickers: {all_nan}')

    #Remove the empty columns
    data_cleaned = data.dropna(axis=1,how='all')
    #logging.info('Data after removing columns with all NaN values:')
    #logging.info(data_cleaned.head())

    return data_cleaned
    
# Calculate percentage change from downloaded data
def calculate_daily_returns(data):
    returns = data.pct_change().dropna()
    #logging.info('Data after calculating returns:')
    #logging.info(returns.head())
    return returns

#Calculate correlation matrix
def calculate_corr_matrix(returns):
    corr_matrix = returns.corr()
    return corr_matrix

#Show the correlation matrix as a clusterplot
def plot_corr_matrix(corr_matrix , output_file = 'correlation_matrix.png'):
    plt.figure(figsize=(16,16))
    sns.clustermap(corr_matrix, cmap="coolwarm")
    plt.title('Correlation Matrix of S&P 500 Daily Returns')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()  



def main():
    #Get tickers from the web
    tickers = get_sp500_tickers()
    #Define Start and End Date of the analysis
    startdate = '2023-05-01'
    enddate = '2023-06-01'
    #Download Data
    data = download_financial_data(tickers,startdate,enddate)
    #Calculate Returns
    returns = calculate_daily_returns(data)
    #Calculate Correlation matrix
    corr_matrix = calculate_corr_matrix(returns)
    #Clustermap Plot of correlation matrix
    plot_corr_matrix(corr_matrix)




    # Check if returns is empty
    if returns.empty:
        logging.error("No returns data available. Please check the data download and date range.")
        print("Error: No returns data available. Please check the data download and date range.")
        return



if __name__ =="__main__":
    main()
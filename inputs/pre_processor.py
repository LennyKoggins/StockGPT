import yfinance as yf
import pandas as pd
import numpy as np

# Load the ticker symbols from the CSV file
ticker_df = pd.read_csv('Ticker.csv')
ticker_symbols = ticker_df['SYMBOL'].tolist()

# Define the number of ticker symbols to process (you can change this to any number, like 10)
number_of_symbols_to_process = 10

# Limit the ticker symbols to the first 'number_of_symbols_to_process'
ticker_symbols = ticker_symbols[:number_of_symbols_to_process]

# Add ".NS" to each ticker symbol
ticker_symbols = [symbol + '.NS' for symbol in ticker_symbols]


# Function to calculate beta
def calculate_beta(stock_returns, market_returns):
    covariance = np.cov(stock_returns, market_returns)[0][1]
    variance = np.var(market_returns)
    beta = covariance / variance
    return beta


# Use an ExcelWriter to save all data to one Excel file with multiple sheets
excel_file_path = "stock_data_with_calculations.xlsx"
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    # Loop through each ticker symbol and run the analysis
    for stock_symbol in ticker_symbols:
        try:
            print(f"Processing {stock_symbol}...")

            # Download stock data for the specified date range
            data = yf.download(stock_symbol, start='2024-01-16', end='2024-09-18')

            # Fetch additional stock info
            stock = yf.Ticker(stock_symbol)
            info = stock.info

            # Initialize an empty DataFrame to store calculated data
            calculated_data = pd.DataFrame(index=data.index)

            # Calculate dynamic parameters only if data is available
            if 'sharesOutstanding' in info and 'Close' in data.columns:
                calculated_data['Market_Cap'] = data['Close'] * info['sharesOutstanding']

            if 'marketCap' in info and 'totalDebt' in info and 'totalCash' in info:
                calculated_data['Enterprise_Value'] = info['marketCap'] + info['totalDebt'] - info['totalCash']

            if 'bookValue' in info and 'Close' in data.columns:
                calculated_data['Price_To_Book'] = data['Close'] / info['bookValue']

            if 'totalRevenue' in info and 'marketCap' in info:
                calculated_data['Price_To_Sales'] = info['marketCap'] / info['totalRevenue']

            # Calculate PE Ratio
            if 'trailingEps' in info and 'Close' in data.columns:
                calculated_data['PE_Ratio'] = data['Close'] / info['trailingEps']

            # Calculate PEG Ratio
            if 'earningsGrowth' in info and info['earningsGrowth'] > 0:
                calculated_data['PEG_Ratio'] = calculated_data['PE_Ratio'] / (info['earningsGrowth'] * 100)
            else:
                calculated_data['PEG_Ratio'] = None

            # Calculate 52-Week Change
            if 'Close' in data.columns and len(data) > 251:
                calculated_data['52WeekChange'] = ((data['Close'] - data['Close'].shift(251)) / data['Close'].shift(
                    251)) * 100
            else:
                calculated_data['52WeekChange'] = None

            # Calculate Dividend Yield
            if 'dividendRate' in info and 'Close' in data.columns:
                calculated_data['Dividend_Yield'] = (info['dividendRate'] / data['Close']) * 100

            # Calculate Payout Ratio
            if 'trailingEps' in info and 'dividendRate' in info and info['trailingEps'] > 0:
                calculated_data['Payout_Ratio'] = (info['dividendRate'] / info['trailingEps']) * 100
            else:
                calculated_data['Payout_Ratio'] = None

            total_debt = info.get('totalDebt', np.nan)  # Assume NaN if not available
            total_equity = info.get('totalEquity', 0)  # Assume zero if not available

            # Calculate Debt to Equity Ratio if possible
            if total_equity != 0:
                debt_to_equity_ratio = total_debt / total_equity
            else:
                debt_to_equity_ratio = np.nan  # Not calculable if total equity is zero

            # Store calculated Debt to Equity Ratio
            calculated_data['Debt_To_Equity'] = debt_to_equity_ratio
            calculated_data['Total_Debt'] = info.get('totalDebt', np.nan)

            # Download market index data (e.g., Nifty 50) for comparison
            market_symbol = '^NSEI'  # Nifty 50 index
            market_data = yf.download(market_symbol, start='2023-09-16', end='2024-09-17')

            # Calculate Beta
            if 'Close' in data.columns and 'Close' in market_data.columns:
                stock_returns = data['Close'].pct_change().dropna()
                market_returns = market_data['Close'].pct_change().dropna()

                # Align the index by date to ensure returns calculations match
                aligned_stock_returns, aligned_market_returns = stock_returns.align(market_returns, join='inner')

                if len(aligned_stock_returns) > 0 and len(aligned_market_returns) > 0:
                    calculated_data['Beta'] = calculate_beta(aligned_stock_returns, aligned_market_returns)

            # Append other direct info parameters if available
            for key in ['profitMargins', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins',
                        'operatingMargins', 'enterpriseToRevenue', 'enterpriseToEbitda', 'auditRisk', 'boardRisk',
                        'compensationRisk', 'shareHolderRightsRisk',
                        'governanceEpochDate', 'compensationAsOfEpochDate', 'dividendRate',
                        'dividendYield', 'payoutRatio', 'fiveYearAvgDividendYield', 'beta', 'bookValue',
                        'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 'earningsQuarterlyGrowth',
                        'lastSplitFactor', 'lastSplitDate', 'firstTradeDateEpochUtc',
                        'financialCurrency']:
                if key in info:
                    calculated_data[key] = info[key]

            # Merge calculated data with the original stock data
            final_data = data.join(calculated_data)

            # Write the data to the Excel file in a new sheet for each ticker
            final_data.to_excel(writer, sheet_name=stock_symbol)

            print(f"Stock data with dynamic calculations for {stock_symbol} saved successfully in the Excel sheet")

        except Exception as e:
            print(f"Error processing {stock_symbol}: {e}")
            continue

print(f"All data saved successfully in {excel_file_path}")


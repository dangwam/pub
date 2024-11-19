
import pandas as pd
import os
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns




### download data using yfinance
start_date  = "2024-01-01"
end_date  = "2024-08-23"
read_flag = 'Y'

def test(start_date, end_date):
    #data = yf.download(stocks, start=start_date, end=end_date)
    data = pd.read_excel(excel_file_path,sheet_name='adj_close_price')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    # Normalize the prices using z-scores
    data_normalized = data.apply(zscore)
    # Plotting
    data_normalized.plot(title='Stock Prices Over Time', ylabel='Stock Price', xlabel='Date', figsize=(12, 8))
    plt.legend(title='Stocks', bbox_to_anchor=(1, 1))
    plt.show()

    # Set the starting point for each stock to 100
    df = data
    df_relative = df / df.iloc[0] * 100
    print(df_relative)
    # Plotting relative prices
    df_relative.plot(title='Relative Stock Prices Over Time', ylabel='Relative Stock Price (Starting at 100)', xlabel='Date', figsize=(12, 8))
    plt.legend(title='Stocks', bbox_to_anchor=(1, 1))
    plt.show()

    # Plotting normalized prices with additional styling
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")  # Use seaborn for improved styling
    df_relative.plot(ax=plt.gca(), legend=True)
    # Adding labels and title
    plt.title('Normalized Stock Prices Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Normalized Stock Price (Z-Score)', fontsize=14)
    
    # Adding a legend outside the plot
    plt.legend(title='Stocks', bbox_to_anchor=(1, 1), loc='upper left')
    
    # Show the plot
    plt.show()


def get_industry_data(start_date, end_date, read_flag):
    formatted_end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
    stocks = ['XLK', 'XLB', 'XLI', 'XLE', 'XLV', 'XLY', 'XLF', 'XLU', 'XLP']
    arg1 = 'industry'
    #base_folder_path = r"C:\Users\pythonProject\MyJupyterNotebooks"
    base_folder_path = r"B:\mayank\fin\stocks\notebook_data"
    subfolder_name = "data"
    folder_path = os.path.join(base_folder_path, subfolder_name)
    # build the file name
    file_name_suffix = arg1 + '_yf_data.xlsx'
    full_file_name = f"{formatted_end_date}_{file_name_suffix}"
    #os.makedirs(folder_path, exist_ok=True)
    # Specify the Excel file path
    excel_file_path = os.path.join(folder_path, full_file_name)
    
    if read_flag == 'Y': ## download data from yfinance ::Open,High,Low,Close,Adj Close,Volume are the data column names
        data = yf.download(stocks, start=start_date, end=end_date).dropna()
        data_close = round(data['Adj Close'], 2)
        df_relative = round(data_close / data_close.iloc[0] * 100, 1)
        #f_relative = data_close.sort_values(by='Adj Close', ascending=False) / data_close.iloc[0] * 100
        #data.to_excel(excel_file_path, index=True, sheet_name=arg1)
        with pd.ExcelWriter(excel_file_path) as writer:
             # Write each DataFrame to a different sheet
             data.to_excel(writer, sheet_name='industry', index=True)
             data_close.to_excel(writer, sheet_name='adj_close_price', index=True)
             df_relative.to_excel(writer, sheet_name='relative_price', index=True)
    else:
        ## read data from the file
        data = pd.read_excel(excel_file_path,sheet_name="industry",index=True)
    # Step 2: Extract information about the MultiIndex
            
    return data_close    

def get_bitcoin_data(start_date, end_date, read_flag):
    formatted_end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
    stocks = ['BTC-USD', 'MSTR', 'BITX', 'BITO', 'ARKB', 'MARA', 'CLSK']
    arg1 = 'crypto'
    #base_folder_path = r"C:\Users\pythonProject\MyJupyterNotebooks"
    base_folder_path = r"B:\mayank\fin\stocks\notebook_data"
    subfolder_name = "data"
    folder_path = os.path.join(base_folder_path, subfolder_name)
    # build the file name
    file_name_suffix = arg1 + '_yf_data.xlsx'
    full_file_name = f"{formatted_end_date}_{file_name_suffix}"
    #os.makedirs(folder_path, exist_ok=True)
    # Specify the Excel file path
    excel_file_path = os.path.join(folder_path, full_file_name)
    
    if read_flag == 'Y': ## download data from yfinance ::Open,High,Low,Close,Adj Close,Volume are the data column names
        data = yf.download(stocks, start=start_date, end=end_date).dropna()
        data_close = round(data['Adj Close'], 2)
        df_relative = round(data_close / data_close.iloc[0] * 100, 1)
        #f_relative = data_close.sort_values(by='Adj Close', ascending=False) / data_close.iloc[0] * 100
        #data.to_excel(excel_file_path, index=True, sheet_name=arg1)
        with pd.ExcelWriter(excel_file_path) as writer:
             # Write each DataFrame to a different sheet
             data.to_excel(writer, sheet_name='industry', index=True)
             data_close.to_excel(writer, sheet_name='adj_close_price', index=True)
             df_relative.to_excel(writer, sheet_name='relative_price', index=True)
    else:
        ## read data from the file
        data = pd.read_excel(excel_file_path,sheet_name="industry",index=True)
    # Step 2: Extract information about the MultiIndex
            
    return data_close    


#df = get_industry_data(start_date, end_date, read_flag)
#test(start_date, end_date)
#print(df.tail(2))

bit_df = get_bitcoin_data(start_date, end_date, read_flag)
print(bit_df.tail(20))




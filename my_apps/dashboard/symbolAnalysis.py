#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from yahooquery import Ticker
import yfinance as yf
import datetime , os, warnings
warnings.filterwarnings('ignore')

#######################
# Page configuration
st.set_page_config(
    page_title="Stock Analysis",
    page_icon=":bar_chart", 
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# Load data -- data functions
@st.cache_data
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()
    extended_symbols = ['RIVN', 'AVGO', 'SPY', 'QQQ', 'TSLA', 'MA']
    extended_companies = ['Rivian Automotive', 'Broadcom Inc', 'SPDR S&P 500 ETF', 'Invesco QQQ Trust', 'Tesla', 'Mastercard']
    # Combine tickers with extended symbols
    tickers.extend(extended_symbols)
    ##tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]), **dict(zip(extended_symbols, extended_companies)))
    return tickers, tickers_companies_dict

@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)

@st.cache_data
def yquery_summary_detail(symbol):
    tk = Ticker(symbol)
    summary_detail = tk.summary_detail
    return summary_detail[symbol]

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode("utf-8")

#######################
## inputs for downloading data
# Sidebar
with st.sidebar:
    st.title('🏂 Stock Analysis')
    st.sidebar.header("Stock Parameters")
    available_tickers, tickers_companies_dict = get_sp500_components()
    ticker = st.sidebar.selectbox("Ticker", available_tickers, format_func=tickers_companies_dict.get)
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)
    start_date = st.sidebar.date_input("Start date", datetime.date(2015, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date.today())

    if start_date > end_date:
        st.sidebar.error("The end date must fall after the start date")

    



#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown("#### ToDay's Price Action")
    summary_detail = yquery_summary_detail(ticker)

    #st.metric(label='test', value=4, delta=2)
    st.dataframe(df_selected_year_sorted,
                 column_order=("states", "population"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "states": st.column_config.TextColumn(
                        "States",
                    ),
                    "population": st.column_config.ProgressColumn(
                        "Population",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_year_sorted.population),
                     )}
                 )
    


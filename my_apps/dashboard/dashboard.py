import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import datetime , os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dashboard", page_icon=":bar_chart", layout="wide")  ## streamlit emojis icons webpage
st.title(":bar_chart: EDA Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

### File uploader
fl = st.file_uploader(":file_folder: Upload", type=(['csv', 'xls', 'xlsx', 'txt']))

if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename)
else:
    ## read from your directory
    df = pd.read_csv('SPY_stock_data_yahoo.csv')

col1, col2 = st.columns((2))
## min and max dates
df['Date'] = pd.to_datetime(df['Date']) 
start_date = pd.to_datetime(df['Date'].min())
end_date = pd.to_datetime(df['Date'].max())
num_records = df['Date'].count()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", start_date))
    
with col2:
    date2 = pd.to_datetime(st.date_input("End Date", end_date))

df_date_range = df[(df['Date'] > start_date) & (df['Date'] < end_date)]


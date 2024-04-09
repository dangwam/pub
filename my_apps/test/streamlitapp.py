# imports
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import numpy as np
import cufflinks as cf
from plotly.offline import iplot
import plotly.express as px

import pandas as pd
from plotly.graph_objs import Figure, Ohlc

# Sample data (replace with your actual data)
data = {
  'date': pd.to_datetime(['2023-12-01', '2023-12-04', '2023-12-05', '2023-12-06']),
  'open': [100, 105, 110, 108],
  'high': [108, 112, 115, 110],
  'low': [98, 102, 105, 102],
  'close': [105, 108, 112, 106]
}
df = pd.DataFrame(data)

# Function to create the Ohlc chart with configurable y-axis range
def create_stock_chart(df, y_axis_range=None):
  fig = Figure(go_layout={'title': 'Stock Price'})
  candlestick = Ohlc(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
  fig.add_trace(candlestick)

  # Update y-axis range if provided
  if y_axis_range:
    fig.update_yaxes(range=y_axis_range)

  # Configure layout for Streamlit
  fig.update_layout(margin={'t': 40, 'l': 20, 'r': 20, 'b': 20}, width=1000, height=500)
  return fig

# Streamlit App
import streamlit as st

# User input for y-axis range
y_axis_min = st.number_input("Minimum Price", min_value=min(df['low']), max_value=max(df['high']))
y_axis_max = st.number_input("Maximum Price", min_value=min(df['low']), max_value=max(df['high']), value=max(df['high']))

# Create the chart with user-defined y-axis range (optional)
if y_axis_min != y_axis_max:
  fig = create_stock_chart(df.copy(), y_axis_range=(y_axis_min, y_axis_max))
else:
  fig = create_stock_chart(df.copy())

# Display the chart in Streamlit
st.plotly_chart(fig)

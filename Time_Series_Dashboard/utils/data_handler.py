import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()
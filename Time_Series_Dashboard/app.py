# app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from models.time_series_model import TimeSeriesAnalysis
from utils.data_handler import fetch_stock_data
from utils.visualization import create_forecast_plot

def main():
    st.set_page_config(page_title="Time Series Dashboard", layout="wide")
    st.title("📈 Stock Price Analysis & Forecasting")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Parameters")
        symbol = st.text_input("Stock Symbol", value="AAPL")
        lookback = st.slider("Training Lookback (days)", 10, 100, 30)
        forecast_days = st.slider("Forecast Days", 5, 30, 7)
        
        today = datetime.now()
        start_date = st.date_input(
            "Start Date",
            value=today - timedelta(days=365),
            max_value=today
        )
        end_date = st.date_input(
            "End Date",
            value=today,
            max_value=today
        )
    
    try:
        # Fetch data
        with st.spinner('Fetching stock data...'):
            df = fetch_stock_data(symbol, start_date, end_date)
            if df.empty:
                st.error("No data available for the selected date range.")
                return
            
            prices = df['Close'].values
            dates = df.index
        
        # Initialize analysis
        analysis = TimeSeriesAnalysis()
        
        # Train model
        with st.spinner('Training model...'):
            X, y = analysis.train_model(prices, seq_length=lookback)
            
            # Generate predictions
            predictions = analysis.predict(X)
            actual = prices[lookback:]
            
            # Generate forecast
            forecast = analysis.generate_forecast(prices, lookback, forecast_days)
        
        # Calculate and display metrics
        mae, rmse = analysis.calculate_metrics(actual, predictions)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Absolute Error", f"${mae:.2f}")
        with col2:
            st.metric("Root Mean Squared Error", f"${rmse:.2f}")
        
        # Create and display plot
        fig = create_forecast_plot(
            dates[lookback:],
            actual,
            predictions.flatten(),
            forecast,
            symbol
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display raw data
        with st.expander("Show Raw Data"):
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
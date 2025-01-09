# README.md
# Time Series Dashboard

An interactive dashboard for stock price analysis and forecasting using LSTM neural networks.

## Features
- Real-time stock data fetching
- Interactive date range selection
- LSTM-based price prediction
- Future price forecasting
- Error analysis and metrics
- Interactive visualizations

## Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
```bash
streamlit run app.py
```

## Usage
1. Enter a stock symbol (e.g., AAPL, GOOGL)
2. Select date range
3. Adjust prediction parameters
4. View forecasts and analysis

## Project Structure
- `app.py`: Main Streamlit application
- `models/`: Contains the LSTM model implementation
- `utils/`: Helper functions for data handling and visualization
- `requirements.txt`: Project dependencies

## Technologies Used
- Streamlit
- TensorFlow
- Plotly
- yfinance
- scikit-learn

## License
MIT License
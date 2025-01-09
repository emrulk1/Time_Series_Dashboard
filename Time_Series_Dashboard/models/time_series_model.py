import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class TimeSeriesAnalysis:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def create_sequences(self, data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:(i + seq_length)]
            target = data[i + seq_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)
    
    def build_model(self, seq_length):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_model(self, data, seq_length=10, epochs=50):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = self.create_sequences(scaled_data, seq_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = self.build_model(seq_length)
        self.model.fit(X, y, epochs=epochs, verbose=0)
        
        return X, y
    
    def predict(self, X):
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def generate_forecast(self, data, lookback, forecast_days):
        last_sequence = data[-lookback:].reshape(1, lookback, 1)
        last_sequence = self.scaler.transform(last_sequence)
        
        forecast = []
        for _ in range(forecast_days):
            next_pred = self.model.predict(last_sequence)
            forecast.append(self.scaler.inverse_transform(next_pred)[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        return forecast
    
    def calculate_metrics(self, actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return mae, rmse

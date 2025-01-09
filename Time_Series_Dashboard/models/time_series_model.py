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
        """Create sequences for LSTM model"""
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:(i + seq_length)]
            target = data[i + seq_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)
    
    def build_model(self, seq_length):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_model(self, data, seq_length=10, epochs=50):
        """Train LSTM model with proper data scaling"""
        # Scale the entire dataset first
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences from scaled data
        X, y = self.create_sequences(scaled_data, seq_length)
        
        # Reshape X to (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build and train model
        self.model = self.build_model(seq_length)
        self.model.fit(X, y, epochs=epochs, verbose=0)
        
        return X, y
    
    def predict(self, X):
        """Generate predictions"""
        # Make predictions
        predictions = self.model.predict(X)
        # Inverse transform the predictions to get actual values
        predictions_reshaped = predictions.reshape(-1, 1)
        predictions_actual = self.scaler.inverse_transform(predictions_reshaped)
        return predictions_actual
    
    def generate_forecast(self, data, lookback, forecast_days):
        """Generate future forecasts"""
        # Scale the input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        # Prepare the last sequence
        last_sequence = scaled_data[-lookback:]
        last_sequence = last_sequence.reshape(1, lookback, 1)
        
        forecast = []
        current_sequence = last_sequence.copy()
        
        # Generate predictions
        for _ in range(forecast_days):
            # Predict next value
            next_pred = self.model.predict(current_sequence)
            
            # Add to forecast list (inverse transform to get actual value)
            forecast.append(self.scaler.inverse_transform(next_pred.reshape(-1, 1))[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0]
        
        return forecast
    
    def calculate_metrics(self, actual, predicted):
        """Calculate performance metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return mae, rmse
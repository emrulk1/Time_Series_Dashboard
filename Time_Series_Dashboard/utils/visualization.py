import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_forecast_plot(dates, actual, predicted, forecast, symbol):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} Stock Price', 'Forecast Error'),
        vertical_spacing=0.2
    )
    
    # Price plot
    fig.add_trace(
        go.Scatter(x=dates, y=actual, name="Actual", line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=predicted, name="Predicted", line=dict(color="red")),
        row=1, col=1
    )
    
    if forecast is not None:
        forecast_dates = pd.date_range(dates[-1], periods=len(forecast)+1)[1:]
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=forecast, name="Forecast",
                      line=dict(color="green", dash="dash")),
            row=1, col=1
        )
    
    # Error plot
    error = actual - predicted
    fig.add_trace(
        go.Scatter(x=dates, y=error, name="Error", line=dict(color="gray")),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_x=0.5,
        title_y=0.95
    )
    
    return fig
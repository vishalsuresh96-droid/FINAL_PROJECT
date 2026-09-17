from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

def get_forecast_data():
    # 1. Sample electricity data create pannrom
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    base = 5000 + 1000 * np.sin(2 * np.pi * dates.dayofyear / 365)
    demand = base + np.random.normal(0, 200, len(dates)) + dates.dayofweek * 50
    df = pd.DataFrame({'timestamp': dates, 'demand_kwh': demand})
    
    # 2. Train / Test split
    train_size = int(len(df) * 0.8)
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    X, y = df[['dayofyear', 'dayofweek']], df['demand_kwh']
    model = GradientBoostingRegressor()
    model.fit(X.iloc[:train_size], y.iloc[:train_size])

    # 3. Graph ku history data
    history = df.iloc[train_size-100:train_size]
    
    # 4. Future 7 days forecast
    future_dates = pd.date_range(start=df['timestamp'].max() + pd.Timedelta(days=1), periods=7, freq='D')
    future_df = pd.DataFrame({'timestamp': future_dates})
    future_df['dayofyear'] = future_df['timestamp'].dt.dayofyear
    future_df['dayofweek'] = future_df['timestamp'].dt.dayofweek
    forecast = model.predict(future_df[['dayofyear', 'dayofweek']])

    return {
        "train_dates": history['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
        "train_values": history['demand_kwh'].round(2).tolist(),
        "forecast_dates": future_dates.strftime('%Y-%m-%d').tolist(),
        "forecast_values": [round(float(x),2) for x in forecast],
        "train_count": train_size,
        "test_count": len(df) - train_size
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/forecast')
def forecast_api():
    data = get_forecast_data()
    return jsonify(data)

if __name__ == '__main__':
    print("Backend running at http://127.0.0.1:5000")
    app.run(debug=True)
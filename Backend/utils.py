import pandas as pd
import numpy as np

def preprocess_data(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df = df.set_index('DateTime').resample('H').mean().reset_index()
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    return df

def build_hybrid_forecast(df, prophet, lstm, scaler):
    df_prophet = df.rename(columns={'DateTime':'ds', 'Global_active_power':'y'})
    forecast = prophet.predict(df_prophet)
    df['prophet_pred'] = forecast['yhat'].values
    df['residual'] = df['Global_active_power'] - df['prophet_pred']
    
    # Prepare residuals for LSTM
    scaled_residuals = scaler.transform(df[['residual']])
    window = 24
    X = []
    for i in range(window, len(scaled_residuals)):
        X.append(scaled_residuals[i-window:i, 0])
    X = np.array(X).reshape(len(X), window, 1)
    
    # Predict residuals
    residual_preds = lstm.predict(X)
    residual_preds = scaler.inverse_transform(residual_preds).flatten()
    df = df.iloc[-len(residual_preds):].copy()
    df['hybrid_pred'] = df['prophet_pred'].values[-len(residual_preds):] + residual_preds
    return df[['DateTime', 'Global_active_power', 'prophet_pred', 'hybrid_pred']]

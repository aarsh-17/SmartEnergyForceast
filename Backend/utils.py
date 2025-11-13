import pandas as pd
import numpy as np

import pandas as pd

import pandas as pd

def preprocess_data(df):
    """
    Preprocesses raw energy consumption data for hybrid forecasting.
    Handles both 'DateTime' and 'Date'+'Time' input formats.
    """

    # --- Ensure DateTime column exists ---
    if 'DateTime' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        else:
            raise ValueError("CSV must contain either 'DateTime' or both 'Date' and 'Time' columns.")
    else:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

    # Drop bad rows
    df = df.dropna(subset=['DateTime'])

    # --- Convert Global_active_power safely ---
    if 'Global_active_power' in df.columns:
        df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    else:
        raise ValueError("'Global_active_power' column missing in CSV")

    # --- Keep only numeric columns for resampling ---
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist() + ['DateTime']
    df = df[numeric_cols]

    # --- Resample hourly, fill missing values ---
    df = (
        df.set_index('DateTime')
          .resample('H')
          .mean(numeric_only=True)
          .interpolate()
          .reset_index()
    )

    # --- Feature engineering ---
    df['hour'] = df['DateTime'].dt.hour
    df['day'] = df['DateTime'].dt.day
    df['month'] = df['DateTime'].dt.month
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    return df


def build_hybrid_forecast(df, prophet, lstm, scaler, window=24):
    """
    Hybrid Prophet + LSTM residual forecasting pipeline.
    Handles short inputs gracefully and ensures numeric stability.
    """

    # --- Prophet baseline prediction ---
    df_prophet = df.rename(columns={'DateTime': 'ds', 'Global_active_power': 'y'})
    forecast = prophet.predict(df_prophet)
    df['prophet_pred'] = forecast['yhat'].values
    df['residual'] = df['Global_active_power'] - df['prophet_pred']

    # --- Ensure residuals are valid numeric values ---
    df['residual'] = pd.to_numeric(df['residual'], errors='coerce').fillna(0)

    # --- Scale residuals safely ---
    scaled_residuals = scaler.transform(df[['residual']])

    # --- Create sliding window input for LSTM ---
    X = []
    for i in range(window, len(scaled_residuals)):
        X.append(scaled_residuals[i - window:i, 0])

    # --- Handle too-short input case ---
    if len(X) == 0:
        # Fallback: use Prophet-only forecast if not enough data
        print("⚠️ Not enough data for LSTM residual correction — using Prophet predictions only.")
        df['hybrid_pred'] = df['prophet_pred']
        return df[['DateTime', 'Global_active_power', 'prophet_pred', 'hybrid_pred']]

    # --- Predict residuals ---
    X = np.array(X).reshape(len(X), window, 1)
    residual_preds = lstm.predict(X, verbose=0)

    # --- Inverse scaling ---
    residual_preds = scaler.inverse_transform(residual_preds).flatten()

    # --- Align predictions with latest timestamps ---
    df_tail = df.iloc[-len(residual_preds):].copy()
    df_tail['hybrid_pred'] = df_tail['prophet_pred'].values + residual_preds
    df.loc[df_tail.index, 'hybrid_pred'] = df_tail['hybrid_pred']

    return df[['DateTime', 'Global_active_power', 'prophet_pred', 'hybrid_pred']]

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib, io
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

from utils import preprocess_data, build_hybrid_forecast

app = FastAPI(title="Smart Energy Consumption Forecaster")

# Allow frontend (React/Streamlit) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models

prophet = joblib.load('model/prophet_model.pkl')  # If saved manually
lstm = tf.keras.models.load_model("model/lstm_model.keras")

lstm.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_squared_error"]
)
scaler = joblib.load("model/scaler.pkl")

@app.get("/")
def home():
    return {"message": "Energy Forecaster API is running. Go to /docs for Swagger UI."}

@app.post("/predict")
async def predict(file: UploadFile):
    """Accept CSV file and return hybrid forecast."""
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Preprocess
    hourly = preprocess_data(df)
    
    # Run Prophet + LSTM hybrid forecast
    forecast_df = build_hybrid_forecast(hourly, prophet, lstm, scaler)

    # Return as JSON
    forecast_json = forecast_df.to_dict(orient="records")
    return {"forecast": forecast_json}

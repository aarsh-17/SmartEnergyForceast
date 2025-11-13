from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib, io
import tensorflow as tf
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import base64
from utils import preprocess_data, build_hybrid_forecast
from fastapi.responses import StreamingResponse

app = FastAPI(title="Smart Energy Consumption Forecaster")

# Allow frontend (React/Streamlit) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
async def predict_image(file: UploadFile):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), sep=',')
    hourly = preprocess_data(df)
    result = build_hybrid_forecast(hourly, prophet, lstm, scaler)

    plt.figure(figsize=(10, 5))
    plt.plot(result["DateTime"], result["Global_active_power"], label="Actual", color="black")
    plt.plot(result["DateTime"], result["prophet_pred"], label="Prophet", color="blue")
    plt.plot(result["DateTime"], result["hybrid_pred"], label="Hybrid", color="red")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
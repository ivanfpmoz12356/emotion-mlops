from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import time
import logging

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# load model artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")


class InputData(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Emotion API CI/CD with logging is running"}


@app.post("/predict")
def predict(data: InputData):

    start_time = time.time()

    logging.info("New prediction request received")

    # input
    features = data.features
    logging.info(f"Input size: {len(features)} features")

    # preprocessing
    X = np.array(features).reshape(1, -1)
    X = scaler.transform(X)

    # prediction
    pred = model.predict(X)
    emotion = le.inverse_transform(pred)

    end_time = time.time()
    duration = end_time - start_time

    logging.info(f"Prediction completed in {duration:.4f} seconds")
    logging.info(f"Predicted emotion: {emotion[0]}")

    return {
        "emotion": emotion[0],
        "prediction_time_sec": round(duration, 4)
    }
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load model artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")


@app.get("/")
def home():
    return {"message": "Emotion Recognition API is running"}


@app.post("/predict")
def predict(features: list):

    # pretvaranje inputa
    X = np.array(features).reshape(1, -1)

    # scaling (ISTO kao training)
    X = scaler.transform(X)

    # prediction
    pred = model.predict(X)

    # decode label
    emotion = le.inverse_transform(pred)

    return {
        "emotion": emotion[0]
    }
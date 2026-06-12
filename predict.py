import joblib
import numpy as np

# load artefakata
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")


# uzmi sample (ZA TEST)
# ovo može kasnije biti real input
sample = np.random.rand(227).reshape(1, -1)


# preprocessing
sample = scaler.transform(sample)

# prediction
pred = model.predict(sample)

emotion = le.inverse_transform(pred)

print("Prediction:", emotion[0])
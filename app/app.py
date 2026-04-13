import os
import joblib
import numpy as np
from fastapi import FastAPI

app = FastAPI()

# Resolve the trained model relative to the project root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(
    fixed_acidity: float,
    volatile_acidity: float,
    citric_acid: float,
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float
):
    features = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    prediction = model.predict(features)

    return {
        "Name": "Priyanka Kumari",
        "Roll_no": "2022BCD0057",
        "wine_quality": int(round(prediction[0]))
    }

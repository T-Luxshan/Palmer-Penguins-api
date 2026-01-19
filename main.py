from fastapi import FastAPI
from pydantic import BaseModel, validator
import pickle
import numpy as np

app = FastAPI(title="Palmer Penguins Species Predictor", version="1.0")

# Load model and class names at startup
with open("penguin_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    class_names = saved["class_names"]

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float

    @validator("bill_length_mm")
    def check_bill_length(cls, v):
        if v < 30 or v > 70:
            raise ValueError("bill_length_mm should be between ~30–70 mm")
        return v

    @validator("bill_depth_mm")
    def check_bill_depth(cls, v):
        if v < 13 or v > 22:
            raise ValueError("bill_depth_mm should be between ~13–22 mm")
        return v

    @validator("flipper_length_mm")
    def check_flipper(cls, v):
        if v < 170 or v > 240:
            raise ValueError("flipper_length_mm should be between ~170–240 mm")
        return v

    @validator("body_mass_g")
    def check_mass(cls, v):
        if v < 2700 or v > 6300:
            raise ValueError("body_mass_g should be between ~2700–6300 g")
        return 


@app.post("/predict")
async def predict(features: PenguinFeatures):
    # Prepare input in the same order as training
    X = np.array([
        features.bill_length_mm,
        features.bill_depth_mm,
        features.flipper_length_mm,
        features.body_mass_g
    ]).reshape(1, -1)

    # Predict class index (0, 1, 2)
    pred_idx = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    return {
        "predicted_species": class_names[pred_idx],
        "probabilities": {
            class_names[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }
    }


@app.get("/")
async def root():
    return {"message": "Palmer Penguins API is running.  → POST to /predict"}
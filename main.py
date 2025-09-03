
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Load model
model = joblib.load("yield_pipeline.pkl")

app = FastAPI(title="Crop Yield Prediction API")

# Input schema
class PredictInput(BaseModel):
    Crop_Type: str = Field(..., example="Corn")
    Soil_Type: str = Field(..., example="Loamy")
    N: float = Field(..., example=84.0)
    P: float = Field(..., example=66.0)
    K: float = Field(..., example=50.0)
    Temperature: float = Field(..., example=20.05)
    Humidity: float = Field(..., example=79.95)
    Wind_Speed: float = Field(..., example=8.59)

@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running"}

@app.post("/predict")
def predict(data: PredictInput):
    input_data = pd.DataFrame([{
        "Crop_Type": data.Crop_Type,
        "Soil_Type": data.Soil_Type,
        "Temperature": data.Temperature,
        "Humidity": data.Humidity,
        "Wind_Speed": data.Wind_Speed,
        "N": data.N,
        "P": data.P,
        "K": data.K
    }])

    try:
        prediction = model.predict(input_data)
        return {
            "input_used": input_data.to_dict(orient="records")[0],
            "predicted_crop_yield": float(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

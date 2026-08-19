from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import warnings

# Suppress scikit-learn version warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time ML API to predict if a Telco customer will churn.",
    version="1.0.0"
)

# 2. Load the artifacts (Model & Scaler) on startup
# Ensure these paths point to where you saved them in Day 1
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    raise RuntimeError("Model artifacts not found. Please run train.py first.")

# 3. Define the Input Data Schema using Pydantic
class ChurnPredictionRequest(BaseModel):
    # We expect a list of floats matching the exact number of features the model trained on
    features: list[float]

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [0.5, 1.2, -0.3, 0.0, 1.0, 0.0, 1.0, -1.5, 0.2] # Truncated example
            }
        }
    }

# 4. Define a simple health-check endpoint
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "API is running. Send POST requests to /predict"}

# 5. Define the Prediction Endpoint
@app.post("/predict")
def predict_churn(request: ChurnPredictionRequest):
    try:
        # Convert the incoming list into a 2D numpy array (1 row, N columns)
        input_data = np.array(request.features).reshape(1, -1)
        
        # Apply the exact same scaling used during training
        scaled_data = scaler.transform(input_data)
        
        # Generate prediction (0 or 1) and probability
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        # Return a structured JSON response
        return {
            "prediction_class": int(prediction),
            "churn_risk": "High" if prediction == 1 else "Low",
            "churn_probability": round(float(probability), 4)
        }
        
    except ValueError as e:
        # Catch feature length mismatches (e.g., sending 10 features when the model expects 45)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import warnings

# Suppress scikit-learn version warnings for cleaner server logs
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Initialize the FastAPI app
app = FastAPI(
    title="IBM Telco Churn Prediction API",
    description="Real-time ML API to predict customer churn based on expanded feature set.",
    version="2.0.0"
)

# 2. Load the artifacts (Model & Scaler) on startup
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Dynamically determine the exact number of features required
    EXPECTED_FEATURES = scaler.n_features_in_
    print(f"API successfully loaded model. Expecting exactly {EXPECTED_FEATURES} features.")
except FileNotFoundError:
    raise RuntimeError("Model or scaler artifacts not found. Please run train.py first to generate them.")

# 3. Define the Input Data Schema using Pydantic
class ChurnPredictionRequest(BaseModel):
    # We enforce that the user must send a list of floats
    features: list[float] = Field(
        ..., 
        description=f"An array of exactly {EXPECTED_FEATURES} numerical features (post-one-hot encoding)."
    )

# 4. Health-check endpoint
@app.get("/")
def read_root():
    return {
        "status": "healthy", 
        "message": "API is running. Send POST requests to /predict",
        "required_feature_count": EXPECTED_FEATURES
    }

# 5. Prediction Endpoint
@app.post("/predict")
def predict_churn(request: ChurnPredictionRequest):
    # SECURITY/VALIDATION: Ensure the user sent the exact right number of features
    if len(request.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400, 
            detail=f"Data shape mismatch. The model expects exactly {EXPECTED_FEATURES} features, but received {len(request.features)}."
        )
    
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
        
    except Exception as e:
        # Catch any other unexpected errors to prevent the server from crashing
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

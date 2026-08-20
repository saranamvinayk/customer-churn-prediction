from fastapi.testclient import TestClient
from src.api import app, EXPECTED_FEATURES

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    # Create a valid dummy payload
    dummy_payload = [0.0] * EXPECTED_FEATURES
    response = client.post("/predict", json={"features": dummy_payload})
    
    assert response.status_code == 200
    assert "prediction_class" in response.json()
    assert "churn_risk" in response.json()

def test_predict_invalid_length():
    # Send too few features to trigger our custom Pydantic/FastAPI error
    bad_payload = [0.0] * (EXPECTED_FEATURES - 1)
    response = client.post("/predict", json={"features": bad_payload})
    
    assert response.status_code == 400

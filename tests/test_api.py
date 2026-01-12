from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_prediction():
    response = client.post("/predict", json={"text": "This movie was amazing!"})
    assert response.status_code == 200

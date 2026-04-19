"""Testes automatizados para o projeto de churn."""

import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.preprocessing import load_and_clean
from src.models.mlp import ChurnMLP


# Test 1: Smoke test — modelo carrega e faz predição
def test_model_smoke():
    """Testa se o modelo carrega e produz output válido."""
    model = ChurnMLP(input_size=30)
    model.eval()
    dummy_input = torch.randn(1, 30)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == torch.Size([])
    assert 0 <= output.item() <= 1


# Test 2: Schema test — validar formato dos dados
def test_data_schema():
    """Testa se o dataset tem as colunas e tipos esperados."""
    df = load_and_clean("data/telco_churn.csv")
    expected_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    for col in expected_cols:
        assert col in df.columns, f"Coluna {col} ausente"
    assert df["Churn"].isin([0, 1]).all(), "Target deve ser 0 ou 1"
    assert df.shape[0] > 5000, "Dataset deve ter mais de 5000 registros"
    assert df.isnull().sum().sum() == 0, "Não deve haver valores nulos"


# Test 3: API test — endpoint retorna 200
def test_api_health():
    """Testa se o endpoint /health retorna status 200."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test 4: API predict test
def test_api_predict():
    """Testa se o endpoint /predict retorna predição válida."""
    client = TestClient(app)
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert 0 <= data["churn_probability"] <= 1

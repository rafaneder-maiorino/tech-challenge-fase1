"""API FastAPI para inferência do modelo de churn."""

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
import torch
from fastapi import FastAPI, Request

from src.api.schemas import CustomerInput, PredictionOutput
from src.data.preprocessing import build_preprocessor, load_and_clean
from src.models.mlp import ChurnMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Variáveis globais do modelo
model = None
preprocessor = None
device = None


def init_model():
    """Carrega modelo e preprocessador."""
    global model, preprocessor, device

    if model is not None:
        return

    logger.info("Carregando modelo e preprocessador...")
    device = torch.device("cpu")

    df = load_and_clean("data/telco_churn.csv")
    X = df.drop("Churn", axis=1)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    n_features = preprocessor.transform(X[:1]).shape[1]
    model = ChurnMLP(n_features)
    model.load_state_dict(torch.load("models/mlp_churn.pt", map_location=device))
    model.eval()
    logger.info("Modelo carregado com sucesso!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle handler — carrega modelo ao iniciar."""
    init_model()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn em telecomunicações usando MLP PyTorch",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    """Middleware para logar latência de cada request."""
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(
        "Método=%s Path=%s Status=%d Latência=%.1fms",
        request.method, request.url.path, response.status_code, duration,
    )
    return response


@app.get("/health")
def health_check():
    """Endpoint de health check."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """Realiza predição de churn para um cliente."""
    init_model()
    input_df = pd.DataFrame([customer.model_dump()])
    input_processed = preprocessor.transform(input_df)
    input_tensor = torch.FloatTensor(input_processed).to(device)

    with torch.no_grad():
        probability = model(input_tensor).item()

    prediction = int(probability >= 0.3)
    risk_level = "alto" if probability >= 0.5 else "médio" if probability >= 0.3 else "baixo"

    return PredictionOutput(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        risk_level=risk_level,
    )

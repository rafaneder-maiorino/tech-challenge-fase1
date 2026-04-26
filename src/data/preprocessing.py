"""Módulo de pré-processamento de dados para previsão de churn."""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

SEED = 42
NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_and_clean(path: str) -> pd.DataFrame:
    """Carrega e limpa o dataset de churn."""
    logger.info("Carregando dataset de %s", path)
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop("customerID", axis=1)
    logger.info("Dataset carregado: %d linhas, %d colunas", df.shape[0], df.shape[1])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Cria o pipeline de pré-processamento."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
        ]
    )
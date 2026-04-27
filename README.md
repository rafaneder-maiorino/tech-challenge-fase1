# Tech Challenge — Fase 1 FIAP

## Previsão de Churn em Telecomunicações com Rede Neural (MLP)

Pipeline end-to-end de Machine Learning para prever churn de clientes
de uma operadora de telecomunicações, utilizando MLP (Multi-Layer Perceptron)
treinada com PyTorch.

## Problema de Negócio

Operadora de telecomunicações perdendo clientes em ritmo acelerado.
O modelo identifica clientes com risco de cancelamento para ações
proativas de retenção, reduzindo custo de churn estimado em R$26.640/mês.

## Resultados

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| DummyClassifier | 0.735 | 0.000 | 0.500 |
| Logistic Regression | 0.808 | 0.603 | 0.843 |
| Random Forest | 0.785 | 0.548 | 0.832 |
| **MLP (PyTorch)** | **0.803** | **0.576** | **0.844** |

Threshold otimizado para 0.3 priorizando Recall (0.74) — captura mais
clientes em risco de churn.

## Estrutura do Projeto

    tech-challenge-fase1/
    ├── src/
    │   ├── data/preprocessing.py       # Pipeline de pré-processamento
    │   ├── models/mlp.py               # Arquitetura MLP PyTorch
    │   ├── models/train.py             # Loop de treinamento
    │   ├── api/main.py                 # API FastAPI
    │   ├── api/schemas.py              # Schemas Pydantic
    │   └── utils/logger.py             # Logging estruturado
    ├── data/                           # Dados (não versionado)
    ├── models/                         # Modelos salvos
    ├── tests/test_model.py             # Testes automatizados
    ├── notebooks/
    │   ├── 01_eda.ipynb                # Análise exploratória
    │   ├── 02_baselines.ipynb          # Baselines + MLflow
    │   └── 03_mlp_pytorch.ipynb        # MLP PyTorch
    ├── docs/
    │   ├── ML_CANVAS.md                # ML Canvas
    │   ├── MODEL_CARD.md               # Model Card
    │   └── MONITORING.md               # Plano de monitoramento
    ├── pyproject.toml                  # Dependências e configuração
    ├── Makefile                        # Comandos de atalho
    └── README.md

## Setup

    git clone https://github.com/rafaneder-maiorino/tech-challenge-fase1.git
    cd tech-challenge-fase1
    python3 -m venv venv
    source venv/bin/activate
    pip install -e ".[dev]"

## Comandos

    make install    # Instalar dependências
    make test       # Rodar testes
    make lint       # Verificar código com ruff
    make run        # Iniciar API

## API

    # Health check
    GET /health

    # Predição
    POST /predict
    Content-Type: application/json

    {
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
        "TotalCharges": 29.85
    }

## Tecnologias

- **PyTorch** — Rede neural MLP
- **Scikit-Learn** — Pipelines e baselines
- **MLflow** — Tracking de experimentos
- **FastAPI** — API de inferência
- **pytest** — Testes automatizados
- **ruff** — Linting

## Dataset

Telco Customer Churn (IBM) — 7.043 clientes, 21 features, classificação binária.

O dataset não está versionado no repositório. Para baixar:

1. Acesse https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Baixe o arquivo `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Renomeie para `telco_churn.csv` e coloque em `data/`

No final, o caminho deve ficar `data/telco_churn.csv` — é assim que os
notebooks e scripts esperam encontrar o arquivo.

## MLflow

Os experimentos do MLflow são gerados ao rodar os notebooks `02_baselines.ipynb`
e `03_mlp_pytorch.ipynb` — cada execução registra métricas, parâmetros e
artefatos na pasta `mlruns/` (criada automaticamente).

Para visualizar a UI:

    mlflow ui

E abra http://localhost:5000 no navegador. Lá dá pra comparar as runs lado a
lado, ver os hiperparâmetros e baixar os modelos salvos.

## Deploy em Nuvem

A API está hospedada no **Azure App Service** e disponível publicamente em:

    https://churn-api-rafael.azurewebsites.net

Endpoints disponíveis:

- `GET /health` — https://churn-api-rafael.azurewebsites.net/health
- `GET /docs` — https://churn-api-rafael.azurewebsites.net/docs (Swagger UI)

## Documentação

- [ML Canvas](docs/ML_CANVAS.md) — Definição do problema
- [Model Card](docs/MODEL_CARD.md) — Performance, limitações e vieses
- [Monitoramento](docs/MONITORING.md) — Plano de monitoramento e deploy

## Autor

Rafael Neder Maiorino — FIAP Pós Tech ML 2026
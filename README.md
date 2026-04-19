# Tech Challenge — Fase 1 FIAP

## Previsão de Churn em Telecomunicações com Rede Neural (MLP)

Pipeline end-to-end de Machine Learning para prever churn de clientes
de uma operadora de telecomunicações, utilizando MLP (Multi-Layer Perceptron)
treinada com PyTorch.

## Estrutura do Projeto

    tech-challenge-fase1/
    ├── src/                    # Código-fonte
    │   ├── data/               # Pré-processamento
    │   ├── models/             # MLP e treinamento
    │   ├── api/                # FastAPI
    │   └── utils/              # Logging e helpers
    ├── data/                   # Dados (não versionado)
    ├── models/                 # Modelos salvos (não versionado)
    ├── tests/                  # Testes automatizados
    ├── notebooks/              # Análises exploratórias
    ├── docs/                   # Documentação
    ├── pyproject.toml          # Dependências e configuração
    ├── Makefile                # Comandos de atalho
    └── README.md               # Este arquivo

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -e ".[dev]"

## Tecnologias

- **PyTorch** — Rede neural MLP
- **Scikit-Learn** — Pipelines e baselines
- **MLflow** — Tracking de experimentos
- **FastAPI** — API de inferência
- **pytest** — Testes automatizados
- **ruff** — Linting

## Dataset

Telco Customer Churn (IBM) — 7.043 clientes, 21 features,
classificação binária.

## Autor

Rafael Neder Maiorino — FIAP Pós Tech ML 2026
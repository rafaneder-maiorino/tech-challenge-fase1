# ML Canvas — Previsão de Churn Telecom

## 1. Problema de Negócio
Operadora de telecomunicações está perdendo clientes em ritmo acelerado.
A diretoria precisa identificar clientes com risco de cancelamento para
ações proativas de retenção.

## 2. Stakeholders
- **Diretoria Comercial:** decisões de retenção e orçamento
- **Equipe de Marketing:** campanhas direcionadas
- **Atendimento ao Cliente:** abordagem proativa

## 3. Dados Disponíveis
- **Fonte:** Dataset Telco Customer Churn (IBM)
- **Volume:** 7.043 clientes, 21 features
- **Tipo:** Dados tabulares (numéricos + categóricos)
- **Target:** Churn (Sim/Não) — classificação binária

## 4. Métricas Técnicas
- **Primária:** AUC-ROC (capacidade de ranquear clientes por risco)
- **Secundárias:** F1-Score, PR-AUC, Recall
- **Justificativa:** Dataset desbalanceado (73/27) → accuracy não é confiável

## 5. Métricas de Negócio
- **Custo de Churn:** Perda de receita mensal do cliente (~R$65/mês médio)
- **Custo de Retenção:** Desconto ou benefício oferecido (~R$20/mês)
- **Meta:** Reduzir churn em 15-20% com ações direcionadas

## 6. Trade-off de Erros
- **Falso Positivo:** Oferecer desconto para quem não ia cancelar → custo baixo
- **Falso Negativo:** Perder cliente que ia cancelar → custo ALTO
- **Decisão:** Priorizar Recall (capturar mais churners) aceitando mais FPs

## 7. SLOs (Service Level Objectives)
- **Latência:** < 200ms por predição
- **Disponibilidade:** 99% uptime
- **Atualização:** Re-treino mensal com dados novos

## 8. Abordagem Técnica
- **Baseline:** DummyClassifier + Regressão Logística (Scikit-Learn)
- **Modelo Principal:** MLP (PyTorch) com early stopping
- **Pipeline:** Pré-processamento com Scikit-Learn Pipeline
- **Tracking:** MLflow para todos os experimentos
- **Serving:** FastAPI (POST /predict)
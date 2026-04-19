# Model Card — Churn Prediction MLP

## Informações Gerais
- **Nome:** ChurnMLP v1.0
- **Tipo:** Multi-Layer Perceptron (classificação binária)
- **Framework:** PyTorch 2.8.0
- **Autor:** Rafael Neder Maiorino
- **Data:** Abril 2026

## Uso Pretendido
Prever a probabilidade de um cliente de telecomunicações cancelar o serviço (churn),
permitindo ações proativas de retenção pela equipe comercial.

## Dados de Treinamento
- **Dataset:** Telco Customer Churn (IBM)
- **Volume:** 7.043 clientes, 19 features após pré-processamento
- **Split:** 80% treino / 20% teste (estratificado)
- **Desbalanceamento:** 73.5% não-churn vs 26.5% churn

## Arquitetura
- Input (30 features) → Linear(64) → ReLU → Dropout(0.3) → Linear(32) → ReLU → Dropout(0.3) → Linear(1) → Sigmoid
- Otimizador: Adam (lr=0.001)
- Loss: BCELoss
- Early stopping: patience=10
- Total de parâmetros: 4.097

## Performance

| Modelo | Accuracy | F1-Score | AUC-ROC | Precision | Recall |
|--------|----------|----------|---------|-----------|--------|
| DummyClassifier | 0.735 | 0.000 | 0.500 | 0.000 | 0.000 |
| Logistic Regression | 0.808 | 0.603 | 0.843 | 0.656 | 0.559 |
| Random Forest | 0.785 | 0.548 | 0.832 | 0.623 | 0.489 |
| **MLP (PyTorch)** | **0.803** | **0.576** | **0.844** | **0.674** | **0.503** |

## Threshold Otimizado
- Threshold padrão: 0.5
- Threshold otimizado por custo: 0.3 (prioriza Recall para capturar mais churners)
- Com threshold 0.3: Recall sobe para 0.74, custo total reduz de R$39.930 para R$26.640

## Limitações
- Treinado apenas com dados históricos de uma operadora; pode não generalizar para outras
- Não captura mudanças temporais (sazonalidade, promoções da concorrência)
- Features limitadas a dados cadastrais e de contrato; não inclui dados de uso (ligações, dados móveis)
- Dataset relativamente pequeno (7.043 registros)
- Desbalanceamento pode afetar predições em cenários com distribuição diferente

## Vieses Conhecidos
- Modelo pode ter viés em relação a Senior Citizens (idosos cancelam mais nos dados)
- Clientes com Fiber Optic aparecem com mais churn, mas pode ser correlação com preço e não causalidade
- Payment Method (Electronic Check) correlacionado com churn, mas pode refletir perfil socioeconômico

## Cenários de Falha
- Clientes novos (tenure=0): pouca informação histórica para predição confiável
- Mudanças bruscas no mercado (novo concorrente, crise econômica) invalidam o modelo
- Clientes com comportamento atípico (ex: contas corporativas em dataset residencial)

## Recomendações
- Re-treinar mensalmente com dados atualizados
- Monitorar data drift nas features principais (tenure, MonthlyCharges, Contract)
- Usar threshold 0.3 para ações de retenção (priorizar capturar churners)
- Combinar predição do modelo com análise humana para casos de alto valor
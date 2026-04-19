# Model Card — Churn Prediction MLP

Documentei aqui o que considero importante saber sobre o modelo que treinei.
Não é um template completo de model card "de produção", mas tentei cobrir os
pontos que mais me ajudaram a entender os limites do que construí.

## Informações Gerais
- **Nome:** ChurnMLP v1.0
- **Tipo:** Multi-Layer Perceptron (classificação binária)
- **Framework:** PyTorch 2.8.0
- **Autor:** Rafael Neder Maiorino
- **Data:** Abril 2026

## Uso Pretendido
A ideia é prever a probabilidade de um cliente de telecom cancelar o serviço
(churn). Com essa probabilidade em mãos, a equipe comercial consegue priorizar
quem abordar com ações de retenção. Não pensei esse modelo para outras decisões
(ex.: precificação ou crédito) — fora do escopo pra que foi treinado.

## Dados de Treinamento
- **Dataset:** Telco Customer Churn (IBM)
- **Volume:** 7.043 clientes, 19 features após pré-processamento
- **Split:** 80% treino / 20% teste (estratificado)
- **Desbalanceamento:** 73.5% não-churn vs 26.5% churn

Percebi logo no começo que o desbalanceamento ia ser um problema pra interpretar
accuracy, então dei mais peso a AUC-ROC e F1 ao longo dos experimentos.

## Arquitetura
- Input (30 features) → Linear(64) → ReLU → Dropout(0.3) → Linear(32) → ReLU → Dropout(0.3) → Linear(1) → Sigmoid
- Otimizador: Adam (lr=0.001)
- Loss: BCELoss
- Early stopping: patience=10
- Total de parâmetros: 4.097

Optei por uma rede pequena de propósito. Testei versões maiores no começo e o
ganho foi marginal, então preferi algo que treina rápido, cabe na cabeça e é
fácil de depurar. O dropout de 0.3 ajudou a estabilizar a validação — sem ele
via um overfitting claro já na época 15.

## Performance

| Modelo | Accuracy | F1-Score | AUC-ROC | Precision | Recall |
|--------|----------|----------|---------|-----------|--------|
| DummyClassifier | 0.735 | 0.000 | 0.500 | 0.000 | 0.000 |
| Logistic Regression | 0.808 | 0.603 | 0.843 | 0.656 | 0.559 |
| Random Forest | 0.785 | 0.548 | 0.832 | 0.623 | 0.489 |
| **MLP (PyTorch)** | **0.803** | **0.576** | **0.844** | **0.674** | **0.503** |

Sendo honesto: a Regressão Logística chegou praticamente no mesmo lugar que a
MLP. Acredito que com um dataset tabular desse tamanho e com features bem
pré-processadas, o ganho de uma rede neural é pequeno. Deixei a MLP como modelo
principal porque o exercício é justamente esse, mas em produção eu olharia com
carinho pro custo-benefício de usar a logística.

## Threshold Otimizado
- Threshold padrão: 0.5
- Threshold otimizado por custo: 0.3 (prioriza Recall para capturar mais churners)
- Com threshold 0.3: Recall sobe para 0.74, custo total reduz de R$39.930 para R$26.640

Essa parte foi uma das mais legais do trabalho pra mim. Quando montei a matriz
de custo e comecei a varrer thresholds, ficou claro que o 0.5 default é quase
sempre uma escolha ruim quando os erros têm custos diferentes. Vou levar essa
lição pros próximos projetos.

## Limitações
- Treinei só com dados históricos de uma operadora; não espero que generalize pra outras realidades
- Nada captura efeitos temporais (sazonalidade, promoções da concorrência, etc.)
- As features são cadastrais e de contrato; não tenho nada de uso real (minutagem, dados, reclamações)
- O dataset é pequeno (7.043 linhas) — com mais dados talvez a MLP se destacasse mais da logística
- Se a distribuição mudar muito, as predições podem ficar descalibradas rápido

## Vieses Conhecidos
- Senior Citizens aparecem com churn mais alto — o modelo provavelmente está pegando isso
- Clientes com Fiber Optic também aparecem com mais churn, mas desconfio que seja correlação com preço, não causalidade
- Payment Method = Electronic Check aparece ligado a churn, o que pode refletir mais perfil socioeconômico do que um sinal real

Levantei esses pontos porque me preocupa que decisões de retenção acabem
reforçando vieses indesejados. Não resolvi isso neste trabalho, mas registrei
pra não esquecer.

## Cenários de Falha
- Clientes novos (tenure=0): pouca história, predição pouco confiável
- Mudanças bruscas de mercado (novo concorrente, crise) derrubam o modelo
- Clientes atípicos (ex.: contas corporativas num dataset residencial)

## Recomendações
- Re-treinar mensalmente com dados atualizados
- Monitorar drift nas features que mais influenciam (tenure, MonthlyCharges, Contract)
- Usar threshold 0.3 quando o objetivo for acionar retenção
- Para clientes de alto valor, combinar predição com análise humana — não confiaria no modelo sozinho nessa faixa

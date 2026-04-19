# ML Canvas — Previsão de Churn Telecom

Montei esse canvas no começo do projeto pra não sair direto pro código sem
pensar no problema. Serviu mais pra eu me organizar do que pra entregar um
artefato bonito — mantive ele curto de propósito.

## 1. Problema de Negócio
Operadora de telecom está perdendo clientes em ritmo acelerado. A diretoria
quer identificar quem tem risco de cancelar pra agir antes. O "valor" aqui não
está em acertar quem vai sair — está em acertar *a tempo* de fazer alguma coisa.

## 2. Stakeholders
- **Diretoria Comercial:** decisões de retenção e orçamento
- **Equipe de Marketing:** campanhas direcionadas
- **Atendimento ao Cliente:** abordagem proativa

## 3. Dados Disponíveis
- **Fonte:** Dataset Telco Customer Churn (IBM)
- **Volume:** 7.043 clientes, 21 features
- **Tipo:** Dados tabulares (numéricos + categóricos)
- **Target:** Churn (Sim/Não) — classificação binária

Percebi durante a EDA que o dataset é bem "limpo" — quase não tem missing,
categorias bem definidas. Isso é ótimo pra aprender, mas me deixa desconfiado
de que dados reais seriam muito mais bagunçados.

## 4. Métricas Técnicas
- **Primária:** AUC-ROC (capacidade de ranquear clientes por risco)
- **Secundárias:** F1-Score, PR-AUC, Recall
- **Justificativa:** Dataset desbalanceado (73/27) → accuracy não é confiável

Escolhi AUC-ROC como principal porque o caso de uso é ranquear clientes por
risco, não classificar um-a-um. Faz mais sentido perguntar "esse cliente está
entre os 20% mais arriscados?" do que "churn sim ou não?".

## 5. Métricas de Negócio
- **Custo de Churn:** Perda de receita mensal do cliente (~R$65/mês médio)
- **Custo de Retenção:** Desconto ou benefício oferecido (~R$20/mês)
- **Meta:** Reduzir churn em 15-20% com ações direcionadas

Os valores são estimativas minhas, não vieram da operadora — usei pra conseguir
montar uma análise de custo mais concreta. Acredito que numa situação real o
próprio negócio informaria esses números.

## 6. Trade-off de Erros
- **Falso Positivo:** Oferecer desconto pra quem não ia cancelar → custo baixo
- **Falso Negativo:** Perder cliente que ia cancelar → custo ALTO
- **Decisão:** Priorizar Recall (capturar mais churners) aceitando mais FPs

Essa foi uma decisão consciente e acabou orientando várias outras escolhas
(threshold 0.3, uso de F1 e Recall pra comparar modelos).

## 7. SLOs (Service Level Objectives)
- **Latência:** < 200ms por predição
- **Disponibilidade:** 99% uptime
- **Atualização:** Re-treino mensal com dados novos

## 8. Abordagem Técnica
- **Baseline:** DummyClassifier + Regressão Logística (Scikit-Learn)
- **Modelo Principal:** MLP (PyTorch) com early stopping
- **Pipeline:** Pré-processamento com Scikit-Learn Pipeline
- **Tracking:** MLflow pra todos os experimentos
- **Serving:** FastAPI (POST /predict)

Comecei pelos baselines de propósito — queria ter uma régua antes de tentar
qualquer coisa "fancy". Isso me salvou de achar que a MLP era brilhante quando,
no fim das contas, a logística quase empata com ela.

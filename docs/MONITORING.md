# Plano de Monitoramento — Churn Prediction

Esse é o plano de monitoramento que esbocei pro modelo. Confesso que nunca
coloquei um modelo de verdade em produção ainda, então muita coisa aqui é
baseada em leitura e no que aprendi nas aulas — fui descrevendo o que *faria*
se tivesse que monitorar esse serviço de verdade.

## 1. Métricas de Modelo
| Métrica | Baseline | Alerta | Ação |
|---------|----------|--------|------|
| AUC-ROC | 0.844 | < 0.80 | Re-treinar modelo |
| F1-Score | 0.576 | < 0.50 | Investigar data drift |
| Recall (threshold 0.3) | 0.743 | < 0.65 | Revisar threshold |
| Latência /predict | < 200ms | > 500ms | Investigar infraestrutura |

Os limiares de alerta eu escolhi meio no chute informado: peguei o baseline e
coloquei uma margem que me pareceu razoável pra diferenciar ruído estatístico
de queda real. Acredito que na prática esses números precisariam ser calibrados
observando a variância natural da métrica por algumas semanas.

## 2. Métricas de Dados (Data Drift)
| Feature | Monitorar | Método |
|---------|-----------|--------|
| tenure | Distribuição | KS test mensal |
| MonthlyCharges | Média e desvio | Z-score |
| Contract | Proporção categorias | Chi-squared test |
| Churn rate | Proporção target | Comparar com treino (26.5%) |

Escolhi essas quatro porque foram as que mais apareceram nas análises de
importância (olhando coeficientes da logística e o comportamento da MLP).
Faz sentido priorizar drift justamente onde o modelo é mais sensível.

## 3. Métricas de Infraestrutura
- **Disponibilidade:** uptime > 99%
- **Latência P95:** < 200ms
- **Taxa de erro:** < 1% das requisições
- **Uso de memória:** < 80% do disponível

## 4. Frequência de Monitoramento
| Verificação | Frequência |
|-------------|-----------|
| Latência e erros da API | Tempo real |
| Volume de requisições | Diário |
| Métricas do modelo (AUC, F1) | Semanal |
| Data drift nas features | Mensal |
| Re-treino completo | Mensal ou quando alertar |

Optei por semanal para métricas de modelo (e não diário) porque em volumes
pequenos a variância dia-a-dia atrapalha mais do que ajuda. Monitoramento muito
barulhento gera fadiga de alerta — algo que li bastante sobre e quero evitar.

## 5. Playbook de Resposta

### Alerta: AUC-ROC caiu abaixo de 0.80
1. Verificar se houve mudança na distribuição dos dados de entrada
2. Comparar features atuais com distribuição de treino (KS test)
3. Se data drift confirmado: re-treinar com dados dos últimos 3 meses
4. Validar novo modelo com holdout antes de deploy
5. Registrar novo experimento no MLflow

### Alerta: Latência acima de 500ms
1. Verificar uso de CPU/memória do servidor
2. Verificar tamanho do payload de entrada
3. Verificar se há contenção no carregamento do modelo
4. Escalar horizontalmente se necessário

### Alerta: Taxa de churn real diverge da prevista
1. Comparar predições do modelo com churn real (quando disponível)
2. Calcular calibration curve
3. Se descalibrado: ajustar threshold ou re-treinar
4. Documentar no MLflow

Percebi escrevendo esse playbook que a parte mais difícil na prática
provavelmente é o passo "comparar com churn real" — o ground truth só chega
depois que o cliente cancela (ou não cancela), então existe um delay natural
que complica o feedback loop. Fica a observação.

## 6. Arquitetura de Deploy
- **Escolha:** Real-time (API REST)
- **Justificativa:** O negócio precisa de predições imediatas para ações de retenção no momento do contato com o cliente (call center, chat). Batch até serviria para campanhas de marketing, mas real-time atende os dois cenários. Optei por esse caminho pensando também que é mais interessante pro aprendizado do que só gerar um CSV.
- **Stack:** FastAPI + Uvicorn
- **Modelo:** Carregado em memória no startup da API

# Roteiro da Apresentação e Guia de Defesa Técnica
**Projeto:** Previsão de Inadimplência em Cartão de Crédito com Redes Neurais Tabulares (PyTorch)  
**Data:** 23 de novembro de 2026

---

## Divisão Sugerida para Apresentação (4 Integrantes — ~10 a 12 min)
- **Integrante 1 (Slides 1–3):** Problema, Contexto Financeiro, Dataset e EDA
- **Integrante 2 (Slides 4–5):** Pré-processamento, Feature Engineering e Prevenção de *Data Leakage*
- **Integrante 3 (Slides 6–7):** Arquiteturas Neurais (Baseline MLP vs. Regularized MLP vs. Tabular ResNet) e Treinamento
- **Integrante 4 (Slides 8–10):** Resultados, Curvas, Otimização por Custo Financeiro, Limitações e Conclusão

---

## Estrutura dos Slides (.pptx / Google Slides)

1. **Slide 1 — Capa**
   - Título: *Previsão de Inadimplência em Cartão de Crédito com Redes Neurais Artificiais*
   - Subtítulo, nomes dos integrantes e data (23/11/2026).

2. **Slide 2 — Definição do Problema e Relevância Financeira**
   - Tarefa: Classificação Binária Supervisionada (`default = 1` vs. `adimplente = 0`).
   - Assimetria de erro no crédito: Falso Negativo (prejuízo do principal) custa muito mais que Falso Positivo (custo de oportunidade).
   - Objetivo: Estimar probabilidade de inadimplência e calibrar *threshold* por custo financeiro.

3. **Slide 3 — Dataset e Análise Exploratória (EDA)**
   - Dataset: *Default of Credit Card Clients* (UCI ID 350, Yeh & Lien 2009) — 30.000 clientes, 23 atributos.
   - Principais achados (colocar 2 gráficos do notebook):
     - Desbalanceamento: 22,12% inadimplentes (acurácia isolada é enganosa).
     - Poder preditivo de `PAY_0`: atraso >= 2 meses dispara taxa de inadimplência (>65%).
     - Outliers em `LIMIT_BAL` / `PAY_AMT` e códigos não documentados em `EDUCATION` (`0,5,6`) e `MARRIAGE` (`0`).

4. **Slide 4 — Pré-processamento e Feature Engineering**
   - Remoção de 35 duplicatas exatas.
   - Saneamento de `EDUCATION` e `MARRIAGE` na categoria `Outros`.
   - **Novas features financeiras criadas pelo grupo:** `UTIL_RATIO_1`, `UTIL_RATIO_MEAN`, `PAY_TO_BILL_1`, `TOTAL_PAY_6M`, `MAX_DELAY`, `NUM_DELAYS`, `PAY_TREND`.

5. **Slide 5 — Divisão dos Dados e Prevenção de *Data Leakage***
   - Divisão estratificada: **70% Treino / 15% Validação / 15% Teste**.
   - `RobustScaler` (numéricas, resistente a outliers) + `OneHotEncoder` (categóricas).
   - Regra de ouro: `fit()` **apenas no Treino**; Validação e Teste recebem apenas `transform()`.

6. **Slide 6 — Arquiteturas Neurais Comparadas (PyTorch)**
   - **Exp 1 — Baseline MLP:** `Linear(35→64→32→1)` + `ReLU`, sem regularização, `BCEWithLogitsLoss` padrão.
   - **Exp 2 — Regularized MLP:** `128→64→32→1` + `BatchNorm1d` + `GELU` + `Dropout(0.3)` + `AdamW` + `pos_weight`.
   - **Exp 3 — Tabular ResNet:** Projeção + 2 Blocos Residuais (`BatchNorm → SiLU → Dropout → Linear` com *skip connection*).

7. **Slide 7 — Hiperparâmetros e Estratégia de Treinamento**
   - Otimizador: `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) + `ReduceLROnPlateau`.
   - Estabilidade: *Gradient Clipping* (`max_norm=2.0`) e `EarlyStopping` (paciência=10 na validação).
   - Tratamento de desbalanceamento: `pos_weight = N_neg / N_pos (~3.52)` na `BCEWithLogitsLoss`.

8. **Slide 8 — Avaliação: Curvas de Treino vs. Validação e Tabela Comparativa**
   - Inserir gráfico das curvas de Loss e ROC-AUC (mostrar controle de *overfitting* nos Exp 2 e 3).
   - Inserir tabela comparativa no Teste: `ROC-AUC`, `PR-AUC`, `F1-Score`, `Recall`, `Precision`.

9. **Slide 9 — Otimização de *Threshold* por Custo Financeiro**
   - Mostrar gráfico de Custo Esperado (`5*FN + 1*FP`) na Validação + Matriz de Confusão no Teste.
   - Destacar o ganho prático: o *threshold* ótimo difere de `0.50` justamente por causa da assimetria de custo e do uso de `pos_weight`.

10. **Slide 10 — Conclusão, Limitações e Próximos Passos**
    - Síntese dos resultados e atendimento ao objetivo.
    - Limitações: dados de 2005 (Taiwan), sem renda declarada, menor explicabilidade nativa.
    - Melhorias futuras: SHAP para explicabilidade regulatória, *Entity Embeddings* / *FT-Transformer*, calibração de probabilidades (*Platt Scaling*).

---

## Guia de Defesa Técnica (Respostas para Perguntas da Banca)

1. **"Por que escolheram MLP e Tabular ResNet em vez de CNN ou LSTM?"**
   - CNNs exploram localidade espacial (imagens) e LSTMs exploram sequências longas. Nossos dados são tabulares heterogêneos (35 features após *encoding*). A `TabularResNet` usa *skip connections* para preservar sinais lineares fortes (ex: `PAY_0`) enquanto aprende interações não lineares profundas sem degradação de gradiente.

2. **"Por que usar `RobustScaler` em vez de `StandardScaler` ou `MinMaxScaler`?"**
   - Variáveis financeiras (`LIMIT_BAL`, `BILL_AMT`, `PAY_AMT`) têm caudas longas e outliers extremos. `MinMaxScaler` comprimiria quase todos os clientes perto de zero; `StandardScaler` teria média/desvio distorcidos. O `RobustScaler` usa mediana e intervalo interquartil (IQR), sendo imune a extremos.

3. **"Por que `BCEWithLogitsLoss` em vez de `Sigmoid` na última camada + `BCELoss`?"**
   - `BCEWithLogitsLoss` combina `Sigmoid` e `Binary Cross-Entropy` internamente usando o truque *log-sum-exp*, evitando *overflow/underflow* numérico nos gradientes.

4. **"Qual o efeito do `pos_weight` e por que precisaram calibrar o *threshold* depois?"**
   - O `pos_weight (~3.52)` multiplica o gradiente dos exemplos positivos (inadimplentes), combatendo o desbalanceamento e aumentando o *Recall*. Como isso desloca as probabilidades previstas para cima, o limiar padrão `0.50` deixa de ser ótimo; por isso calibramos o *threshold* na validação minimizando o custo `5*FN + 1*FP`.

5. **"Como garantiram que não houve *overfitting* nem *data leakage*?"**
   - *Data leakage*: duplicatas removidas antes do split; `RobustScaler`, `OneHotEncoder`, `pos_weight` e `opt_threshold` calculados **exclusivamente** no treino/validação, nunca no teste.
   - *Overfitting*: monitoramos a curva de validação com `BatchNorm1d`, `Dropout(0.30)`, `weight_decay` no `AdamW` e `EarlyStopping`.

6. **"Por que `ReLU` no baseline e `GELU`/`SiLU` nas redes seguintes?"**
   - `ReLU` zera gradientes para entradas negativas (*dying ReLU*). `GELU` e `SiLU` (*Swish*) são suaves e não-monotônicas próximas a zero, melhorando o fluxo de gradiente em redes tabulares com `BatchNorm`.

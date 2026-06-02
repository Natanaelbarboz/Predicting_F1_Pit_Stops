# 🏎️ Predicting F1 Pit Stops - Kaggle Competition

## 📌 Sobre o Projeto

Este projeto foi desenvolvido para a competição **Predicting F1 Pit Stops** do Kaggle.

O objetivo é prever a probabilidade de um piloto realizar um **pit stop na próxima volta**, utilizando informações relacionadas à corrida, pneus e condições da prova.

O modelo foi desenvolvido utilizando técnicas de Machine Learning supervisionado e alcançou um score de:

**🏆 Public Score: 0.94776**

---

## 🎯 Objetivo

Construir um modelo capaz de identificar a probabilidade de ocorrência de um pit stop na volta seguinte, auxiliando na tomada de decisão estratégica durante uma corrida.

---

## 📊 Métrica de Avaliação

A competição utiliza como métrica principal:

* ROC AUC (Receiver Operating Characteristic - Area Under Curve)

Resultado obtido:

| Métrica             | Valor   |
| ------------------- | ------- |
| ROC AUC Local       | 0.95    |
| Public Score Kaggle | 0.94776 |

---

## 🛠️ Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Yellowbrick
* Joblib
* Matplotlib

---

## 📂 Estrutura do Projeto

```text
📦 Predicting-F1-Pit-Stops
│
├── 0 - Bases
    ├── train.csv
    ├── test.csv
├── 1 - Scripts
    ├── f1_predicting_v2.py
    ├── pipeline_modelo_f1_predicting_v2.pkl
    ├── Predicoes.csv
│
├── requirements.txt
└── README.md
```

---

## 🔍 Pré-processamento

As seguintes etapas foram aplicadas aos dados:

### Tratamento de Variáveis Categóricas

Foi utilizado One-Hot Encoding para as colunas:

* Race
* Compound

```python
pd.get_dummies()
```

### Remoção de Colunas

As colunas abaixo foram removidas por não serem utilizadas no treinamento:

* id
* Driver

---

## 🤖 Modelo Utilizado

Foi utilizado o algoritmo:

### XGBoost Classifier

Principais hiperparâmetros:

```python
XGBClassifier(
    n_estimators=7000,
    learning_rate=0.01,
    max_depth=12,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1
)
```

---

## ⚖️ Tratamento de Classes Desbalanceadas

Para lidar com o desbalanceamento entre as classes, foi utilizada a estratégia:

```python
scale_pos_weight
```

Calculada automaticamente através da proporção entre exemplos negativos e positivos presentes no conjunto de treinamento.

---

## 📈 Validação do Modelo

### Divisão dos Dados

```python
train_test_split(
    test_size=0.3,
    stratify=y
)
```

### Validação Cruzada

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True
)
```

A utilização do Stratified K-Fold garante que a distribuição das classes seja preservada em todas as partições de validação.

---

## 📉 Avaliação

### Matriz de Confusão

A matriz de confusão foi utilizada para analisar:

* Verdadeiros Positivos
* Verdadeiros Negativos
* Falsos Positivos
* Falsos Negativos

### Curva ROC

A Curva ROC foi utilizada para avaliar a capacidade discriminativa do modelo.

Resultado obtido:

**AUC ≈ 0.95**

---

## 💾 Persistência do Modelo

Após o treinamento, o modelo foi salvo utilizando Joblib:

```python
joblib.dump(
    pipeline,
    "pipeline_modelo_f1_predicting_v2.pkl"
)
```

O pipeline contém:

* Modelo treinado
* Lista de features utilizadas

---

## 🚀 Geração de Previsões

Após carregar o modelo salvo:

```python
model.predict_proba(X)
```

As probabilidades geradas são exportadas para:

```text
Predicoes.csv
```

Formato:

| id | PitNextLap |
| -- | ---------- |
| 1  | 0.91       |
| 2  | 0.04       |
| 3  | 0.87       |

---

## 📌 Principais Aprendizados

* Tratamento de dados categóricos
* Classificação binária com XGBoost
* Balanceamento de classes
* Validação cruzada estratificada
* Avaliação utilizando ROC AUC
* Persistência de modelos para produção
* Construção de pipelines de inferência

---

## 🔗 Competição

Predicting F1 Pit Stops - Kaggle Playground Series S6E5

---

## 👨‍💻 Autor

Natanael Lima

Profissional da área de Dados com foco em Machine Learning, Analytics e desenvolvimento de soluções baseadas em dados.

LinkedIn: [https://www.linkedin.com/in/natanael-barboza/]
Kaggle: [https://www.kaggle.com/natanaelblima]
GitHub: [https://github.com/Natanaelbarboz]

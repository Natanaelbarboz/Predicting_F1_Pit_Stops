# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    roc_auc_score
)

from yellowbrick.classifier import (
    ConfusionMatrix,
    ROCAUC
)

from xgboost import XGBClassifier

# =============================================================================
# PREPROCESSAMENTO
# =============================================================================

def preprocessar_dados(df):

    print('Processando dados...')

    df = df.copy()

    # -------------------------------------------------------------------------
    # ONE HOT ENCODING
    # -------------------------------------------------------------------------

    colunas_categoricas = [
        'Race',
        'Compound'
    ]

    df = pd.get_dummies(
        df,
        columns=colunas_categoricas,
        dummy_na=False
    )

    return df


# =============================================================================
# TREINAMENTO
# =============================================================================

def treinar_modelo(base_treino):

    print('Carregando base de treino')

    df = pd.read_csv(base_treino)

    # -------------------------------------------------------------------------
    # TARGET
    # -------------------------------------------------------------------------

    y = df['PitNextLap']

    # -------------------------------------------------------------------------
    # REMOVER COLUNAS
    # -------------------------------------------------------------------------

    X = df.drop(
        columns=[
            'PitNextLap',
            'id',
            'Driver',
        ]
    )

    # -------------------------------------------------------------------------
    # PREPROCESSAMENTO
    # -------------------------------------------------------------------------

    X = preprocessar_dados(X)

    # -------------------------------------------------------------------------
    # SPLIT
    # -------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # -------------------------------------------------------------------------
    # BALANCEAMENTO
    # -------------------------------------------------------------------------

    negativos = (y_train == 0).sum()
    positivos = (y_train == 1).sum()

    scale_pos_weight = negativos / positivos

    print(f'Scale Pos Weight: {scale_pos_weight:.2f}')

    # -------------------------------------------------------------------------
    # MODELO
    # -------------------------------------------------------------------------

    print('Treinando modelo XGBoost')

    model = XGBClassifier(
        n_estimators=7000,
        learning_rate=0.01,
        max_depth=10,
        min_child_weight=7,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=200
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    # -------------------------------------------------------------------------
    # AUC LOCAL
    # -------------------------------------------------------------------------

    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)

    print(f'\nROC AUC LOCAL: {auc:.5f}')

    # -------------------------------------------------------------------------
    # VALIDAÇÃO CRUZADA
    # -------------------------------------------------------------------------

    print('\nExecutando validação cruzada')

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        XGBClassifier(
            n_estimators=7000,
            learning_rate=0.01,
            max_depth=10,
            min_child_weight=7,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        ),
        X,
        y,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    print(f'\nAUC Médio CV: {scores.mean():.5f}')
    print(f'Desvio padrão: {scores.std():.5f}')

    # -------------------------------------------------------------------------
    # MATRIZ DE CONFUSÃO
    # -------------------------------------------------------------------------

    mapping = {
        0: 'NÃO',
        1: 'SIM'
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    cm_viz = ConfusionMatrix(
        model,
        classes=['NÃO', 'SIM'],
        label_encoder=mapping,
        ax=ax
    )

    cm_viz.fit(X_train, y_train)
    cm_viz.score(X_test, y_test)
    cm_viz.show()

    # -------------------------------------------------------------------------
    # ROC CURVE
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 6))

    roc_viz = ROCAUC(model, ax=ax)

    roc_viz.fit(X_train, y_train)
    roc_viz.score(X_test, y_test)
    roc_viz.show()

    # -------------------------------------------------------------------------
    # SALVAR PIPELINE
    # -------------------------------------------------------------------------

    pipeline = {
        'model': model,
        'features': X.columns.tolist()
    }

    joblib.dump(
        pipeline,
        'pipeline_modelo_f1_predicting_v2.pkl'
    )

    print('\nModelo salvo com sucesso')


# =============================================================================
# PREVISÃO
# =============================================================================

def prever_novos_dados(base_teste):

    print('\nCarregando novos dados')

    df_teste = pd.read_csv(base_teste)

    ids = df_teste['id'].values

    # -------------------------------------------------------------------------
    # REMOVER COLUNAS
    # -------------------------------------------------------------------------

    X = df_teste.drop(
        columns=[
            'id',
            'Driver'
        ]
    )

    # -------------------------------------------------------------------------
    # PREPROCESSAMENTO
    # -------------------------------------------------------------------------

    X = preprocessar_dados(X)

    # -------------------------------------------------------------------------
    # CARREGAR MODELO
    # -------------------------------------------------------------------------

    pipeline = joblib.load(
        'pipeline_modelo_f1_predicting_v2.pkl'
    )

    model = pipeline['model']

    features_treinamento = pipeline['features']

    # -------------------------------------------------------------------------
    # AJUSTAR COLUNAS
    # -------------------------------------------------------------------------

    for coluna in features_treinamento:

        if coluna not in X.columns:
            X[coluna] = 0

    # Garantir mesma ordem
    X = X[features_treinamento]

    # -------------------------------------------------------------------------
    # PREDIÇÃO
    # -------------------------------------------------------------------------

    print('Gerando probabilidades')

    probs = model.predict_proba(X)[:, 1]

    # -------------------------------------------------------------------------
    # RESULTADO
    # -------------------------------------------------------------------------

    resultado = pd.DataFrame({
        'id': ids,
        'PitNextLap': probs
    })

    resultado.to_csv(
        'Predicoes.csv',
        index=False
    )

    print('\nArquivo Predicoes.csv salvo com sucesso')

    print(resultado.head())

    return resultado


# =============================================================================
# EXECUÇÃO
# =============================================================================

base_treino = '../0 - Bases/train.csv'
base_teste = '../0 - Bases/test.csv'

treinar_modelo(base_treino)

resultado = prever_novos_dados(base_teste)
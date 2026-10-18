
# ============================================
# PROYECTO: INTENCION DE COMPRA EN LINEA
# Grupo 5 - Tecnologias Emergentes
# Etapa 3: Modelado
# ============================================

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.datos import crear_preprocesador

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def crear_modelo_base(X_train):
    """
    Crea un modelo base que predice la clase mas frecuente.
    Sirve como referencia para comparar modelos posteriores.
    """

    preprocesador = crear_preprocesador(X_train)

    modelo = Pipeline([
        ("preprocesamiento", preprocesador),
        ("clasificador", DummyClassifier(strategy="most_frequent"))
    ])

    return modelo


def crear_modelo_regresion_logistica(X_train):
    """
    Crea un modelo de regresion logistica
    para predecir la intencion de compra.
    """

    preprocesador = crear_preprocesador(X_train)

    modelo = Pipeline([
        ("preprocesamiento", preprocesador),
        ("clasificador", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    return modelo


def crear_red_neuronal_v1(numero_variables):
    """
    Primera version de la red neuronal.
    Recibe variables ya preprocesadas.
    """

    modelo = keras.Sequential([
        keras.Input(shape=(numero_variables,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return modelo


def crear_red_neuronal_v2(numero_variables):
    """
    Segunda version de la red neuronal.
    Usa dos capas ocultas y regularizacion Dropout.
    """

    modelo = keras.Sequential([
        keras.Input(shape=(numero_variables,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return modelo


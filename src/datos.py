
# ============================================
# PROYECTO: INTENCION DE COMPRA EN LINEA
# Grupo 5 - Tecnologias Emergentes
# Etapa 2: Datos y exploracion
# ============================================

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Rutas del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATASET = RUTA_PROYECTO / "data" / "online_shoppers_intention.csv"


def cargar_datos():
    """Carga el dataset original."""

    if not RUTA_DATASET.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset: {RUTA_DATASET}"
        )

    return pd.read_csv(RUTA_DATASET)


def explorar_datos(datos):
    """Muestra informacion sobre la calidad del dataset."""

    print("\n========== DIMENSIONES ==========")
    print(f"Filas: {datos.shape[0]}")
    print(f"Columnas: {datos.shape[1]}")

    print("\n========== VALORES AUSENTES ==========")
    print(datos.isnull().sum())

    print("\n========== FILAS DUPLICADAS ==========")
    print(f"Duplicados exactos: {datos.duplicated().sum()}")

    print("\n========== DISTRIBUCION DE REVENUE ==========")
    print(datos["Revenue"].value_counts())
    print(
        datos["Revenue"].value_counts(normalize=True) * 100
    )


def dividir_datos(datos):
    """
    Separa los datos en entrenamiento, validacion y prueba.
    Mantiene la proporcion de las clases con estratificacion.
    """

    # Separar variables predictoras y variable objetivo
    X = datos.drop(columns=["Revenue"])
    y = datos["Revenue"]

    # 70 % entrenamiento y 30 % temporal
    X_train, X_temporal, y_train, y_temporal = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Dividir el 30 % temporal en 15 % validacion y 15 % prueba
    X_val, X_test, y_val, y_test = train_test_split(
        X_temporal,
        y_temporal,
        test_size=0.50,
        random_state=42,
        stratify=y_temporal
    )

    print("\n========== DIVISION DE DATOS ==========")
    print(f"Entrenamiento: {len(X_train)} registros")
    print(f"Validacion:    {len(X_val)} registros")
    print(f"Prueba:        {len(X_test)} registros")

    print("\nProporcion de compras por conjunto:")
    print(f"Entrenamiento: {y_train.mean():.4f}")
    print(f"Validacion:    {y_val.mean():.4f}")
    print(f"Prueba:        {y_test.mean():.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def crear_preprocesador(X_train):
    """
    Prepara las variables numericas y categoricas.
    El preprocesador se ajustara solamente con entrenamiento.
    """

    # Identificar columnas numericas y categoricas
    columnas_numericas = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    columnas_categoricas = X_train.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()

    # Preparacion de variables numericas
    pipeline_numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler())
    ])

    # Preparacion de variables categoricas
    pipeline_categorico = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Unir ambos procesos
    preprocesador = ColumnTransformer([
        ("numericas", pipeline_numerico, columnas_numericas),
        ("categoricas", pipeline_categorico, columnas_categoricas)
    ])

    return preprocesador

if __name__ == "__main__":
    datos = cargar_datos()
    explorar_datos(datos)
    dividir_datos(datos)
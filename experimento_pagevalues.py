
# Experimento: comparar modelos con y sin PageValues

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.datos import cargar_datos, dividir_datos
from src.modelo import crear_modelo_regresion_logistica


# 1. Cargar los datos y dividirlos
datos = cargar_datos()

X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(datos)

# Convertir la variable objetivo a números: False = 0, True = 1
y_train = y_train.astype(int)
y_val = y_val.astype(int)


# 2. Crear las dos versiones de los datos
# Modelo A: conserva todas las variables
X_train_completo = X_train.copy()
X_val_completo = X_val.copy()

# Modelo B: elimina PageValues
X_train_sin_pagevalues = X_train.drop(columns=["PageValues"])
X_val_sin_pagevalues = X_val.drop(columns=["PageValues"])


# 3. Entrenar y evaluar cada modelo usando validación
modelos = [
    (
        "Modelo A: con PageValues",
        X_train_completo,
        X_val_completo
    ),
    (
        "Modelo B: sin PageValues",
        X_train_sin_pagevalues,
        X_val_sin_pagevalues
    )
]

for nombre, X_entrenamiento, X_validacion in modelos:

    print("\n" + "=" * 55)
    print(nombre)
    print("=" * 55)

    # Crear el modelo y ajustar el preprocesamiento
    # únicamente con los datos de entrenamiento
    modelo = crear_modelo_regresion_logistica(X_entrenamiento)
    modelo.fit(X_entrenamiento, y_train)

    # Predecir sobre el conjunto de validación
    predicciones = modelo.predict(X_validacion)

    # Calcular las métricas
    print(f"Accuracy:           {accuracy_score(y_val, predicciones):.4f}")
    print(
        f"Balanced accuracy:  "
        f"{balanced_accuracy_score(y_val, predicciones):.4f}"
    )
    print(
        f"Precision:          "
        f"{precision_score(y_val, predicciones, zero_division=0):.4f}"
    )
    print(f"Recall:              {recall_score(y_val, predicciones, zero_division=0):.4f}")
    print(f"F1-score:            {f1_score(y_val, predicciones, zero_division=0):.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_val, predicciones))

print("\nExperimento terminado.")
print("El conjunto de prueba no se utilizó.")

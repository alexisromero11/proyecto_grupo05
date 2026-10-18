
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.datos import cargar_datos, dividir_datos
from src.modelo import crear_modelo_regresion_logistica


# 1. Cargar y dividir los datos
datos = cargar_datos()

X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(datos)

y_train = y_train.astype(int)
y_val = y_val.astype(int)

# 2. Identificar las variables que vamos a excluir
variables_excluir = [
    "PageValues",
    "BounceRates",
    "ExitRates"
]

X_train_reducido = X_train.drop(columns=variables_excluir)
X_val_reducido = X_val.drop(columns=variables_excluir)

# 3. Entrenar con los datos de entrenamiento
modelo = crear_modelo_regresion_logistica(X_train_reducido)
modelo.fit(X_train_reducido, y_train)

# 4. Evaluar solamente con los datos de validación
predicciones = modelo.predict(X_val_reducido)

print("\n=== MODELO SIN LAS TRES VARIABLES ===")
print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y_val, predicciones):.4f}"
)
print(
    f"Precision: "
    f"{precision_score(y_val, predicciones, zero_division=0):.4f}"
)
print(
    f"Recall: "
    f"{recall_score(y_val, predicciones, zero_division=0):.4f}"
)
print(
    f"F1-score: "
    f"{f1_score(y_val, predicciones, zero_division=0):.4f}"
)

print("\nEl conjunto de prueba no se utilizó.")

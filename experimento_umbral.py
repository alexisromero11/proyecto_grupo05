
import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.datos import cargar_datos, dividir_datos
from src.modelo import crear_modelo_regresion_logistica


# 1. Cargar y dividir los datos
datos = cargar_datos()

X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(datos)

y_train = y_train.astype(int)
y_val = y_val.astype(int)


# 2. Entrenar la regresión logística
modelo = crear_modelo_regresion_logistica(X_train)
modelo.fit(X_train, y_train)


# 3. Obtener probabilidades de compra en validación
probabilidades = modelo.predict_proba(X_val)[:, 1]


# 4. Probar diferentes umbrales
resultados = []

for umbral in np.arange(0.10, 0.91, 0.01):

    predicciones = (probabilidades >= umbral).astype(int)

    balanced = balanced_accuracy_score(y_val, predicciones)

    resultados.append({
        "umbral": round(float(umbral), 2),
        "balanced_accuracy": balanced,
        "precision": precision_score(
            y_val, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y_val, predicciones, zero_division=0
        ),
        "f1": f1_score(
            y_val, predicciones, zero_division=0
        )
    })


# 5. Elegir el umbral con mayor balanced accuracy.
# Si hay empate, preferir el más cercano a 0.50.
mejor = max(
    resultados,
    key=lambda r: (
        r["balanced_accuracy"],
        -abs(r["umbral"] - 0.50)
    )
)

# 6. Mostrar los resultados del umbral elegido
umbral_elegido = mejor["umbral"]
predicciones_finales = (
    probabilidades >= umbral_elegido
).astype(int)

print("\n=== SELECCION DEL UMBRAL ===")
print(f"Umbral elegido:       {umbral_elegido:.2f}")
print(f"Balanced accuracy:    {mejor['balanced_accuracy']:.4f}")
print(f"Precision:            {mejor['precision']:.4f}")
print(f"Recall:               {mejor['recall']:.4f}")
print(f"F1-score:             {mejor['f1']:.4f}")

print("\nMatriz de confusion:")
print(confusion_matrix(y_val, predicciones_finales))

print("\nSolo se utilizo entrenamiento y validacion.")
print("El conjunto de prueba no se utilizo.")
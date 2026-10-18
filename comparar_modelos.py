import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.datos import cargar_datos, dividir_datos, crear_preprocesador
from src.modelo import (
    crear_modelo_base,
    crear_modelo_regresion_logistica,
    crear_red_neuronal_v1,
    crear_red_neuronal_v2
)


def evaluar_modelo(nombre, y_real, probabilidades):
    """Calcula las métricas usando el umbral de decisión 0.5."""

    predicciones = (probabilidades >= 0.5).astype(int)

    print(f"\n========== {nombre} ==========")
    print(f"Accuracy: {accuracy_score(y_real, predicciones):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_real, predicciones):.4f}"
    )
    print(
        f"Precision de compra: "
        f"{precision_score(y_real, predicciones, zero_division=0):.4f}"
    )
    print(
        f"Recall de compra: "
        f"{recall_score(y_real, predicciones, zero_division=0):.4f}"
    )
    print(
        f"F1 de compra: "
        f"{f1_score(y_real, predicciones, zero_division=0):.4f}"
    )
    print("Matriz de confusion:")
    print(confusion_matrix(y_real, predicciones))


# 1. Cargar los datos y realizar la misma division estratificada.
datos = cargar_datos()

X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(datos)

y_train = y_train.astype(int)
y_val = y_val.astype(int)

# La variable objetivo debe estar codificada como 0 y 1.
y_train = y_train.astype(int)
y_val = y_val.astype(int)

# 2. Evaluar el modelo base trivial.
modelo_base = crear_modelo_base(X_train)
modelo_base.fit(X_train, y_train)
pred_base = modelo_base.predict(X_val)

print("\n========== MODELO BASE ==========")
print(f"Accuracy: {accuracy_score(y_val, pred_base):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_val, pred_base):.4f}")
print(f"Precision de compra: {precision_score(y_val, pred_base, zero_division=0):.4f}")
print(f"Recall de compra: {recall_score(y_val, pred_base, zero_division=0):.4f}")
print(f"F1 de compra: {f1_score(y_val, pred_base, zero_division=0):.4f}")
print("Matriz de confusion:")
print(confusion_matrix(y_val, pred_base))

# 3. Evaluar la regresion logistica.
modelo_logistico = crear_modelo_regresion_logistica(X_train)
modelo_logistico.fit(X_train, y_train)
prob_logistico = modelo_logistico.predict_proba(X_val)[:, 1]

evaluar_modelo("REGRESION LOGISTICA", y_val, prob_logistico)

# 4. Ajustar el preprocesador solo con el conjunto de entrenamiento.
preprocesador = crear_preprocesador(X_train)
X_train_preparado = preprocesador.fit_transform(X_train)
X_val_preparado = preprocesador.transform(X_val)

X_train_preparado = np.asarray(X_train_preparado, dtype="float32")
X_val_preparado = np.asarray(X_val_preparado, dtype="float32")

# 5. Entrenar y evaluar las redes neuronales.
for nombre, creador in [
    ("RED NEURONAL V1", crear_red_neuronal_v1),
    ("RED NEURONAL V2", crear_red_neuronal_v2)
]:
    modelo = creador(X_train_preparado.shape[1])

    parada_temprana = __import__("tensorflow").keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    modelo.fit(
        X_train_preparado,
        y_train,
        validation_data=(X_val_preparado, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[parada_temprana],
        verbose=0
    )

    probabilidades = modelo.predict(X_val_preparado, verbose=0).ravel()
    evaluar_modelo(nombre, y_val, probabilidades)

print("\nComparacion terminada. El conjunto de prueba no se ha utilizado.")


import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.datos import cargar_datos, dividir_datos, crear_preprocesador
from src.modelo import (
    crear_modelo_base,
    crear_modelo_regresion_logistica,
    crear_red_neuronal_v1,
    crear_red_neuronal_v2
)


# 1. Cargar y dividir los datos
datos = cargar_datos()

X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(datos)

y_train = y_train.astype(int)
y_val = y_val.astype(int)


# 2. Preparar los datos para las redes neuronales
# El preprocesador se ajusta únicamente con entrenamiento.
preprocesador = crear_preprocesador(X_train)

X_train_nn = np.asarray(
    preprocesador.fit_transform(X_train),
    dtype="float32"
)

X_val_nn = np.asarray(
    preprocesador.transform(X_val),
    dtype="float32"
)


# 3. Entrenar el modelo trivial
modelo_base = crear_modelo_base(X_train)
modelo_base.fit(X_train, y_train)

probabilidades_base = modelo_base.predict_proba(X_val)[:, 1]


# 4. Entrenar la regresión logística
modelo_logistico = crear_modelo_regresion_logistica(X_train)
modelo_logistico.fit(X_train, y_train)

probabilidades_logistico = modelo_logistico.predict_proba(X_val)[:, 1]


# 5. Entrenar las dos redes neuronales
resultados_modelos = [
    ("Modelo base", probabilidades_base),
    ("Regresión logística", probabilidades_logistico)
]

for nombre, creador in [
    ("Red neuronal V1", crear_red_neuronal_v1),
    ("Red neuronal V2", crear_red_neuronal_v2)
]:
    modelo = creador(X_train_nn.shape[1])

    parada_temprana = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    modelo.fit(
        X_train_nn,
        y_train,
        validation_data=(X_val_nn, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[parada_temprana],
        verbose=0
    )

    probabilidades = modelo.predict(X_val_nn, verbose=0).ravel()
    resultados_modelos.append((nombre, probabilidades))


# 6. Buscar el mejor umbral para cada modelo
umbrales = np.round(np.arange(0.10, 0.91, 0.01), 2)

print("\n=== COMPARACION DE UMBRALES EN VALIDACION ===")

for nombre, probabilidades in resultados_modelos:

    mejor = None

    for umbral in umbrales:
        predicciones = (probabilidades >= umbral).astype(int)

        balanced = balanced_accuracy_score(y_val, predicciones)
        precision = precision_score(
            y_val, predicciones, zero_division=0
        )
        recall = recall_score(
            y_val, predicciones, zero_division=0
        )
        f1 = f1_score(y_val, predicciones, zero_division=0)

        candidato = {
            "umbral": float(umbral),
            "balanced_accuracy": balanced,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Si hay empate, preferir el umbral más cercano a 0.50.
        if (
            mejor is None
            or candidato["balanced_accuracy"] > mejor["balanced_accuracy"]
            or (
                np.isclose(
                    candidato["balanced_accuracy"],
                    mejor["balanced_accuracy"]
                )
                and abs(candidato["umbral"] - 0.50)
                < abs(mejor["umbral"] - 0.50)
            )
        ):
            mejor = candidato

    print(f"\n--- {nombre} ---")
    print(f"Umbral elegido:      {mejor['umbral']:.2f}")
    print(
        f"Balanced accuracy:   "
        f"{mejor['balanced_accuracy']:.4f}"
    )
    print(f"Precision:           {mejor['precision']:.4f}")
    print(f"Recall:              {mejor['recall']:.4f}")
    print(f"F1-score:            {mejor['f1']:.4f}")

print("\nComparación terminada.")
print("Solo se utilizaron entrenamiento y validación.")
print("El conjunto de prueba no se utilizó.")


# ============================================
# PROYECTO: INTENCION DE COMPRA EN LINEA
# Grupo 5 - Tecnologias Emergentes
# Etapa 3: Entrenamiento de red neuronal V2
# ============================================

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.datos import (
    cargar_datos,
    dividir_datos,
    crear_preprocesador
)
from src.modelo import crear_red_neuronal_v2


# Reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)


# 1. Cargar y dividir los datos
datos = cargar_datos()

X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(datos)


# 2. Ajustar el preprocesador solo con entrenamiento
preprocesador = crear_preprocesador(X_train)

X_train_procesado = preprocesador.fit_transform(X_train)
X_val_procesado = preprocesador.transform(X_val)

X_train_procesado = np.asarray(X_train_procesado, dtype="float32")
X_val_procesado = np.asarray(X_val_procesado, dtype="float32")


# 3. Crear la segunda red neuronal
modelo = crear_red_neuronal_v2(X_train_procesado.shape[1])


# 4. Entrenar la red neuronal
modelo.fit(
    X_train_procesado,
    y_train.astype("float32"),
    validation_data=(
        X_val_procesado,
        y_val.astype("float32")
    ),
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]
)


# 5. Predecir las clases de validacion
probabilidades = modelo.predict(X_val_procesado, verbose=0).ravel()
predicciones = (probabilidades >= 0.5).astype(int)


# 6. Evaluar el modelo
print("\n========== RED NEURONAL V2 ==========")
print("Accuracy:", round(accuracy_score(y_val, predicciones), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_val, predicciones), 4)
)
print(
    "Precision de compra:",
    round(precision_score(y_val, predicciones, zero_division=0), 4)
)
print(
    "Recall de compra:",
    round(recall_score(y_val, predicciones, zero_division=0), 4)
)
print("F1 de compra:", round(f1_score(y_val, predicciones, zero_division=0), 4))
print("\nMatriz de confusion:")
print(confusion_matrix(y_val, predicciones))

print("\nEntrenamiento de la red neuronal V2 completado.")

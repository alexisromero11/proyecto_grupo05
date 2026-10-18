# Etapa 3: Comparación de modelos

## 1. Objetivo

Comparar un modelo base trivial, una regresión logística y dos versiones de una red neuronal para predecir si una sesión de navegación terminará en una compra (`Revenue`).

## 2. Metodología

Se utilizó una división estratificada de los datos:

- Entrenamiento: 8,631 registros.
- Validación: 1,849 registros.
- Prueba: 1,850 registros.

La proporción de compras en validación fue de aproximadamente 15.47 %. El preprocesador se ajustó únicamente con el conjunto de entrenamiento.

Los cuatro modelos se evaluaron sobre el mismo conjunto de validación, con un umbral de clasificación de 0.5. El conjunto de prueba no se utilizó para seleccionar modelos.

## 3. Resultados de validación

| Modelo | Accuracy | Balanced accuracy | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Modelo base | 0.8453 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Regresión logística | 0.8480 | 0.8016 | 0.5060 | 0.7343 | 0.5991 |
| Red neuronal V1 | 0.8886 | 0.7484 | 0.6724 | 0.5455 | 0.6023 |
| Red neuronal V2 | 0.8956 | 0.7411 | 0.7291 | 0.5175 | 0.6053 |

Las métricas de precisión, recall y F1 corresponden a la clase positiva (`Revenue = True`).

## 4. Análisis

El modelo base alcanza una accuracy de 0.8453, pero no identifica ninguna compra. Esto evidencia que la exactitud por sí sola es insuficiente debido al desbalance de clases.

La regresión logística obtiene el mayor recall (0.7343) y la mayor balanced accuracy (0.8016). Detecta 210 de las 286 compras reales del conjunto de validación, aunque también produce falsos positivos.

La red neuronal V1 alcanza un F1 de 0.6023, mientras que la V2 obtiene 0.6053. La V2 registra la mayor accuracy (0.8956), precisión (0.7291) y F1 (0.6053) de esta ejecución, pero su recall es de 0.5175.

## 5. Conclusión provisional

No existe un modelo superior en todas las métricas. La regresión logística es preferible si la prioridad es detectar la mayor proporción posible de compras reales. La red neuronal V2 presenta el mejor F1 en esta ejecución.

La selección definitiva deberá justificarse según el costo de los falsos positivos y falsos negativos, utilizando únicamente el conjunto de validación. Las redes neuronales pueden producir resultados ligeramente distintos entre ejecuciones debido a su entrenamiento estocástico.

La evaluación final con el conjunto de prueba queda pendiente para la etapa de evaluación y versionado.

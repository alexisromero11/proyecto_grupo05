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

Los resultados muestran que ningún modelo obtiene el mejor desempeño en todas las métricas. El modelo base alcanza una accuracy de 0.8453, pero no identifica compras, por lo que no resulta adecuado para el objetivo del proyecto.

La regresión logística obtiene la mayor balanced accuracy (0.8016) y el mayor recall (0.7343), por lo que identifica una mayor proporción de las sesiones que terminan en compra. Sin embargo, su precisión (0.5060) indica que una parte importante de las predicciones positivas son falsos positivos.

La red neuronal V1 obtiene un F1 de 0.6023 y la V2 un F1 de 0.6053. Aunque la V2 alcanza la mayor accuracy, precisión y F1 en esta ejecución, su recall es inferior al de la regresión logística. Por tanto, no se puede afirmar que la red neuronal sea superior en todos los aspectos.

El experimento adicional mostró una disminución del F1 al excluir `PageValues`, `BounceRates` y `ExitRates`. Este resultado evidencia que las variables influyen en el rendimiento, pero no demuestra por sí mismo que exista fuga de información.

Como el objetivo es predecir durante la navegación, la selección final también debe considerar qué variables están disponibles en el instante de la predicción. El conjunto de datos contiene registros por sesión y no permite demostrar, por sí solo, que todas las variables estén disponibles en tiempo real.

Por ahora, la regresión logística y la red neuronal V2 son alternativas que deben considerarse según el equilibrio entre falsos positivos y falsos negativos. La elección definitiva deberá justificarse con los resultados de validación y la finalidad del sistema. La evaluación final con el conjunto de prueba se realizará en la etapa correspondiente.

## Experimento adicional: exclusión de variables

Se evaluó una regresión logística excluyendo las variables `PageValues`, `BounceRates` y `ExitRates`, debido a que su disponibilidad durante la navegación debe analizarse antes de utilizarlas en un sistema de predicción anticipada.

Los resultados obtenidos en el conjunto de validación fueron:

| Métrica | Resultado |
|---|---:|
| Balanced accuracy | 0.6459 |
| Precision | 0.2535 |
| Recall | 0.6329 |
| F1-score | 0.3620 |

En comparación con la regresión logística que conserva todas las variables, el F1-score disminuyó de 0.5991 a 0.3620. Esto indica una reducción del rendimiento predictivo al excluir las tres variables.

Este experimento constituye un análisis de sensibilidad y no demuestra por sí solo la existencia de fuga de información. Antes de seleccionar el modelo definitivo, se debe considerar si las variables utilizadas están disponibles en el instante en que se pretende realizar la predicción. El conjunto de prueba no se utilizó en este experimento.

## Experimento de selección del umbral

Debido al desbalance de clases del dataset, se seleccionó el umbral de clasificación de la regresión logística mediante el conjunto de validación. Se consideró que los falsos positivos y los falsos negativos tienen una importancia similar, por lo que se utilizó la balanced accuracy como criterio principal.

Se probaron umbrales entre 0.10 y 0.90, con incrementos de 0.01. El mejor resultado de la búsqueda se obtuvo con un umbral de 0.37.

| Métrica | Umbral 0.50 | Umbral 0.37 |
|---|---:|---:|
| Balanced accuracy | 0.8016 | 0.8123 |
| Precision | 0.5060 | 0.3859 |
| Recall | 0.7343 | 0.8811 |
| F1-score | 0.5991 | 0.5367 |

Con el umbral de 0.37 se identificaron 252 de las 286 compras reales, mientras que 34 compras no fueron detectadas. También se obtuvieron 401 falsos positivos y 1,162 verdaderos negativos.

El resultado muestra que el umbral seleccionado mejora ligeramente la balanced accuracy y aumenta el recall, a cambio de una reducción de la precisión. Esta decisión es provisional y corresponde a la regresión logística; antes de la evaluación final deberá verificarse su coherencia con el modelo elegido y el objetivo del sistema.

La selección se realizó únicamente con entrenamiento y validación. El conjunto de prueba no se utilizó.
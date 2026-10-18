# Comparación de modelos de clasificación

## 1. Objetivo

Comparar cuatro modelos para predecir si una sesión de navegación en una tienda virtual termina en una compra (`Revenue`): un modelo base, regresión logística y dos versiones de una red neuronal.

La comparación se realiza utilizando el conjunto de validación, manteniendo reservado el conjunto de prueba para la evaluación final.

## 2. Resultados

| Métrica | Modelo base | Regresión logística | Red neuronal V1 | Red neuronal V2 |
|---|---:|---:|---:|---:|
| Accuracy | 0.8453 | 0.8480 | 0.8908 | 0.8913 |
| Balanced accuracy | 0.5000 | 0.8016 | 0.7597 | 0.7386 |
| Precision de compra | 0.0000 | 0.5060 | 0.6736 | 0.7014 |
| Recall de compra | 0.0000 | 0.7343 | 0.5699 | 0.5175 |
| F1 de compra | 0.0000 | 0.5991 | 0.6174 | 0.5956 |

## 3. Análisis

### Modelo base

Predice siempre la clase mayoritaria, es decir, que la sesión no termina en compra. Obtiene una accuracy de 0.8453, pero no detecta ninguna compra. Por ello, sirve como referencia y no como solución final.

### Regresión logística

Obtiene un recall de compra de 0.7343 y una balanced accuracy de 0.8016. Es el modelo que mejor detecta las compras reales entre las cuatro alternativas evaluadas.

### Red neuronal V1

Alcanza un F1 de compra de 0.6174, el mayor de los cuatro modelos. Su precision es 0.6736 y su recall es 0.5699. Presenta el mejor equilibrio entre precision y recall según el F1 obtenido con el umbral de clasificación utilizado.

### Red neuronal V2

Obtiene la mayor accuracy (0.8913) y la mayor precision de compra (0.7014), pero su recall es 0.5175 y su F1 es 0.5956. Por tanto, su mayor accuracy no significa que sea la mejor alternativa para detectar compras.

## 4. Conclusión provisional

La red neuronal V1 es la candidata provisional si se prioriza el F1 de compra. La regresión logística es una alternativa importante si el objetivo principal es detectar una mayor proporción de sesiones que terminan en compra.

La selección definitiva deberá considerar el objetivo del proyecto, los costes relativos de los falsos positivos y falsos negativos y, si se ajusta el umbral, los resultados obtenidos en validación. El conjunto de prueba se utilizará únicamente para la evaluación final del modelo seleccionado.

## 5. Consideraciones metodológicas

- Los resultados corresponden al conjunto de validación.
- El conjunto de prueba no se ha utilizado para seleccionar los modelos.
- La variable objetivo es `Revenue`.
- La distribución de clases está desequilibrada, por lo que no se utiliza la accuracy como único criterio.
- Los resultados registrados corresponden al umbral de clasificación de 0.5 para los modelos que producen probabilidades.
- Las cifras se registran a partir de las ejecuciones realizadas durante el desarrollo del proyecto.
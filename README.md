# Proyecto Grupo 5 — Predicción de intención de compra en línea

## 1. Descripción del proyecto

Este proyecto utiliza el dataset **Online Shoppers Purchasing Intention**, publicado por el repositorio UCI Machine Learning Repository.

El objetivo es desarrollar modelos de clasificación que predigan si una sesión de navegación de un usuario terminará en una compra (`Revenue`).

La variable objetivo `Revenue` tiene dos clases:
- `True`: la sesión terminó en una compra.
- `False`: la sesión no terminó en una compra.

## 2. Dataset

- **Fuente:** UCI Machine Learning Repository.
- **Enlace:** https://archive.ics.uci.edu/dataset/468
- **Archivo local esperado:** `data/online_shoppers_intention.csv`

El dataset no se incluye en GitHub. Cada integrante debe obtenerlo de la fuente oficial y colocarlo en la carpeta `data/`.

## 3. Estructura del proyecto

```text
proyecto_grupo05/
├── README.md
├── requirements.txt
├── entrenar.py
├── entrenar_v2.py
├── src/
│   ├── __init__.py
│   ├── datos.py
│   └── modelo.py
├── docs/
│   └── comparacion_modelos.md
├── notebooks/
│   └── exploracion.ipynb
├── data/
│   └── online_shoppers_intention.csv
└── modelos/
```

## 4. Requisitos

- Python compatible con las versiones de las bibliotecas indicadas en `requirements.txt`.
- Git.
- Visual Studio Code u otro editor de Python.

## 5. Instalación

Clonar el repositorio:

```bash
git clone https://github.com/alexisromero11/proyecto_grupo05.git
cd proyecto_grupo05
```

Crear un entorno virtual en Windows:

```powershell
python -m venv .venv
```

Activarlo en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

Descargar el dataset desde la fuente oficial indicada en la sección 2 y guardarlo en `data/online_shoppers_intention.csv`.

## 6. Ejecución

Ejecutar el entrenamiento de la primera versión de la red neuronal:

```powershell
python entrenar.py
```

Ejecutar el entrenamiento de la segunda versión:

```powershell
python entrenar_v2.py
```

Los scripts utilizan los módulos de `src/` para cargar, dividir y preprocesar los datos y construir los modelos.

## 7. Modelos evaluados

El proyecto compara:

1. Un modelo base trivial (`DummyClassifier`).
2. Una regresión logística.
3. Una red neuronal de primera versión (V1).
4. Una red neuronal de segunda versión (V2).

La comparación de resultados se documenta en `docs/comparacion_modelos.md`.

Debido al desbalance de clases, se consideran métricas como precisión, recall y F1, además de accuracy.

## 8. Prevención de fuga de información

Los datos se dividen en conjuntos de entrenamiento, validación y prueba. El preprocesador se ajusta con los datos de entrenamiento y luego se aplica a los demás conjuntos.

El conjunto de prueba debe reservarse para la evaluación final y no utilizarse para seleccionar modelos o ajustar parámetros.

## 9. Fuente y referencia

Sakar, C. O., y otros. *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/468

DOI: https://doi.org/10.24432/C5F88Q

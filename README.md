# Predicción de retrasos en entregas
Proyecto de análisis de datos y Machine Learning aplicado a logística para analizar los factores relacionados con los retrasos en las entregas y desarrollar un modelo capaz de anticiparlos.

## Herramientas utilizadas
Excel / Power Query, Power BI, SQL, Python (Scikit-learn)

## Objetivo
El objetivo del proyecto es analizar qué factores están relacionados con las entregas tardías y desarrollar un modelo predictivo que permita identificar pedidos con riesgo de retraso.

La variable objetivo utilizada es `entrega_tarde`:
- `0` = entrega a tiempo
- `1` = entrega tardía

## Proceso del proyecto
1. Limpieza y preparación de los datos.
2. Análisis exploratorio en Excel.
3. Creación de dashboard en Power BI.
4. Análisis mediante SQL.
5. Preparación de datos en Python.
6. Entrenamiento y comparación de modelos de Machine Learning.
7. Ajuste del umbral de clasificación.
8. Selección del modelo final.

## Machine Learning
Se compararon tres modelos: Regresión Logistica, Arbol de decision y Random Forest y la Regresión Logística presentó el mejor rendimiento general.

## Modelo final
Se ajustó el umbral de clasificación de la Regresión Logística presentando mejores resultados 0,4.

## Principales conclusiones
Entre las variables más relevantes aparecen:distancia, transportista, coste de envío, prioridad, provincia, categoría

También se observaron patrones como un peor comportamiento de TransNova y mejores resultados relativos de Rapidex, Sevilla y la categoría Deporte.


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:02:57 2024

@author: javit
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar los datos
def cargar_datos_series(base_path, dataset_name):
    data = []
    labels = []

    for folder, subfolders, files in os.walk(base_path):
        # Filtrar solo carpetas de variedades de aceitunas
        if folder.endswith('Arbequina') or folder.endswith('Picual'):
            class_label = os.path.basename(folder)  # La etiqueta es el nombre de la clase (Arbequina, Picual)
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(folder, file)
                    df = pd.read_csv(file_path)
                    
                    # Nos aseguramos que cada archivo CSV tenga exactamente 201 puntos
                    if len(df) == 201:
                        # Calcular la magnitud en dB: 20 * log10(sqrt(Real^2 + Imaginary^2))
                        magnitud_db = 20 * np.log10(np.sqrt(df['Real Part']**2 + df['Imaginary Part']**2))
                        
                        # Combinar dataset_name con class_label para crear etiquetas jerárquicas
                        label = f"{dataset_name}_{class_label}"
                        
                        # Añadir la magnitud en dB como una secuencia (1D) en vez de 3 columnas
                        data.append(magnitud_db.values.reshape(-1, 1))  # Reshape para mantener la forma (201, 1)
                        labels.append(label)

    # Convertir a arrays de numpy para manejo eficiente
    data = np.array(data)  # (n_samples, 201, 1)
    labels = np.array(labels)
    return data, labels

# Función para seleccionar la carpeta con Tkinter
def seleccionar_carpeta():
    root = tk.Tk()
    root.withdraw()  # Esconde la ventana principal de tkinter
    carpeta_seleccionada = filedialog.askdirectory()  # Abre el diálogo para seleccionar carpeta
    return carpeta_seleccionada

# Paso 2: Seleccionar las carpetas donde se encuentran los dos datasets
print("Selecciona la carpeta para 'MartinDeGrinyones'")
ruta_martin = seleccionar_carpeta()
print("Selecciona la carpeta para 'Abril'")
ruta_abril = seleccionar_carpeta()

if not ruta_martin or not ruta_abril:
    print("No se seleccionó alguna de las carpetas.")
else:
    # Cargar datos de ambos datasets
    print(f"Cargando datos de MartinDeGrinyones desde: {ruta_martin}")
    data_martin, labels_martin = cargar_datos_series(ruta_martin, "MartinDeGrinyones")

    print(f"Cargando datos de Abril desde: {ruta_abril}")
    data_abril, labels_abril = cargar_datos_series(ruta_abril, "Abril")

    # Combinar ambos datasets
    data = np.vstack((data_martin, data_abril))
    labels = np.hstack((labels_martin, labels_abril))

    # Paso 3: Separar el 20% para validación externa
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    # Paso 4: Flatten de las secuencias para usar en RandomForest
    # (n_samples, 201, 1) -> (n_samples, 201)
    X_train_flatten = X_train.reshape((X_train.shape[0], -1))
    X_test_flatten = X_test.reshape((X_test.shape[0], -1))

    # Paso 5: Configurar el clasificador y realizar validación cruzada
    # Usamos un pipeline con escalado de características y un clasificador RandomForest
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))

    # Obtener predicciones usando cross-validation en el conjunto de entrenamiento
    y_train_pred = cross_val_predict(clf, X_train_flatten, y_train, cv=5)

    # Calcular el accuracy en cross-validation (entrenamiento)
    accuracy_cv = accuracy_score(y_train, y_train_pred)

    # Entrenar el clasificador con el conjunto de entrenamiento completo
    clf.fit(X_train_flatten, y_train)

    # Evaluar el modelo en el conjunto de prueba externo
    y_test_pred = clf.predict(X_test_flatten)

    # Calcular el accuracy en el conjunto de prueba
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Mostrar los resultados
    print(f'Accuracy en Cross-Validation: {accuracy_cv:.2f}')
    print(f'Accuracy en Test: {accuracy_test:.2f}')

    # Paso 6: Generar la matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred, labels=clf.named_steps['randomforestclassifier'].classes_)

    # Obtener el número total de muestras de prueba
    total_test_samples = len(y_test)

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 8))  # Aumentar el tamaño de la figura
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clf.named_steps['randomforestclassifier'].classes_,
                yticklabels=clf.named_steps['randomforestclassifier'].classes_)

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    # Rotar las etiquetas del eje x y ajustar el tamaño de la fuente
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.title(f'Matriz de Confusión (Total muestras de prueba: {total_test_samples})', fontsize=14)
    plt.tight_layout()  # Ajustar el layout para que no se recorten las etiquetas

    # Mostrar la matriz de confusión
    plt.show()

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar los datos
def cargar_datos_series(base_path):
    data = []
    labels = []

    for folder, subfolders, files in os.walk(base_path):
        # Filtrar solo carpetas de variedades de aceitunas
        if folder.endswith('Arbequina') or folder.endswith('Picual') or folder.endswith('HojiBlanca') or folder.endswith('CorniCabra'):
            label = os.path.basename(folder)  # La etiqueta es el nombre de la carpeta (Arbequina, Picual, Cornicabra)
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(folder, file)
                    df = pd.read_csv(file_path)
                    
                    # Nos aseguramos que cada archivo CSV tenga exactamente 201 puntos
                    if len(df) == 201:
                        # Calcular la magnitud en dB: 20 * log10(sqrt(Real^2 + Imaginary^2))
                        magnitud_db = 20 * np.log10(np.sqrt(df['Real Part']**2 + df['Imaginary Part']**2))
                        
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

# Paso 2: Seleccionar la carpeta donde se encuentran los datos
ruta_datos = seleccionar_carpeta()
if not ruta_datos:
    print("No se seleccionó ninguna carpeta.")
else:
    print(f"Carpeta seleccionada: {ruta_datos}")
    data, labels = cargar_datos_series(ruta_datos)

    # Paso 3: Separar el 30% para validación externa
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)

    # Paso 4: Flatten de las secuencias para usar en RandomForest
    # (n_samples, 201, 1) -> (n_samples, 201)
    X_train_flatten = X_train.reshape((X_train.shape[0], -1))
    X_test_flatten = X_test.reshape((X_test.shape[0], -1))

    # Paso 5: Configurar el clasificador y realizar validación cruzada
    # Usamos un pipeline con escalado de características y un clasificador RandomForest
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))

    # Realizar validación cruzada en los datos de entrenamiento
    scores = cross_val_score(clf, X_train_flatten, y_train, cv=5, scoring='accuracy')

    # Entrenar el clasificador con el conjunto de entrenamiento completo
    clf.fit(X_train_flatten, y_train)

    # Evaluar el modelo en el conjunto de prueba externo
    test_score = clf.score(X_test_flatten, y_test)

    # Resultados de validación cruzada
    print(f'Validación cruzada (Accuracy): {scores.mean():.2f} ± {scores.std():.2f}')
    print(f'Exactitud en el conjunto de prueba externo: {test_score:.2f}')

    # Paso 6: Generar la matriz de confusión
    y_pred = clf.predict(X_test_flatten)  # Predicciones en el conjunto de prueba
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)  # Matriz de confusión

    # Obtener el número total de muestras de prueba
    total_test_samples = len(y_test)

    # Graficar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.named_steps['randomforestclassifier'].classes_, yticklabels=clf.named_steps['randomforestclassifier'].classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Título de la matriz de confusión con el número total de muestras de prueba
    plt.title(f'Matriz de Confusión (Total muestras de prueba: {total_test_samples})')
    
    # Mostrar la matriz de confusión
    plt.show()

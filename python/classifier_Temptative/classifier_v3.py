import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar los datos
def cargar_datos_series(base_path, dataset_name):
    data = []
    labels = []

    for folder, subfolders, files in os.walk(base_path):
        # Ajustar a las nuevas clases
        if folder.endswith('Arbequina') or folder.endswith('Picual') or folder.endswith('CorniCabra') or folder.endswith('HojiBlanca'):
            class_label = os.path.basename(folder)
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(folder, file)
                    df = pd.read_csv(file_path)
                    
                    if len(df) == 201:
                        #magnitud_db = 20 * np.log10(np.sqrt(df['Real Part']**2 + df['Imaginary Part']**2))
                        magnitud_db = np.sqrt(df['Real Part']**2 + df['Imaginary Part']**2)
                        # Crear etiquetas jerárquicas
                        label = f"{dataset_name}_{class_label}"
                        data.append(magnitud_db.values.reshape(-1, 1))
                        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Función para seleccionar la carpeta
def seleccionar_carpeta():
    root = tk.Tk()
    root.withdraw()  
    return filedialog.askdirectory()

# Paso 2: Seleccionar las carpetas
print("Selecciona la carpeta para 'MartinDeGrinyones'")
ruta_martin = seleccionar_carpeta()
print("Selecciona la carpeta para 'Abril'")
ruta_abril = seleccionar_carpeta()

if not ruta_martin or not ruta_abril:
    print("No se seleccionó alguna de las carpetas.")
else:
    print(f"Cargando datos de MartinDeGrinyones desde: {ruta_martin}")
    data_martin, labels_martin = cargar_datos_series(ruta_martin, "MartinDeGrinyones")

    print(f"Cargando datos de Abril desde: {ruta_abril}")
    data_abril, labels_abril = cargar_datos_series(ruta_abril, "Abril")

    # Combinar ambos datasets
    data = np.vstack((data_martin, data_abril))
    labels = np.hstack((labels_martin, labels_abril))

    # Separar el 20% para validación externa y el 80% para entrenamiento y prueba
    X_temp, X_val, y_temp, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42,shuffle=True)

    # Separar el conjunto temporal en entrenamiento y prueba (60% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.30, stratify=y_temp, random_state=42,shuffle=True)

    # Flatten de las secuencias para usar en RandomForest
    X_train_flatten = X_train.reshape((X_train.shape[0], -1))
    X_test_flatten = X_test.reshape((X_test.shape[0], -1))
    X_val_flatten = X_val.reshape((X_val.shape[0], -1))

    # Configurar el clasificador
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))

    # Obtener predicciones usando cross-validation
    y_train_pred = cross_val_predict(clf, X_train_flatten, y_train, cv=5)
    accuracy_cv = accuracy_score(y_train, y_train_pred)

    clf.fit(X_train_flatten, y_train)
    y_test_pred = clf.predict(X_test_flatten)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Mostrar los resultados
    print('\n')
    print(f'Accuracy en Cross-Validation: {accuracy_cv:.2f}')
    print(f'Accuracy en Test: {accuracy_test:.2f}')

    # Mostrar el classification report
    print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

    # Generar la matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred, labels=clf.named_steps['randomforestclassifier'].classes_)

    total_test_samples = len(y_test)

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clf.named_steps['randomforestclassifier'].classes_,
                yticklabels=clf.named_steps['randomforestclassifier'].classes_)

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(f'Matriz de Confusión (Total muestras de prueba: {total_test_samples})', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Paso 3: Clustering de todo el dataset usando PCA
    # Aplanar los datos antes de PCA
    data_flatten = data.reshape((data.shape[0], -1))

    # Reducir a tres dimensiones usando PCA
    pca = PCA(n_components=3)  
    data_pca = pca.fit_transform(data_flatten)

    # Filtrar muestras que tienen componentes grandes
    # Establecer un umbral basado en el percentil 95 para cada componente
    pca_thresholds = np.percentile(data_pca, 85, axis=0)
    filtered_indices = np.all(data_pca <= pca_thresholds, axis=1)

    # Filtrar los datos y etiquetas
    data_pca_filtered = data_pca[filtered_indices]
    labels_filtered = labels[filtered_indices]

    # Crear un dataframe para facilitar la visualización
    df = pd.DataFrame(data_pca_filtered, columns=['PCA1', 'PCA2', 'PCA3'])
    df['Label'] = labels_filtered

    # Definir paletas de colores
    color_map = {
        'MartinDeGrinyones_Arbequina': 'green',
        'MartinDeGrinyones_Picual': 'yellow',
        'MartinDeGrinyones_Cornicabra': 'blue',  # Nuevo color para Cornicabra
        'Abril_Arbequina': 'red',
        'Abril_Picual': 'orange',
        'Abril_HojiBlanca': 'purple'  # Nuevo color para HojiBlanca
    }

    # Graficar PCA1 vs PCA2
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for label, color in color_map.items():
        subset = df[df['Label'] == label]
        plt.scatter(subset['PCA1'], subset['PCA2'], color=color, label=label, alpha=0.6)
    plt.title('PCA1 vs PCA2')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()

    # Graficar PCA1 vs PCA3
    plt.subplot(1, 3, 2)
    for label, color in color_map.items():
        subset = df[df['Label'] == label]
        plt.scatter(subset['PCA1'], subset['PCA3'], color=color, label=label, alpha=0.6)
    plt.title('PCA1 vs PCA3')
    plt.xlabel('PCA1')
    plt.ylabel('PCA3')
    plt.legend()

    # Graficar PCA2 vs PCA3
    plt.subplot(1, 3, 3)
    for label, color in color_map.items():
        subset = df[df['Label'] == label]
        plt.scatter(subset['PCA2'], subset['PCA3'], color=color, label=label, alpha=0.6)
    plt.title('PCA2 vs PCA3')
    plt.xlabel('PCA2')
    plt.ylabel('PCA3')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Graficar en 3D utilizando PCA1, PCA2 y PCA3
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for label, color in color_map.items():
        subset = df[df['Label'] == label]
        ax.scatter(subset['PCA1'], subset['PCA2'], subset['PCA3'], color=color, label=label, alpha=0.6)

    # Etiquetas de los ejes
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.set_title('Plot 3D de las tres primeras componentes principales')

    # Mostrar la leyenda y la gráfica
    ax.legend()
    plt.show()
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Importar módulo 3D de matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

# Paso 1: Cargar los datos
def cargar_datos_series(base_path, dataset_name,centF):
    data = []
    labels = []

    for folder, subfolders, files in os.walk(base_path):
        # Ajustar a las nuevas clases incluyendo los datos de test externo
        if folder.endswith('x50mg') or folder.endswith('x100mg') or folder.endswith('x150mg') or folder.endswith('x200mg'):
            class_label = os.path.basename(folder)
            # Inicializar base_s21
            base_s21 = None
            
            # Primer bucle para identificar y calcular s21 para el archivo base
            for file in files:
                if file.startswith('base') and file.endswith('.csv'):
                    file_path = os.path.join(folder, file)
                    df_base = pd.read_csv(file_path)
                    
                    # Comprobar las columnas
                    if 'Real Part' in df_base.columns and 'Imaginary Part' in df_base.columns and len(df_base) == 201:
                        # Calcular s21 para el archivo base
                        base_s21 = 20 * np.log10(np.abs(df_base['Real Part'] + 1j * df_base['Imaginary Part']))
                        #base_s21 = (np.abs(df_base['Real Part'] + 1j * df_base['Imaginary Part']))
                        #print(f"Cargando archivo base: {file_path}")
                    else:
                        print(f"\nEl archivo base {file} no tiene las columnas esperadas o no tiene 201 filas")

            # Segundo bucle para calcular s21 para los demás archivos
            plt.figure(figsize=(10, 6))
            for file in files:
                if file.endswith('.csv') and not file.startswith('base'):
                    file_path = os.path.join(folder, file)
                    df = pd.read_csv(file_path)

                    # Imprimir las primeras filas del archivo CSV para verificar el contenido
                    #print(f"Cargando archivo: {file_path}")
                    #print(df.head())  # Verifica las columnas y contenido
                    
                    if 'Real Part' in df.columns and 'Imaginary Part' in df.columns and len(df) == 201:
                        # Calcular s21 para el archivo actual
                        s21 = 20 * np.log10(np.abs(df['Real Part'] + 1j * df['Imaginary Part']))
                        #s21 = (np.abs(df['Real Part'] + 1j * df['Imaginary Part']))
                        # Restar el s21 del archivo base
                        s21_corrected = -(s21 - base_s21)
                        #s21_corrected = (s21/base_s21)                        # idx_min = np.argmax(s21_corrected)
                        # freq_min = df['Freq (Hz)'].iloc[idx_min]
                        # s21_min = s21_corrected.iloc[idx_min]
        
                        # deltaF = centF - freq_min
                        # freq_corrected = df['Freq (Hz)'].values+deltaF 
                        # f_interp = interp1d(freq_corrected, s21_corrected, bounds_error=False, fill_value="extrapolate")
                        # s21_corrected_interpolated = f_interp(df['Freq (Hz)'])  # Interpolamos a las frecuencias originales
                        s21_corrected_interpolated = s21_corrected                      
                        # Almacenar en el array
                        data.append(s21_corrected_interpolated.values.reshape(-1, 1))
                        # Crear etiquetas jerárquicas
                        label = f"{dataset_name}_{class_label}"
                        labels.append(label)
                        plt.plot(df['Freq (Hz)']/1e9, s21_corrected_interpolated, color='gray', alpha=1, label=label)
                    else:
                        print(f"El archivo {file} no tiene las columnas esperadas o no tiene 201 filas")

                    plt.title(f"All spectra from {class_label}")
                    plt.xlabel('Frecuencia (GHz)')
                    plt.ylabel(r'20$log_{10}(|S21|)$ (dB)')
                     
                    plt.grid(True)
                    plt.show()    
                    
    data = np.array(data)
    labels = np.array(labels)
    
    #print(f"Forma final de los datos cargados: {data.shape}")
    return data, labels


def cargar_datos_series_test(base_path, dataset_name,centF):
    data = []
    labels = []

    for folder, subfolders, files in os.walk(base_path):
        # Ajustar a las nuevas clases incluyendo los datos de test externo
        if folder.endswith('x50mg_test') or folder.endswith('x100mg_test') or folder.endswith('x150mg_test') or folder.endswith('x200mg_test'):
            class_label = os.path.basename(folder).replace('_test', '')  # Eliminar _test para obtener la clase
            
            # Inicializar base_s21
            base_s21 = None
            
            # Primer bucle para identificar y calcular s21 para el archivo base
            for file in files:
                if file.startswith('base') and file.endswith('.csv'):
                    file_path = os.path.join(folder, file)
                    df_base = pd.read_csv(file_path)
                    
                    # Comprobar las columnas
                    if 'Real Part' in df_base.columns and 'Imaginary Part' in df_base.columns and len(df_base) == 201:
                        # Calcular s21 para el archivo base
                        base_s21 = 20 * np.log10(np.abs(df_base['Real Part'] + 1j * df_base['Imaginary Part']))
                        #base_s21 = (np.abs(df_base['Real Part'] + 1j * df_base['Imaginary Part']))
                        #print(f"Cargando archivo base: {file_path}")
                    else:
                        print(f"El archivo base {file} no tiene las columnas esperadas o no tiene 201 filas")
            
            plt.figure(figsize=(10, 6))
            # Segundo bucle para calcular s21 para los demás archivos
            for file in files:
                if file.endswith('.csv') and not file.startswith('base'):
                    file_path = os.path.join(folder, file)
                    df = pd.read_csv(file_path)

                    # Imprimir las primeras filas del archivo CSV para verificar el contenido
                    #print(f"Cargando archivo: {file_path}")
                    #print(df.head())  # Verifica las columnas y contenido
                    
                    if 'Real Part' in df.columns and 'Imaginary Part' in df.columns and len(df) == 201:
                        # Calcular s21 para el archivo actual
                        s21 = 20 * np.log10(np.abs(df['Real Part'] + 1j * df['Imaginary Part']))
                        #s21 = (np.abs(df['Real Part'] + 1j * df['Imaginary Part']))
                        # Restar el s21 del archivo base
                        s21_corrected = -(s21 - base_s21)
                        #s21_corrected = (s21/base_s21)
                        s21_corrected_interpolated = s21_corrected
                        # idx_min = np.argmax(s21_corrected)
                        # freq_min = df['Freq (Hz)'].iloc[idx_min]
                        # s21_min = s21_corrected.iloc[idx_min]                        
                        
                        # deltaF = centF - freq_min
                        # freq_corrected = df['Freq (Hz)'].values+deltaF 
                        # f_interp = interp1d(freq_corrected, s21_corrected, bounds_error=False, fill_value="extrapolate")
                        # s21_corrected_interpolated = f_interp(df['Freq (Hz)'])  # Interpolamos a las frecuencias originales

                        # Almacenar en el array
                        data.append(s21_corrected_interpolated.values.reshape(-1, 1))
                        # Crear etiquetas jerárquicas
                        # if class_label == 'x100mg':    
                        #     label = f"{dataset_name}_x50mg"
                        # else:
                        #     label = f"{dataset_name}_{class_label}"
                        label = f"{dataset_name}_{class_label}"
                        labels.append(label)
                        plt.plot(df['Freq (Hz)']/1e9, s21_corrected_interpolated, color='gray', alpha=1, label=label)
                    else:
                        print(f"El archivo {file} no tiene las columnas esperadas o no tiene 201 filas")
                    
                    plt.title('All spectra from '+label)
                    plt.xlabel('Frecuencia (GHz)')
                    plt.ylabel(r'20$log_{10}(|S21|)$ (dB)')
                     
                    plt.grid(True)
                    plt.show()                        
                       
    data = np.array(data)
    labels = np.array(labels)
    
    #print(f"Forma final de los datos cargados: {data.shape}")
    return data, labels

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

# Función para seleccionar la carpeta
def seleccionar_carpeta(etiqueta_muestra):
    root = tk.Tk()
    root.withdraw()  
    print(f"Selecciona la carpeta para '{etiqueta_muestra}'")
    return filedialog.askdirectory()

# Paso 2: Seleccionar las carpetas correspondientes
centF = 2.5e9
ruta_x50mg = seleccionar_carpeta('x50mg')
ruta_x100mg = seleccionar_carpeta('x100mg')
ruta_x150mg = seleccionar_carpeta('x150mg')
ruta_x200mg = seleccionar_carpeta('x200mg')

if not ruta_x50mg or not ruta_x100mg or not ruta_x150mg or not ruta_x200mg:
    print("No se seleccionó alguna de las carpetas.")
else:
    #print(f"Cargando datos de x50mg desde: {ruta_x50mg}")
    data_x50mg, labels_x50mg = cargar_datos_series(ruta_x50mg, "Muestra", centF)

    #print(f"Cargando datos de x100mg desde: {ruta_x100mg}")
    data_x100mg, labels_x100mg = cargar_datos_series(ruta_x100mg, "Muestra", centF)

    #print(f"Cargando datos de x150mg desde: {ruta_x150mg}")
    data_x150mg, labels_x150mg = cargar_datos_series(ruta_x150mg, "Muestra", centF)

    #print(f"Cargando datos de x200mg desde: {ruta_x200mg}")
    data_x200mg, labels_x200mg = cargar_datos_series(ruta_x200mg, "Muestra", centF)

    # Combinar los datasets
    data = np.vstack((data_x50mg, data_x100mg, data_x150mg, data_x200mg))
    labels = np.hstack((labels_x50mg, labels_x100mg, labels_x150mg, labels_x200mg))
    
    # Separar en conjunto de entrenamiento (60%), prueba (20%) y validación externa (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.4, stratify=labels, random_state=42)

    # Dividir el conjunto temporal (X_temp, y_temp) en test y validación externa
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Flatten de las secuencias para usar en RandomForest
    X_train_flatten = X_train.reshape((X_train.shape[0], -1))
    X_test_flatten = X_test.reshape((X_test.shape[0], -1))
    X_val_flatten = X_val.reshape((X_val.shape[0], -1))
 
    # Configurar el clasificador
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=10, n_estimators=30,
    min_samples_split=3, max_leaf_nodes=5,random_state=42))

    # Obtener predicciones usando cross-validation en el conjunto de entrenamiento
    y_train_pred = cross_val_predict(clf, X_train_flatten, y_train, cv=5)
    accuracy_cv = accuracy_score(y_train, y_train_pred)

    clf.fit(X_train_flatten, y_train)

    # Evaluar en el conjunto de prueba (test)
    y_test_pred = clf.predict(X_test_flatten)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Evaluar en el conjunto de validación externa
    y_val_pred = clf.predict(X_val_flatten)
    accuracy_val = accuracy_score(y_val, y_val_pred)

    # Mostrar los resultados
    print(f'\nAccuracy en Cross-Validation: {accuracy_cv:.2f}')
    print(f'Accuracy en Test: {accuracy_test:.2f}')
    print(f'Accuracy en Validación Externa: {accuracy_val:.2f}')

    # Generar la matriz de confusión para el conjunto de validación externa
    cm = confusion_matrix(y_val, y_val_pred, labels=clf.named_steps['randomforestclassifier'].classes_)

    total_val_samples = len(y_val)

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clf.named_steps['randomforestclassifier'].classes_,
                yticklabels=clf.named_steps['randomforestclassifier'].classes_)

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(f'Matriz de Confusión (Total muestras de validación: {total_val_samples})', fontsize=14)
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
    pca_thresholds = np.percentile(data_pca, 95, axis=0)
    filtered_indices = np.all(data_pca <= pca_thresholds, axis=1)

    # Filtrar los datos y etiquetas
    data_pca_filtered = data_pca[filtered_indices]
    labels_filtered = labels[filtered_indices]

    # Crear un dataframe para facilitar la visualización
    df = pd.DataFrame(data_pca_filtered, columns=['PCA1', 'PCA2', 'PCA3'])
    df['Label'] = labels_filtered

    # Definir paletas de colores
    color_map = {
        'Muestra_x50mg': 'green',
        'Muestra_x100mg': 'yellow',
        'Muestra_x150mg': 'blue',
        'Muestra_x200mg': 'purple'
    }

    # # Graficar PCA1 vs PCA2
    # plt.figure(figsize=(15, 5))

    # plt.subplot(1, 3, 1)
    # for label, color in color_map.items():
    #     subset = df[df['Label'] == label]
    #     plt.scatter(subset['PCA1'], subset['PCA2'], color=color, label=label, alpha=0.6)
    # plt.title('PCA1 vs PCA2')
    # plt.xlabel('PCA1')
    # plt.ylabel('PCA2')
    # plt.legend()

    # # Graficar PCA1 vs PCA3
    # plt.subplot(1, 3, 2)
    # for label, color in color_map.items():
    #     subset = df[df['Label'] == label]
    #     plt.scatter(subset['PCA1'], subset['PCA3'], color=color, label=label, alpha=0.6)
    # plt.title('PCA1 vs PCA3')
    # plt.xlabel('PCA1')
    # plt.ylabel('PCA3')
    # plt.legend()

    # # Graficar PCA2 vs PCA3
    # plt.subplot(1, 3, 3)
    # for label, color in color_map.items():
    #     subset = df[df['Label'] == label]
    #     plt.scatter(subset['PCA2'], subset['PCA3'], color=color, label=label, alpha=0.6)
    # plt.title('PCA2 vs PCA3')
    # plt.xlabel('PCA2')
    # plt.ylabel('PCA3')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()


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

    # Paso 3: Cargar nuevas muestras para test externo
    print("\nAhora selecciona las nuevas carpetas para el test externo:")

    ruta_ext_x50mg = seleccionar_carpeta('Test Externo x50mg')
    ruta_ext_x100mg = seleccionar_carpeta('Test Externo x100mg')
    ruta_ext_x150mg = seleccionar_carpeta('Test Externo x150mg')
    ruta_ext_x200mg = seleccionar_carpeta('Test Externo x200mg')

    # Cargar las nuevas muestras del test externo
    data_ext_x50mg, labels_ext_x50mg = cargar_datos_series_test(ruta_ext_x50mg, "Muestra", centF)
    data_ext_x100mg, labels_ext_x100mg = cargar_datos_series_test(ruta_ext_x100mg, "Muestra", centF)
    data_ext_x150mg, labels_ext_x150mg = cargar_datos_series_test(ruta_ext_x150mg, "Muestra", centF)
    data_ext_x200mg, labels_ext_x200mg = cargar_datos_series_test(ruta_ext_x200mg, "Muestra", centF)

    # Combinar los datasets de test externo
    data_ext = np.vstack((data_ext_x50mg, data_ext_x100mg, data_ext_x150mg, data_ext_x200mg))
    labels_ext = np.hstack((labels_ext_x50mg, labels_ext_x100mg, labels_ext_x150mg, labels_ext_x200mg))
    
# Asegurarnos de que los datos del test externo estén correctamente aplanados
# Revisar la forma de los datos de test externo
#print(f"Forma de data_ext antes del reshape: {data_ext.shape}")

# Verificar que el reshape esté aplanando correctamente las muestras
# Cambiaremos el reshape para asegurarnos de que no haya errores en el formato de los datos
try:
    data_ext_flatten = data_ext.reshape((data_ext.shape[0], -1))
    #print(f"Forma de data_ext después del reshape: {data_ext_flatten.shape}")
except ValueError as e:
    print(f"Error al hacer reshape: {e}")

# Si las dimensiones son correctas, proceder con la predicción
if data_ext_flatten.shape[1] > 0:  # Comprobamos que haya al menos una característica
    y_ext_pred = clf.predict(data_ext_flatten)
    accuracy_ext = accuracy_score(labels_ext, y_ext_pred)
    print(f'\nAccuracy en Test Externo: {accuracy_ext:.2f}')

    # Generar la matriz de confusión para el test externo
    cm_ext = confusion_matrix(labels_ext, y_ext_pred, labels=clf.named_steps['randomforestclassifier'].classes_)
    
    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_ext, annot=True, fmt='d', cmap='Blues',
                xticklabels=clf.named_steps['randomforestclassifier'].classes_,
                yticklabels=clf.named_steps['randomforestclassifier'].classes_)

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(f'Matriz de Confusión para el Test Externo', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Proyección del test externo en las tres PCs ya calculadas
    data_ext_pca = pca.transform(data_ext_flatten)

    # Crear un dataframe para facilitar la visualización
    df_ext = pd.DataFrame(data_ext_pca, columns=['PCA1', 'PCA2', 'PCA3'])
    df_ext['Label'] = labels_ext

    # Graficar las proyecciones del test externo en 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for label, color in color_map.items():
        subset = df_ext[df_ext['Label'] == label]
        ax.scatter(subset['PCA1'], subset['PCA2'], subset['PCA3'], color=color, label=label, alpha=0.6)

    for label, color in color_map.items():
        subset = df[df['Label'] == label]
        ax.scatter(subset['PCA1'], subset['PCA2'], subset['PCA3'], color=color, label=label, alpha=0.6,marker='*')

    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.set_title('Proyección en 3D del Test Externo en las tres PCs')

    ax.legend()
    plt.show()

else:
    print("Error: No se encontraron características en los datos del test externo después del reshape.")

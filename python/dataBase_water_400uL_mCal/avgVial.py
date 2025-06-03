import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def seleccionar_carpeta():
    root = tk.Tk()
    root.withdraw()

    carpeta = filedialog.askdirectory(title="Selecciona una carpeta que contenga archivos CSV")
    
    if carpeta:
        print(f"Carpeta seleccionada: {carpeta}")
        procesar_archivos_csv(carpeta)
    else:
        print("No se ha seleccionado ninguna carpeta.")

def procesar_archivos_csv(carpeta):
    archivos_csv = [archivo for archivo in os.listdir(carpeta) if archivo.endswith('.csv')]
    archivo_base = next((archivo for archivo in archivos_csv if archivo.startswith('base')), None)
    
    if archivo_base:
        print(f"Archivo base encontrado: {archivo_base}")
        ruta_base = os.path.join(carpeta, archivo_base)
        
        # Leer el archivo base
        df_base = pd.read_csv(ruta_base)
        if set(['Freq (Hz)', 'Real Part', 'Imaginary Part']).issubset(df_base.columns):
            # Calcular el espectro base S21
            s21_base = 20 * np.log10(np.abs(df_base['Real Part'] + 1j * df_base['Imaginary Part']))
            frecuencias_base = df_base['Freq (Hz)'].values
        else:
            print(f"El archivo base {archivo_base} no contiene las columnas esperadas.")
            return
    else:
        print("No se encontró ningún archivo base.")
        return
    
    if archivos_csv:
        print(f"Procesando {len(archivos_csv) - 1} archivos CSV...")
        
        lista_s21 = []
        lista_frecuencias = []
        frecuencias_min = []
        s21_min_values = []

        for archivo in archivos_csv:
            if archivo == archivo_base:  # Saltar el archivo base
                continue

            ruta = os.path.join(carpeta, archivo)
            df = pd.read_csv(ruta)
            
            if set(['Freq (Hz)', 'Real Part', 'Imaginary Part']).issubset(df.columns):
                s21 = 20 * np.log10(np.abs(df['Real Part'] + 1j * df['Imaginary Part']))
                
                # Interpolar el espectro base a la misma frecuencia para restarlo
                interpolador_base = interp1d(frecuencias_base, s21_base, kind='linear', fill_value="extrapolate")
                s21_base_interpolado = interpolador_base(df['Freq (Hz)'].values)
                
                # Restar el espectro base del espectro actual
                s21 -= s21_base_interpolado

                lista_s21.append(s21.values)
                lista_frecuencias.append(df['Freq (Hz)'].values)

                idx_min = np.argmin(s21)
                freq_min = df['Freq (Hz)'].iloc[idx_min]
                s21_min = s21.iloc[idx_min]
                
                frecuencias_min.append(freq_min)
                s21_min_values.append(s21_min)
            else:
                print(f"Advertencia: El archivo {archivo} no contiene las columnas esperadas.")
        
        lista_frecuencias = np.array(lista_frecuencias)
        lista_s21 = np.array(lista_s21)

        # Paso 1: Calcular la frecuencia promedio de los mínimos (como al principio)
        freq_promedio_min = np.mean(frecuencias_min)
        print(f"Frecuencia promedio del mínimo: {freq_promedio_min} Hz")
        
        # Paso 2: Calcular el valor promedio de los mínimos de S21
        s21_min_promedio = np.mean(s21_min_values)
        print(f"Valor promedio del mínimo de S21: {s21_min_promedio} dB")
        
        # Paso 3: Alinear los espectros en torno a la frecuencia promedio del mínimo con interpolación
        freq_centrada, s21_sintetico = alinear_y_generar_espectro_sintetico(
            lista_frecuencias, lista_s21, frecuencias_min, freq_promedio_min
        )
        
        # Paso 4: Guardar el CSV con el espectro sintético
        guardar_csv_sintetico(freq_centrada, s21_sintetico, carpeta)
        
        # Generar gráfico
        generar_grafico(lista_frecuencias, lista_s21, freq_centrada, s21_sintetico)
    else:
        print("No se encontraron archivos CSV en la carpeta seleccionada.")

def alinear_y_generar_espectro_sintetico(frecuencias, s21_list, frecuencias_min, freq_promedio_min):
    """
    Alinear los espectros para que sus mínimos coincidan con la frecuencia promedio del mínimo.
    Luego, interpolar y promediar los valores de S21.
    """
    s21_alineados = []
    frecuencias_alineadas = []
    
    # Definir un rango común de frecuencias para la interpolación
    freq_min_comun = max([min(f) for f in frecuencias])
    freq_max_comun = min([max(f) for f in frecuencias])
    freq_comun = np.linspace(freq_min_comun, freq_max_comun, 1000)  # 1000 puntos comunes

    for i in range(len(frecuencias)):
        # Desplazamiento necesario para alinear el mínimo del espectro en la frecuencia promedio
        desplazamiento = freq_promedio_min - frecuencias_min[i]
        
        # Desplazar las frecuencias para alinear el mínimo
        frec_ajustada = frecuencias[i] + desplazamiento

        # Interpolación del espectro desplazado
        interpolador = interp1d(frec_ajustada, s21_list[i], kind='linear', fill_value="extrapolate")
        s21_interpolado = interpolador(freq_comun)

        # Guardar los espectros alineados e interpolados
        s21_alineados.append(s21_interpolado)

    # Convertir la lista de espectros alineados a array numpy para el promedio
    s21_alineados = np.array(s21_alineados)

    # Promediar los espectros alineados para generar el espectro sintético
    s21_sintetico = np.mean(s21_alineados, axis=0)

    return freq_comun, s21_sintetico

def guardar_csv_sintetico(frecuencias, s21_sintetico, carpeta):
    df_sintetico = pd.DataFrame({
        'Freq (Hz)': frecuencias,
        's21avg': s21_sintetico
    })
    
    ruta_salida = os.path.join(carpeta, 'sinteticVial.csv')
    df_sintetico.to_csv(ruta_salida, index=False)
    print(f"Archivo CSV generado: {ruta_salida}")

def generar_grafico(frecuencias, lista_s21, freq_promedio, s21_sintetico):
    plt.figure(figsize=(10, 6))
    
    for i in range(lista_s21.shape[0]):
        plt.plot(frecuencias[i]/1e9, lista_s21[i], color='gray', alpha=0.5, label=f'Espectro {i+1}' if i == 0 else "")
    
    plt.plot(freq_promedio/1e9, s21_sintetico, color='red', linewidth=2.5, label='Espectro sintético (promedio)')
    
    plt.title('All spectra and sintetic spectrum')
    plt.xlabel('Frecuencia (GHz)')
    plt.ylabel(r'20$log_{10}(|S21|)$ (dB)')
    #plt.legend()
    
    plt.grid(True)
    plt.show()

seleccionar_carpeta()

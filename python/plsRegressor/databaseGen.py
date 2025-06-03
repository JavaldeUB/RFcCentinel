# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 06:38:56 2024

@author: javit
"""

import os
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog

def load_csv_files_from_folder(folder_path):
    """
    Carga archivos CSV desde una carpeta específica, excluyendo los archivos que comienzan con "baseline"
    y el archivo "sinteticVial.csv". Calcula el valor absoluto y la fase de Real + i*Imaginary para cada archivo,
    y organiza los datos en DataFrames de magnitudes y fases.

    Args:
        folder_path (str): La ruta de la carpeta que contiene los archivos CSV.

    Returns:
        combined_df (pd.DataFrame): DataFrame con los datos originales y la etiqueta.
        magnitude_df (pd.DataFrame): DataFrame con magnitudes y frecuencia organizadas por filas de archivo.
        phase_df (pd.DataFrame): DataFrame con fases y frecuencia organizadas por filas de archivo.
    """
    folder_name = os.path.basename(folder_path)
    etiqueta_numerica = int(folder_name.replace("x", ""))  # Quitar 'x' y convertir a número
    combined_df = pd.DataFrame()  # DataFrame para almacenar los archivos originales de esta carpeta
    magnitude_data = []  # Lista para almacenar las filas de magnitudes
    phase_data = []  # Lista para almacenar las filas de fases

    freq_axis = None  # Inicializamos para capturar la frecuencia una vez

    # Procesar archivos en la carpeta
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv") and not file_name.startswith("baseline") and file_name != "sinteticVial.csv":
            file_path = os.path.join(folder_path, file_name)
            temp_df = pd.read_csv(file_path)
            temp_df['etiqueta'] = etiqueta_numerica  # Añadir la etiqueta numérica
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

            # Guardar el eje de frecuencias solo una vez
            if freq_axis is None:
                freq_axis = temp_df["Freq (Hz)"].values

            # Calcular el valor absoluto de Real Part + i * Imaginary Part
            magnitude = np.sqrt(temp_df["Real Part"]**2 + temp_df["Imaginary Part"]**2)
            row_magnitude = [etiqueta_numerica] + list(magnitude)
            magnitude_data.append(row_magnitude)

            # Calcular la fase de Real Part + i * Imaginary Part
            phase = np.arctan2(temp_df["Imaginary Part"], temp_df["Real Part"])  # Calcula la fase en radianes
            row_phase = [etiqueta_numerica] + list(phase)
            phase_data.append(row_phase)

    # Crear DataFrame de magnitudes
    freq_columns = ["etiqueta"] + ["Freq (Hz)_" + str(i+1) for i in range(len(magnitude))]  # Nombres de columna de frecuencias
    magnitude_df = pd.DataFrame(magnitude_data, columns=freq_columns)
    phase_df = pd.DataFrame(phase_data, columns=freq_columns)

    # Añadir la fila de frecuencia como la primera fila de ambos DataFrames
    freq_row = ["Frecuencia (Hz)"] + list(freq_axis)
    magnitude_df.loc[-1] = freq_row  # Añadir fila en el índice -1 para que esté al inicio
    phase_df.loc[-1] = freq_row
    magnitude_df.index = magnitude_df.index + 1  # Desplazar índice para que la fila se ubique correctamente
    phase_df.index = phase_df.index + 1
    magnitude_df = magnitude_df.sort_index()  # Reordenar el índice
    phase_df = phase_df.sort_index()

    return combined_df, magnitude_df, phase_df

def select_folder_and_load_csvs():
    """
    Permite al usuario seleccionar carpetas una a una, combina todos los CSVs en un DataFrame combinado,
    y crea DataFrames de magnitudes y fases de Real + i*Imaginary organizados por filas de archivo.

    Returns:
        pd.DataFrame: DataFrame combinado con los datos originales y etiquetas.
        pd.DataFrame: DataFrame combinado con las magnitudes y frecuencias organizadas por filas de archivo.
        pd.DataFrame: DataFrame combinado con las fases y frecuencias organizadas por filas de archivo.
    """
    all_data_df = pd.DataFrame()
    all_magnitudes_df = pd.DataFrame()
    all_phases_df = pd.DataFrame()

    # Configurar Tkinter para abrir el diálogo de selección de carpetas
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter

    while True:
        print("Selecciona una carpeta que contenga archivos CSV (o cancela para finalizar):")
        folder_path = filedialog.askdirectory(title="Selecciona una carpeta con archivos CSV")
        
        if not folder_path:
            print("No se seleccionaron más carpetas.")
            break
        
        # Cargar los archivos CSV de la carpeta seleccionada
        folder_data_df, folder_magnitude_df, folder_phase_df = load_csv_files_from_folder(folder_path)
        
        # Combinar datos originales, magnitudes y fases en sus respectivos DataFrames
        all_data_df = pd.concat([all_data_df, folder_data_df], ignore_index=True)
        all_magnitudes_df = pd.concat([all_magnitudes_df, folder_magnitude_df], ignore_index=True)
        all_phases_df = pd.concat([all_phases_df, folder_phase_df], ignore_index=True)

        # Pregunta al usuario si quiere seleccionar otra carpeta
        continuar = input("¿Quieres añadir otra carpeta? (s/n): ")
        if continuar.lower() != 's':
            break

    return all_data_df, all_magnitudes_df, all_phases_df

df_original, df_magnitudes, df_fases = select_folder_and_load_csvs()
df_original.to_csv('s21_allEthanol_RealImg.csv', sep=',', encoding='utf-8', index=False, header=True)
df_magnitudes.to_csv('s21_allEthanol_mag.csv', sep=',', encoding='utf-8', index=False, header=True)
df_fases.to_csv('s21_allEthanol_ph.csv', sep=',', encoding='utf-8', index=False, header=True)
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:02:08 2024

@author: javit
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Función para seleccionar una carpeta
def select_folder():
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de tkinter
    folder_path = filedialog.askdirectory()  # Abre la ventana de selección de carpeta
    return folder_path

# Selecciona la carpeta
folder_path = select_folder()

# Verifica si se seleccionó una carpeta
if not folder_path:
    print("No se seleccionó ninguna carpeta")
    exit()

# Obtiene la lista de archivos .csv en la carpeta
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Verifica si hay archivos .csv en la carpeta
if not csv_files:
    print("No se encontraron archivos .csv en la carpeta seleccionada")
    exit()

# Inicializa una figura para el gráfico conjunto
plt.figure()

# Recorre cada archivo .csv
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    
    # Lee el archivo .csv y lo guarda en un DataFrame de pandas
    df = pd.read_csv(file_path)
    
    # Verifica que las columnas 'RealPart', 'ImaginaryPart' y 'Freq_Hz_' existan
    if {'Real Part', 'Imaginary Part', 'Freq (Hz)'}.issubset(df.columns):
        # Calcula s21 = RealPart + i*ImaginaryPart
        s21 = df['Real Part'] + 1j * df['Imaginary Part']
        
        # Añade una columna con el valor absoluto de s21
        df['abs_s21'] = 20*np.log10(np.abs(s21))
        
        # Muestra que se ha añadido la columna 's21' y 'abs_s21'
        print(f"Columns s21 y abs_s21 added to: {file}")
        
        # Plot de abs(s21) vs Freq_Hz_/1e9 (frecuencia en GHz)
        plt.plot(df['Freq (Hz)'] / 1e9, df['abs_s21'])
    else:
        print(f"Columns Real Part, Imaginary Part o Freq (Hz) were not found: {file}")

# Configuración del gráfico
plt.xlabel('Freq. (GHz)')
plt.ylabel(r'$|s_{21}|$')
plt.title(r'$|s_{21}|$'+' vs Frequency')
plt.grid(True)

# Mostrar el gráfico
plt.show()

print("Every file has been processed.")

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:09:56 2024

@author: javit
"""

import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


def cargar_datos(file_path):
    df = pd.read_csv(file_path)
    return df
# Función para seleccionar la carpeta

def seleccionar_archivo(etiqueta_muestra):
    root = tk.Tk()
    root.withdraw()  
    print(f"Selecciona el archivo '{etiqueta_muestra}'")
    return filedialog.askopenfile()

def transform(x, y):
    Tr = y.T @ np.linalg.pinv(x)
    return Tr

path2genericVial = seleccionar_archivo('generic Vial')
path2sinteticVial = seleccionar_archivo('sintetic Vial')
path2test = seleccionar_archivo('Test')

df_gen = cargar_datos(path2genericVial)
df_sint = cargar_datos(path2sinteticVial)
df_test = cargar_datos(path2test)

Tr = transform(df_sint['s21avg'].values,df_gen['s21avg'].values)

s21Test = 20*np.log10(np.abs(df_test['Real Part'].values+1j*df_test['Imaginary Par'].values))
s21Trt = s21Test @ Tr
plt.figure(figsize=(10, 6))


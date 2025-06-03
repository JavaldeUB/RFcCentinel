# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:34:21 2024

@author: javit
"""
import pandas as pd
from tkinter import Tk, filedialog

class FileProcessor:
    def __init__(self):
        self.genVial_data = None
        self.sinteticVial_data = None
        self.deltaF = None
        self.deltaS21 = None

    def open_file_dialog(self, title="Select File"):
        """Abre un diálogo para seleccionar un archivo CSV"""
        root = Tk()
        root.withdraw()  # Ocultar la ventana principal de Tkinter
        file_path = filedialog.askopenfilename(title=title, filetypes=[("CSV files", "*.csv")])
        return file_path

    def read_csv(self, file_path):
        """Lee el archivo CSV y retorna un DataFrame de pandas"""
        return pd.read_csv(file_path)

    def process_files(self):
        """Método principal para procesar los archivos CSV y calcular deltaF y deltaS21"""
        # 1. Selección del archivo de genVial
        genVial_path = self.open_file_dialog("Seleccionar archivo de genVial")
        self.genVial_data = self.read_csv(genVial_path)
        
        # 2. Selección del archivo de sinteticVial
        sinteticVial_path = self.open_file_dialog("Seleccionar archivo de sinteticVial")
        self.sinteticVial_data = self.read_csv(sinteticVial_path)

        # 3. Encontrar el mínimo de s21avg en ambos archivos
        min_genVial_s21avg = self.genVial_data['s21avg'].min()
        min_sinteticVial_s21avg = self.sinteticVial_data['s21avg'].min()

        # 4. Encontrar las frecuencias correspondientes al mínimo de s21avg
        freq_genVial_min = self.genVial_data[self.genVial_data['s21avg'] == min_genVial_s21avg]['Freq (Hz)'].values[0]
        freq_sinteticVial_min = self.sinteticVial_data[self.sinteticVial_data['s21avg'] == min_sinteticVial_s21avg]['Freq (Hz)'].values[0]

        # 5. Calcular deltaF y deltaS21
        self.deltaF = freq_genVial_min - freq_sinteticVial_min
        self.deltaS21 = min_genVial_s21avg - min_sinteticVial_s21avg

        # 6. Retornar deltaF y deltaS21
        return self.deltaF, self.deltaS21

# Uso de la clase
if __name__ == "__main__":
    processor = FileProcessor()
    deltaF, deltaS21 = processor.process_files()
    print(f"deltaF: {deltaF}")
    print(f"deltaS21: {deltaS21}")


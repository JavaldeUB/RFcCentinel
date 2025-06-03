import os
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

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

def select_DBScsv_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select a CSV file"
    )
    return file_path

def optimise_pls_cv(X, y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components=n_comp)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)

    # Calculate scores
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    
    return (y_cv, r2, mse, rpd)

# Create a sample dataframe for demonstration
database_file = select_DBScsv_file()
df_magnitudes = pd.read_csv(database_file)
# Separate features and labels
df = df_magnitudes.iloc[1:,:]
non_numeric_mask = df.applymap(lambda x: pd.to_numeric(x, errors='coerce')).isna()
rows_with_strings = non_numeric_mask.any(axis=1)
string_indices = df.index[rows_with_strings]
df_strings = df.loc[rows_with_strings]  # Rows containing strings
df_cleaned = df.drop(index=string_indices)  # DataFrame without strings
numeric_cols = df_cleaned.select_dtypes(include=[np.number])
# Apply log10 to numeric columns only
df_cleaned[numeric_cols.columns] = numeric_cols.apply(np.log10)

X = df_cleaned.drop(columns=['etiqueta']).values
y = df_cleaned['etiqueta'].values
X2 = savgol_filter(X, 17, polyorder=2, deriv=2)

r2s = []
mses = []
rpds = []
xticks = np.arange(1, 41)
for n_comp in xticks:
    y_cv, r2, mse, rpd = optimise_pls_cv(X2, y, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)


print(f"Best parameters: {best_params}")
print(f"RMSECV: {best_score}")
print(f"R² CV: {best_r2_cv}")
# Step 3: Test on the left-out labels and plot results
best_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=best_params['poly__degree'])),
    ('pls', PLSRegression(n_components=best_params['pls__n_components']))
])
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

# Calculate RMSE for the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)
print(f"RMSE of test: {rmse_test}")
print(f"R² of test: {r2_test}")

# Bland-Altman plot with Limits of Agreement at 95% confidence interval
mean_diff = np.mean(y_test - y_pred_test)
std_diff = np.std(y_test - y_pred_test)
loa_upper = mean_diff + 1.96 * std_diff
loa_lower = mean_diff - 1.96 * std_diff

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted vs Real')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.plot(y_test,loa_upper, color='r', linestyle='--', label='LoA Upper')
plt.plot(y_test,loa_lower, color='g', linestyle='--', label='LoA Lower')

plt.xlabel('Real Labels')
plt.ylabel('Predicted Labels')
plt.title('Real vs Predicted Labels with Limits of Agreement (Bland-Altman)')
plt.legend()
plt.text(0.05, 0.95, f'RMSECV: {best_score:.2f}\nR² CV: {best_r2_cv:.2f}\nRMSE of test: {rmse_test:.2f}\nR² of test: {r2_test:.2f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()

# Bland-Altman plot
mean_values = (y_test + y_pred) / 2
diff_values = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(mean_values, diff_values, color='blue', label='Difference vs Mean')
plt.axhline(mean_diff, color='black', linestyle='--', label='Mean Difference')
plt.axhline(loa_upper, color='r', linestyle='--', label='LoA Upper')
plt.axhline(loa_lower, color='g', linestyle='--', label='LoA Lower')

plt.xlabel('Mean of Real and Predicted Labels')
plt.ylabel('Difference between Real and Predicted Labels')
plt.title('Bland-Altman Plot')
plt.legend()
plt.text(0.05, 0.95, f'RMSECV: {best_score:.2f}\nRMSE of test: {rmse_test:.2f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()
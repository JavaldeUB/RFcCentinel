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
df_temp = df_cleaned.iloc[2:, :]
X = df_temp.drop(columns=['etiqueta']).values
y = df_temp['etiqueta'].reset_index(drop=True)
X2 = savgol_filter(X, 17, polyorder=2, deriv=2)
X = pd.DataFrame(X2)

# Step 1: Separate 3 labels randomly, ensuring 96 and 20 are always in the training set
labels_to_keep = [96, 20]
remaining_labels = list(set(y.unique()) - set(labels_to_keep))
np.random.seed(42)  # For reproducibility
labels_to_leave_out = np.random.choice(remaining_labels, 3, replace=False)

# Create training and testing datasets
train_mask = ~y.isin(labels_to_leave_out)
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

# Randomize the training dataset
train_data = pd.concat([X_train, y_train], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)
X_train = train_data.drop(columns=['etiqueta'])
y_train = train_data['etiqueta']


# Step 2: Cross-validation to optimize polynomial degree and number of components
param_grid = {
    'poly__degree': [1,2,3],
    'pls__n_components': [2,3,4]
}

logo = LeaveOneGroupOut()

# Initialize plot for real-time RMSECV evolution
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Iteration')
ax.set_ylabel('RMSECV')
ax.set_title('RMSECV Evolution')
line, = ax.plot([], [], 'b-')
plt.show()

rmsecv_values = []
best_score = float('inf')
best_params = None

# Manually iterate over the parameter grid
for degree in param_grid['poly__degree']:
    print(f"Degree of the polyPLS: {degree}")
    for n_components in param_grid['pls__n_components']:
        print(f"Components of the polyPLS: {n_components}")
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('pls', PLSRegression(n_components=n_components))
        ])
        
        scores = []
        r2_scores_cv = []
        for train_index, test_index in logo.split(X_train, y_train, groups=y_train):
            X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
            
            pipeline.fit(X_train_cv, y_train_cv)
            y_pred_cv = pipeline.predict(X_test_cv)
            scores.append(mean_squared_error(y_test_cv, y_pred_cv))
            r2_scores_cv.append(r2_score(y_test_cv, y_pred_cv))
                        
        rmsecv = np.sqrt(np.mean(scores))
        print(f"RMSECV: {rmsecv}")
        print(f"R² CV: {r2_scores_cv[-1]}")
        rmsecv_values.append(rmsecv)
        
        if rmsecv < best_score:
            best_score = rmsecv
            best_params = {'poly__degree': degree, 'pls__n_components': n_components}
            best_r2_cv = np.mean(r2_scores_cv)
            
        # Update plot
        line.set_xdata(range(len(rmsecv_values)))
        line.set_ydata(rmsecv_values)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
        plt.show()

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
mean_diff = np.mean(y_test - y_pred)
std_diff = np.std(y_test - y_pred)
loa_upper = mean_diff + 1.96 * std_diff
loa_lower = mean_diff - 1.96 * std_diff

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Real')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.axhline(y=loa_upper, color='r', linestyle='--', label='LoA Upper')
plt.axhline(y=loa_lower, color='g', linestyle='--', label='LoA Lower')

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
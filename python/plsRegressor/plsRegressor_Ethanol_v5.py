import os
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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

def optimise_pls_cv_manual(X, y, n_comp, degree):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('pls', PLSRegression(n_components=n_comp))
    ])    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    r2s, mses, rpds = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test).ravel()
        
        # Compute metrics for each fold
        r2s.append(r2_score(y_test, y_pred))
        mses.append(mean_squared_error(y_test, y_pred))
        rpds.append(y_test.std() / np.sqrt(mean_squared_error(y_test, y_pred)))
    
    return np.mean(r2s), np.mean(mses), np.mean(rpds)

# Plot the mses
def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.figure()  # Create a new figure for each plot
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective == 'min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.ylabel(ylabel)
        plt.title(f'PLS - {ylabel}')

        # Show plot
        plt.show()


    
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
y = df_temp['etiqueta'].reset_index(drop=True).values.astype(float)
X2 = savgol_filter(X, 17, polyorder=2, deriv=2)

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42, stratify=y)

# Test with N components using training data
r2s = []
mses = []
rpds = []
xticks = np.arange(1, 11)

for n_comp in xticks:
    print(f"LVs used: {n_comp}")
    r2, mse, rpd = optimise_pls_cv_manual(X_train, y_train, n_comp, 2)  # Perform optimisation on training data
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)

# Plot metrics to determine the best number of components
plot_metrics(mses, 'MSE', 'min')
plot_metrics(rpds, 'RPD', 'max')

# Evaluate the model on test data using the optimal number of components (e.g., 11 here)
optimal_n_comp = 15
pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('pls', PLSRegression(n_components=n_comp))
])
pipeline.fit(X_train, y_train)
y_pred_test = pipeline.predict(X_test)

# Evaluate performance on the test set
r2_test = r2_score(y_test, y_pred_test)
mse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
rpd_test = y_test.std() / np.sqrt(mse_test)

print(f"Test Set Results - R2: {r2_test:.4f}, RMSE: {mse_test:.4f}, RPD: {rpd_test:.4f}")
# Calculate the differences and averages
averages = (y_pred_test + y_test) / 2
differences = y_test - y_pred_test
mean_diff = np.mean(y_test - y_pred_test)
std_diff = np.std(y_test - y_pred_test)
loa_upper = mean_diff + 1.96 * std_diff
loa_lower = mean_diff - 1.96 * std_diff
loa_upper_dynamic = loa_upper + (averages)
loa_lower_dynamic = loa_lower + (averages)


plt.figure(figsize=(10, 6))
plt.scatter( y_test,y_pred_test, color='blue', label='Predicted vs Real')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k-', lw=2)
plt.plot(averages,loa_lower_dynamic, color='r', linestyle='-', label=f'{loa_lower:.3f}')
plt.plot(averages,loa_upper_dynamic, color='g', linestyle='-', label=f'{loa_upper:.3f}')

plt.xlabel('Real Labels')
plt.ylabel('Predicted Labels')
plt.title('Real vs Predicted Labels with Limits of Agreement (Bland-Altman)')
plt.legend()
plt.show()

# Add RMSE and R^2 as text annotations
plt.text(0.05, 0.95, f'RMSEP: {mse_test:.2f}\nR²: {r2_test:.2f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# # Plot actual vs predicted for test data
# plt.figure(figsize=(6, 6))
# with plt.style.context('ggplot'):
#     plt.scatter(y_test, y_pred_test, color='red', label='Test data')
#     plt.plot(y_test, y_test, '-g', label='Expected regression line')
#     z = np.polyfit(y_test.flatten(), y_pred_test.flatten(), 1)
#     plt.plot(y_test, np.polyval(z, y_test), color='blue', label='Predicted regression line')
#     plt.xlabel('Actual')
#     plt.ylabel('Predicted')
#     plt.legend()
#     plt.show()



% Abre una ventana para seleccionar una carpeta
folder_path = uigetdir;

% Verifica si se ha seleccionado una carpeta
if folder_path == 0
    disp('No se seleccionó ninguna carpeta');
    return;
end

% Obtiene la lista de archivos .csv en la carpeta
csv_files = dir(fullfile(folder_path, '*.csv'));

% Verifica si hay archivos .csv en la carpeta
if isempty(csv_files)
    disp('No se encontraron archivos .csv en la carpeta seleccionada');
    return;
end

% Inicializa una celda para almacenar las tablas
tables = cell(1, length(csv_files));

% Inicializa una figura para el plot conjunto
figure;
hold on; % Permite sobreponer gráficos

% Recorre cada archivo .csv
for i = 1:length(csv_files)
    % Obtiene el nombre completo del archivo
    file_name = fullfile(folder_path, csv_files(i).name);
    
    % Lee el archivo .csv y lo guarda en una tabla
    tables{i} = readtable(file_name);
    
    % Verifica que las columnas 'RealPart', 'ImaginaryPart' y 'Freq' existan en la tabla
    if all(ismember({'RealPart', 'ImaginaryPart', 'Freq_Hz_'}, tables{i}.Properties.VariableNames))
        % Calcula s21 = RealPart + i*ImaginaryPart
        tables{i}.s21 = tables{i}.RealPart + 1i * tables{i}.ImaginaryPart;
        
        % Añade una columna con el valor absoluto de s21
        tables{i}.abs_s21 = abs(tables{i}.s21);
        
        % Muestra que se ha añadido la columna 's21' y 'abs_s21'
        disp(['Columna s21 y abs_s21 añadidas a: ', csv_files(i).name]);
        
        % Plot de abs(s21) vs Freq/1e9 (frecuencia en GHz)
        plot(tables{i}.Freq_Hz_ / 1e9, tables{i}.abs_s21, 'DisplayName', csv_files(i).name);
    else
        disp(['Las columnas RealPart, ImaginaryPart o Freq no se encontraron en: ', csv_files(i).name]);
    end
end

% Configuración del gráfico
xlabel('Frecuencia (GHz)');
ylabel('|s21|');
title('Gráfico conjunto de |s21| vs Frecuencia');
grid on; % Añade una cuadrícula al gráfico
hold off; % Termina el hold para el gráfico

% Mostrar un mensaje al final
disp('Todos los archivos han sido procesados y el gráfico ha sido creado.');

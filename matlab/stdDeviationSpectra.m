% Cargar la base de datos
T = readtable('organized_data_with_baseline.csv');

% Obtener las columnas numéricas (frecuencias)
data = T{:, 1:201}; % Ajusta si hay más columnas de frecuencia
concentration = T.Concentration;
time = T.Time;

% Crear matriz vacía para los datos normalizados
normalized_data = [];

% Obtener tiempos únicos
unique_times = unique(time);

for i = 1:length(unique_times)
    % Filtrar por tiempo
    time_filter = time == unique_times(i);
    time_data = data(time_filter, :);
    time_concentration = concentration(time_filter);
    
    % Encontrar la fila con concentración 0 (baseline)
    baseline = time_data(time_concentration == 0, :);
    
    % Restar la baseline a cada fila con concentración distinta de 0
    for j = 1:length(time_concentration)
        if time_concentration(j) ~= 0
            normalized_row = time_data(j, :) - baseline;
            normalized_data = [normalized_data; normalized_row];
        end
    end
end

% Guardar la matriz normalizada
writematrix(normalized_data, 'normalized_data.csv');

disp('Normalización completada y guardada.')

% Separar los datos normalizados por concentración
concentration_levels = unique(concentration(concentration ~= 0));
normalized_data_groups = cell(length(concentration_levels), 1);

for k = 1:length(concentration_levels)
    index = concentration == concentration_levels(k);
    normalized_data_groups{k} = normalized_data(index(concentration ~= 0), :);
end

disp('Datos separados por concentración.')

% Promediar las filas de cada concentración
averaged_data = zeros(length(concentration_levels), size(normalized_data, 2));

for k = 1:length(concentration_levels)
    averaged_data(k, :) = mean(normalized_data_groups{k}, 1);
end

% Mostrar mensaje de finalización
disp('Datos promediados por concentración.')
freq = linspace(1.6e9, 3e9, size(averaged_data, 2)); % Frecuencias de 1.6GHz a 3GHz
% Encontrar mínimos y sus posiciones en frecuencia
% Encontrar mínimos y sus posiciones en frecuencia
min_freq_positions = zeros(length(concentration_levels), 1);
std_freq_positions = zeros(length(concentration_levels), 1);

for k = 1:length(concentration_levels)
    [~, min_pos] = min(averaged_data(k, :));
    min_freq_positions(k) = freq(min_pos);
    
    % Encontrar las posiciones de los mínimos para cada fila
    min_positions = zeros(size(normalized_data_groups{k}, 1), 1);
    for j = 1:size(normalized_data_groups{k}, 1)
        [~, pos] = min(normalized_data_groups{k}(j, :));
        min_positions(j) = freq(pos);
    end
    
    % Calcular la desviación estándar de las posiciones de los mínimos
    std_freq_positions(k) = std(min_positions);
end

% Mostrar mínimos encontrados con desviación estándar
disp('Frecuencias de mínimos por concentración con desviación estándar:');
disp(table(concentration_levels, min_freq_positions, std_freq_positions, 'VariableNames', {'Concentration', 'MinFrequency', 'StdDeviation'}));

%Después de pasar por el cftool (cfit_averageSpectra.sfit)
p1 = 111.7362;
p2 = -259.1698;
q = -2.28;
RMSE = 8.9211;
Rsquare = 0.9571;

red_concentrations = cat(2,repmat(40,1,50),repmat(60,1,50),repmat(80,1,50));
% Encontrar las posiciones de los mínimos para cada fila
red_normalized_data_groups = {normalized_data_groups{4}; normalized_data_groups{6}; normalized_data_groups{8}};
min_positions_total = [];
for k = 1:3    
    % Encontrar las posiciones de los mínimos para cada fila
    min_positions = zeros(size(red_normalized_data_groups{k}, 1), 1);
    for j = 1:size(normalized_data_groups{k}, 1)
        [~, pos] = min(normalized_data_groups{k}(j, :));
        min_positions(j) = freq(pos);
    end
    
    min_positions_total = cat(1,min_positions_total,min_positions./1e9);
end

predCon = (p1.*min_positions_total+p2)./(min_positions_total+q);
rmsTotal = sqrt(sqrt(mean((red_concentrations.'-predCon).^2)));

red_min_freq_positions = [min_freq_positions(4), min_freq_positions(6), min_freq_positions(8)];
predCon = (p1.*red_min_freq_positions+p2)./(red_min_freq_positions+q);
rmseTotal =rmse(red_concentrations,predCon);
% Graficar los datos promediados
% freq = linspace(1.6e9, 3e9, size(averaged_data, 2)); % Frecuencias de 1.6GHz a 3GHz
% figure;
% hold on;
% colors = jet(length(concentration_levels));
% 
% for k = 1:length(concentration_levels)
%     plot(freq, 20*log10(abs(averaged_data(k, :))), 'Color', colors(k, :), 'DisplayName', sprintf('%d ppm', concentration_levels(k)));
% end
% 
% xlabel('Frecuencia (Hz)');
% ylabel('20*log10(|S21|)');
% title('Datos Normalizados Promediados');
% legend show;
% grid on;
% disp('Gráfico generado.')

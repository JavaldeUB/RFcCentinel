function folderPath = seleccionar_carpeta()
    % Create a file selection dialog
    folderPath = uigetdir('Select a folder containing CSV files');
    
    if folderPath
        disp(['Selected folder: ', folderPath]);
        procesar_archivos_csv(folderPath);
    else
        disp('No folder selected.');
    end
end
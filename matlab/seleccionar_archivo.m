function fullpath = seleccionar_archivo(startFolder)
%SELECT_EXISTING_FILE Abre una ventana para seleccionar un archivo existente.
%   INPUT:
%     startFolder - Carpeta inicial desde la cual se abre el diálogo.
%   OUTPUT:
%     fullpath    - Ruta completa del archivo seleccionado.

    % Validar la carpeta de entrada
    if ~ischar(startFolder) && ~isstring(startFolder)
        error('The path must be a chain of characters.');
    end

    % Si no existe la carpeta, error
    if ~exist(startFolder, 'dir')
        error('The especified route does not extist: %s', startFolder);
    end

    % Lanzar diálogo para seleccionar archivo
    [fileName, pathName] = uigetfile({'*.csv', 'Comma separated files (*.csv)'}, ...
                                     'Select a file', ...
                                     startFolder);

    % Si el usuario cancela
    if isequal(fileName, 0) || isequal(pathName, 0)
        fullpath = '';
        disp('Selection cancel by the user.');
        return;
    end

    % Ruta completa del archivo
    fullpath = fullfile(pathName, fileName);
end

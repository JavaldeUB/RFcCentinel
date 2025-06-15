function fullpath = salvar_archivo(folderPath)
%DEFINE_SAVE_PATH Abre una ventana para guardar archivo en folderPath.
%   INPUT:
%     folderPath - Ruta inicial de la carpeta donde se guardará el archivo.
%   OUTPUT:
%     fullpath   - Ruta completa del archivo seleccionado por el usuario.

    % Validar que folderPath es válido
    if ~ischar(folderPath) && ~isstring(folderPath)
        error('Path must be a chain of characters.');
    end

    % Crear carpeta si no existe
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end

    % Lanzar diálogo para guardar archivo
    [fileName, pathName] = uiputfile({'*.csv','Comma separated file (*.csv)'}, ...
                                     'Save File As', ...
                                     fullfile(folderPath, 'datos.csv'));

    % Verificar si el usuario canceló
    if isequal(fileName, 0) || isequal(pathName, 0)
        fullpath = '';
        disp('Operation cancel by the user.');
        return;
    end

    % Construir ruta completa
    fullpath = fullfile(pathName, fileName);
end

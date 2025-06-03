% Script to compile x[label]_n.csv files into a single CSV file
% Each row corresponds to a file, with columns for complex values
% and an extra column for the label.
% Additionally, generates a baseline CSV file with the baseline of each label.

% Define the labels to look for
labels = [96, 80, 70, 60, 50, 40, 30, 20, 0];
warning('off','all')
warning
% Prompt user to select the general folder containing manyDaysFolders
disp('Select the general folder containing the database');
generalFolder = uigetdir('', 'Select General Folder');
if generalFolder == 0
    disp('No folder selected. Exiting...');
    return;
end

% Preallocate cell arrays to store main data and baseline data
compiledData = {};
baselineData = {};
sinteticData = {};
frequencyPoints = [];

% Get all subfolders of the general folder and its subdirectories
allFolders = dir(fullfile(generalFolder, '**/*.*/*.*')); % Get all items in the directory tree
allFolders = allFolders([allFolders.isdir]); % Keep only directories
%%
%%Itarete through each label
for labelIdx = 1:length(allFolders)
    if find(allFolders(labelIdx).name == 'x') 
        currentFolder = fullfile(allFolders(labelIdx).folder, allFolders(labelIdx).name);

        label = str2num(allFolders(labelIdx).name(end-1:end));
        subfolderName = ['x', num2str(label)];
        subfolderPath = currentFolder;
        
        % Check if the subfolder exists
        if isfolder(subfolderPath)
            disp(['Processing subfolder: ', subfolderPath]);
            
            % Process the baseline file
            baselineFile = fullfile(subfolderPath, 'baseline1p6to3GHz.csv');
            if isfile(baselineFile)
                disp(['Reading baseline file: ', baselineFile]);
                baselineTable = readtable(baselineFile);

                % Check that the file has the required columns
                if all(ismember({'Freq_Hz_', 'RealPart', 'ImaginaryPart'}, baselineTable.Properties.VariableNames))
                    % Extract frequency points if not already set
                    if isempty(frequencyPoints)
                        frequencyPoints = baselineTable.('Freq_Hz_');
                    end

                    % Compute the baseline complex values
                    baselineComplex = baselineTable.('RealPart') + 1i * baselineTable.('ImaginaryPart');

                    % Store the baseline as a single row with the label
                    baselineData = [baselineData; {baselineComplex.', label}];
                else
                    warning(['Baseline file ', baselineFile, ' does not have the required columns. Skipping...']);
                end
            else
                warning(['Baseline file not found in subfolder: ', subfolderPath]);
            end

            sinteticFile = fullfile(subfolderPath, 'sinteticVial.csv');
            if isfile(sinteticFile)
                disp(['Reading sintetic file: ', sinteticFile]);
                sinteticTable = readtable(sinteticFile);

                % Check that the file has the required columns
                if all(ismember({'Freq_Hz_', 's21avg'}, sinteticTable.Properties.VariableNames))
                    % Extract frequency points if not already set
                    frequencyPointsS = sinteticTable.('Freq_Hz_');
                    
                    % Compute the baseline complex values
                    sintetic = sinteticTable.('s21avg');

                    % Store the baseline as a single row with the label
                    sinteticData = [sinteticData; {sintetic.', label}];
                else
                    warning(['Sintetic file ', sinteticFile, ' does not have the required columns. Skipping...']);
                end
            else
                warning(['sinteticVial file not found in subfolder: ', subfolderPath]);
            end
            
            % Get all x[label]_n.csv files in the subfolder
            csvPattern = fullfile(subfolderPath, ['x', num2str(label), 'mg_*.csv']);
            csvFiles = dir(csvPattern);
            if isempty(csvFiles)
                csvPattern = fullfile(subfolderPath, ['x', num2str(label), '_*.csv']);
                csvFiles = dir(csvPattern);
            end
                

            % Process each file
            for fileIdx = 1:length(csvFiles)
                filePath = fullfile(csvFiles(fileIdx).folder, csvFiles(fileIdx).name);
                disp(['Reading file: ', filePath]);

                % Read the file and extract the 'Real Part' and 'Imaginary Part'
                fileData = readtable(filePath);

                % Check that the file has the required columns
                if all(ismember({'Freq_Hz_', 'RealPart', 'ImaginaryPart'}, fileData.Properties.VariableNames))
                    % Verify frequency consistency
                    if isempty(frequencyPoints)
                        frequencyPoints = fileData.('Freq_Hz_');
                    elseif ~isequal(frequencyPoints, fileData.('Freq_Hz_'))
                        error('Frequency points mismatch in file: %s', filePath);
                    end

                    % Compute the complex values: Real Part + i*Imaginary Part
                    complexValues = fileData.('RealPart') + 1i * fileData.('ImaginaryPart');

                    % Convert complex values to a row vector
                    rowData = complexValues.';

                    % Append the label at the end (as a cell to avoid conversion to complex)
                    rowData = [rowData, {label}];

                    % Add to compiled data
                    compiledData = [compiledData; rowData];
                else
                    warning(['File ', filePath, ' does not have the required columns. Skipping...']);
                end
            end
        end
    end
end

%%
%Save data
% Save the main compiled data
if ~isempty(compiledData)
    % Convert complex values to strings for CSV export
    a = cell2mat(compiledData);
    columnNames = [cellstr(num2str(frequencyPoints./1e9, '%.10g')).', {'Label'}];
    compiledTable = array2table(a, 'VariableNames', columnNames);

    % Save the compiled data to a CSV file
    outputFileName = 'compiled_data.csv';
    writetable(compiledTable, outputFileName);
    disp(['Main data saved to ', outputFileName]);
else
    disp('No main data to save.');
end

% Save the baseline data
if ~isempty(baselineData)
    % Convert baseline complex values to strings for CSV export
    b = cell2mat(baselineData);
    % Create table with appropriate column names
    baselineColumnNames = [cellstr(num2str(frequencyPoints./1e9, '%.10g')).', {'Label'}];
    baselineTable = array2table(b, 'VariableNames', baselineColumnNames);

    % Save the baseline data to a CSV file
    baselineFileName = 'baseline_data.csv';
    writetable(baselineTable, baselineFileName);
    disp(['Baseline data saved to ', baselineFileName]);
else
    disp('No baseline data to save.');
end

% Save the baseline data
if ~isempty(sinteticData)
    % Convert baseline complex values to strings for CSV export
    cS = cell2mat(sinteticData);
    % Create table with appropriate column names
    sinteticColumnNames = [cellstr(num2str(frequencyPointsS./1e9, '%.10g')).', {'Label'}];
    sinteticTable = array2table(cS, 'VariableNames', sinteticColumnNames);

    % Save the baseline data to a CSV file
    sinteticFileName = 'sintetic_data.csv';
    writetable(sinteticTable, sinteticFileName);
    disp(['Sintetic data saved to ', baselineFileName]);
else
    disp('No baseline data to save.');
end

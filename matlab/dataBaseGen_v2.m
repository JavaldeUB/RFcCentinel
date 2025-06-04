% Define the base folder
baseFolder = 'C:\Users\jalon\Documents\curso24_25\Investigacion\UB\GRAF\RFcCentinel\python\dataBase_EthH2O_1p2mL_1p6to3GHz_random'; % Replace with your base folder path
outputFile = 'organized_data_with_baseline.csv'; % Output CSV file
warning('off','all')
warning
% Initialize an empty array for storing data
allData = [];

% Get the list of date-time folders
dateFolders = dir(fullfile(baseFolder, '2024*')); % Assuming folders start with "2024"
dateFolders = dateFolders([dateFolders.isdir]);

% Iterate through each date-time folder
for i = 1:length(dateFolders)
    dateFolderName = dateFolders(i).name;
    dateFolderPath = fullfile(baseFolder, dateFolderName);
    
    % Extract day and time from the folder name
    day = dateFolderName(1:8); % YYYYMMDD
    time = dateFolderName(10:end); % HHMM
    
    % Read and process the baseline measurement
    baselineFile = fullfile(dateFolderPath, 'baseline1p6to3GHz.csv');
    if isfile(baselineFile)
        baselineData = readtable(baselineFile);
        
        % Compute the 20*log10(RealPart + 1i*ImaginaryPart) for the baseline
        baselineMagnitudeLog = 20 * log10(abs(baselineData.RealPart + 1i * baselineData.ImaginaryPart))';
        
        % Append baseline data as the first row for the day
        baselineRow = [baselineMagnitudeLog, 0, 0, str2double(day), str2double(time)];
        allData = [allData; baselineRow]; %#ok<AGROW>
    end
    
    % Get subfolders (x10, x20, ..., x96)
    sampleFolders = dir(fullfile(dateFolderPath, 'x*'));
    sampleFolders = sampleFolders([sampleFolders.isdir]);
    
    for j = 1:length(sampleFolders)
        sampleFolderName = sampleFolders(j).name; % e.g., x10
        concentration = str2double(sampleFolderName(2:end)); % Extract concentration
        
        sampleFolderPath = fullfile(dateFolderPath, sampleFolderName);
        
        % Get CSV files in the current folder
        csvFiles = dir(fullfile(sampleFolderPath, '*.csv'));
        
        for k = 1:length(csvFiles)
            csvFileName = csvFiles(k).name;
            vialStr = regexp(csvFileName, 'v(\d+)', 'tokens', 'once');
            vial = str2double(vialStr{1}); % Extract vial number
            
            % Read the CSV file
            filePath = fullfile(sampleFolderPath, csvFileName);
            data = readtable(filePath);
            
            % Compute the 20*log10(RealPart + 1i*ImaginaryPart)
            magnitudeLog = 20 * log10(abs(data.RealPart + 1i * data.ImaginaryPart))';
            
            % Append the computed row with additional metadata
            rowData = [magnitudeLog, concentration, vial, str2double(day), str2double(time)];
            allData = [allData; rowData]; %#ok<AGROW>
        end
    end
end

% Convert the data into a table with appropriate column names
frequencyColumns = arrayfun(@(x) sprintf('Freq_%d', x), 1:201, 'UniformOutput', false);
variableNames = [frequencyColumns, {'Concentration', 'Vial', 'Day', 'Time'}];
dataTable = array2table(allData, 'VariableNames', variableNames);

% Write the final table to a CSV file
writetable(dataTable, outputFile);
disp(['Data has been successfully written to ', outputFile]);
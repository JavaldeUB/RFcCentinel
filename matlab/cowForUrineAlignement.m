% Define the base folder
baseFolder = 'C:\Users\javit\Documents\curso24_25\Investigacion\UB\GRAF\RFcCentinel\measurements\Urine'; % Replace with your base folder path
outputFile = 'cowSpectrum_baseline.csv'; % Output CSV file
warning('off','all')
warning
% Initialize an empty array for storing data
allData = [];

% Get the list of date-time folders
dateFolders = dir(fullfile(baseFolder, 'orinaViales*')); % Assuming folders start with "orinaViales"
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
            vialStr = regexp(csvFileName, 'r0_c00_s(\d+)', 'tokens', 'once');
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
allData_shorted = sortrows(allData,203);
% Convert the data into a table with appropriate column names
frequencyColumns = arrayfun(@(x) sprintf('Freq_%d', x), 1:201, 'UniformOutput', false);
variableNames = [frequencyColumns, {'Concentration', 'Vial', 'Day', 'Time'}];
dataTable = array2table(allData_shorted, 'VariableNames', variableNames);

%Asymetric Least Squares for baseline remmoval
X_preBSL = -allData_shorted(2:end,1:201);
z = zeros(size(X_preBSL));
for i=1:size(X_preBSL,1)
    z(i,:) = asymmetric_least_squares(X_preBSL(i,:),1e6,0.001,10);
end
X = X_preBSL-z;
figure;plot(X.');

%optim_cow for optimal parameters for cow alignment
optim_space = [5 30 2 20]; %segment lenght limits and slack limits
options = [0 3 50 .7];
ref = ref_select(X,[],[1 5]);
[optim_pars,OS,diagnos] = optim_cow(X,optim_space,options,ref);
diagnos

%cow with the parameters optimized
[Warping,XWarped,Diagnos] = cow(ref,X,optim_pars(1),optim_pars(2));

%Plot warped and raw
ax(1) = subplot(2,1,1);
plot(XWarped.');
title('Warped');
axis tight

ax(2) = subplot(2,1,2);
plot(X.');
title('Raw_data');
axis tight
shg

linkaxes(ax,'xy');
shg

%mean of the aligned data.
X_ref = mean(XWarped,1);
variableNames2 = frequencyColumns;
dataTable2 = array2table(-X_ref, 'VariableNames', variableNames2);
baselineFile2 = fullfile(dateFolderPath, outputFile);
writetable(dataTable2, baselineFile2);

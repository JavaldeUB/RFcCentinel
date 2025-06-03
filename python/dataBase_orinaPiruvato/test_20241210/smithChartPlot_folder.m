% MATLAB script to plot Smith Charts from multiple CSV files

% Open file dialog to select folder
folder = uigetdir(pwd, 'Select the Folder Containing CSV Files');
if folder == 0
    error('No folder selected. Program terminated.');
end

% Get list of CSV files starting with 'x'
csvFiles = dir(fullfile(folder, 'x*.csv'));

% Check if there are any matching files
if isempty(csvFiles)
    error('No files starting with "x" were found in the selected folder.');
end

% Initialize a figure for the Smith Chart
figure;
%smithplot('grid', 'on'); % Initialize Smith Chart with grid
hold on;

% Loop through each file and plot its data
for i = 1:length(csvFiles)
    % Construct full file path
    filePath = fullfile(folder, csvFiles(i).name);
    
    % Read the CSV file
    data = readmatrix(filePath);
    
    % Ensure the file has at least three columns
    if size(data, 2) < 3
        warning('File "%s" does not have three columns. Skipping.', csvFiles(i).name);
        continue;
    end
    
    % Extract columns
    freq = data(:, 1);           % Frequency (Hz)
    realPart = data(:, 2);       % Real part
    imagPart = data(:, 3);       % Imaginary part
    
    % Convert real and imaginary parts to a complex reflection coefficient
    reflectionCoeff = realPart + 1j * imagPart;
    
    % Plot the data on the Smith Chart
    smithplot(freq, reflectionCoeff); % Use filename as label
    hold on;
end

% Customize the plot
title('Smith Chart for Multiple CSV Files');
%legend('show'); % Show a legend to distinguish files
hold off;


function procesar_archivos_csv(folderPath)
    % List all CSV files in the folder
    files = dir(fullfile(folderPath, '*.csv'));
    baseFile = ''; 
    
    % Find the base file
    for i = 1:length(files)
        if startsWith(files(i).name, 'base')
            baseFile = files(i).name;
            break;
        end
    end
    
    if ~isempty(baseFile)
        disp(['Base file found: ', baseFile]);
        baseFilePath = fullfile(folderPath, baseFile);
        
        % Read the base file
        baseData = readtable(baseFilePath);
        if all(ismember({'Freq_Hz_', 'RealPart', 'ImaginaryPart'}, baseData.Properties.VariableNames))
            % Calculate base S21 spectrum
            s21Base = 20 * log10(abs(baseData.RealPart + j * baseData.ImaginaryPart));
            freqBase = baseData.Freq_Hz_;
        else
            disp(['The base file ', baseFile, ' does not contain the expected columns.']);
            return;
        end
    else
        disp('No base file found.');
        return;
    end
    
    % Process all CSV files
    disp(['Processing ', num2str(length(files) - 1), ' CSV files...']);
    
    s21List = [];
    freqList = [];
    minFreqs = [];
    s21MinValues = [];
    
    for i = 1:length(files)
        if strcmp(files(i).name, baseFile)
            continue; % Skip the base file
        end
        
        filePath = fullfile(folderPath, files(i).name);
        data = readtable(filePath);
        
        if all(ismember({'Freq_Hz_', 'RealPart', 'ImaginaryPart'}, data.Properties.VariableNames))
            s21 = 20 * log10(abs(data.RealPart + j * data.ImaginaryPart));
            
            % Interpolate the base spectrum to match current frequency
            interpBase = interp1(freqBase, s21Base, data.Freq_Hz_, 'linear', 'extrap');
            s21 = s21 - interpBase;  % Subtract base spectrum
            
            s21List = [s21List; s21.'];
            freqList = [freqList; data.Freq_Hz_];
            
            [~, idxMin] = min(s21);
            minFreqs = [minFreqs; data.Freq_Hz_(idxMin)];
            s21MinValues = [s21MinValues; s21(idxMin)];
        else
            disp(['Warning: ', files(i).name, ' does not contain the expected columns.']);
        end
    end
    
    % Step 1: Calculate the average minimum frequency
    avgMinFreq = mean(minFreqs);
    disp(['Average minimum frequency: ', num2str(avgMinFreq), ' Hz']);
    
    % Step 2: Calculate the average of the minimum S21 values
    avgS21Min = mean(s21MinValues);
    disp(['Average minimum S21 value: ', num2str(avgS21Min), ' dB']);
    
    % Step 3: Align spectra around the average minimum frequency and generate synthetic spectrum
    [alignedFreqs, syntheticS21] = align_and_generate_synthetic_spectrum(freqList, s21List, minFreqs, avgMinFreq);
    
    % Step 4: Save the synthetic spectrum to a CSV
    %save_synthetic_csv(alignedFreqs, syntheticS21, folderPath);
    
    % Generate the plot
    generate_plot(freqList, s21List, alignedFreqs, syntheticS21);
end

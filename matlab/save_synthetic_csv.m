
function save_synthetic_csv(freqs, syntheticS21, folderPath)
    % Save the synthetic spectrum to a CSV file
    syntheticData = table(freqs', syntheticS21', 'VariableNames', {'Freq (Hz)', 's21avg'});
    outputFilePath = fullfile(folderPath, 'synthetic_spectrum.csv');
    writetable(syntheticData, outputFilePath);
    disp(['Synthetic CSV file generated: ', outputFilePath]);
end


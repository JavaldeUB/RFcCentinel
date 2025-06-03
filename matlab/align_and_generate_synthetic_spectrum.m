
function [alignedFreqs, syntheticS21] = align_and_generate_synthetic_spectrum(freqList, s21List, minFreqs, avgMinFreq)
    % Align the spectra so that their minima coincide with the average minimum frequency
    alignedS21 = [];
    alignedFreqs = [];
    
    % Define a common frequency range for interpolation
    freqMinCommon = max(min(freqList));
    freqMaxCommon = min(max(freqList));
    if length(freqList)<20000
        freqList = reshape(freqList.',[201,98]).';
    elseif length(freqList)>20100
        freqList = reshape(freqList.',[201,101]).';
    else
        freqList = reshape(freqList.',[201,100]).';
    end
    commonFreqs = linspace(freqMinCommon, freqMaxCommon, 1000); % 1000 common points
    [a,~] = size(freqList);
    for i = 1:a
        % Calculate the shift required to align the spectrum's minimum with the average minimum frequency
        shift = avgMinFreq - minFreqs(i);
        
        % Shift the frequencies to align the minimum
        shiftedFreqs = freqList(i,:) + shift;
        
        % Interpolate the shifted spectrum
        interpS21 = interp1(shiftedFreqs, s21List(i, :), commonFreqs, 'linear', 'extrap');
        interpS21old(i,:) =interp1(freqList(i,:), s21List(i, :), commonFreqs, 'linear', 'extrap'); 
        % Store the aligned spectrum
        alignedS21 = [alignedS21; interpS21];
    end
    
    % Average the aligned spectra to generate the synthetic spectrum
    syntheticS21 = mean(alignedS21, 1);
    diff = interpS21old-syntheticS21;
    squareDiff = diff.^2;
    stdDeviation = std(squareDiff, 0, 1);
    meanStdDeviation = mean(stdDeviation);
    disp(['Mean Standard Deviation: ', num2str(meanStdDeviation)]);
    alignedFreqs = commonFreqs;
end

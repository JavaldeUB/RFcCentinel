function generate_plot(freqList, s21List, alignedFreqs, syntheticS21)
    % Generate a plot of the spectra
    figure;
    hold on;
    
    for i = 1:size(s21List, 1)
        plot(freqList(i,:)./1e9, s21List(i, :), 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
    end
    
    plot(alignedFreqs/1e9, syntheticS21, 'r', 'LineWidth', 2.5);
    
    title('All Spectra and Synthetic Spectrum');
    xlabel('Frequency (GHz)');
    ylabel('20*log10(|S21|) (dB)');
    grid on;
    hold off;
end

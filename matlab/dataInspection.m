colors = jet(length(uniqueNumbers));
for j=1:9
    figure;
    for i=(j-1)*98+1:j*98
        hold on;
        [val, idx]=min(20.*log10(abs(X_bc(i,:))));
        plot(freqAxis,20.*log10(abs(X_bc(i,:))),'Color', colors(j, :))
        text(freqAxis(idx), val, num2str(i), ...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'middle');
    end
    plot(fAxisSintetic,sintetics(j,:),'Color','k','LineWidth',2)
end

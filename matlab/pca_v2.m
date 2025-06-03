%% Set up the Import Options and import the data
clear all;
opts = delimitedTextImportOptions("NumVariables", 202);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["etiqueta", "FreqHz_1", "FreqHz_2", "FreqHz_3", "FreqHz_4", "FreqHz_5", "FreqHz_6", "FreqHz_7", "FreqHz_8", "FreqHz_9", "FreqHz_10", "FreqHz_11", "FreqHz_12", "FreqHz_13", "FreqHz_14", "FreqHz_15", "FreqHz_16", "FreqHz_17", "FreqHz_18", "FreqHz_19", "FreqHz_20", "FreqHz_21", "FreqHz_22", "FreqHz_23", "FreqHz_24", "FreqHz_25", "FreqHz_26", "FreqHz_27", "FreqHz_28", "FreqHz_29", "FreqHz_30", "FreqHz_31", "FreqHz_32", "FreqHz_33", "FreqHz_34", "FreqHz_35", "FreqHz_36", "FreqHz_37", "FreqHz_38", "FreqHz_39", "FreqHz_40", "FreqHz_41", "FreqHz_42", "FreqHz_43", "FreqHz_44", "FreqHz_45", "FreqHz_46", "FreqHz_47", "FreqHz_48", "FreqHz_49", "FreqHz_50", "FreqHz_51", "FreqHz_52", "FreqHz_53", "FreqHz_54", "FreqHz_55", "FreqHz_56", "FreqHz_57", "FreqHz_58", "FreqHz_59", "FreqHz_60", "FreqHz_61", "FreqHz_62", "FreqHz_63", "FreqHz_64", "FreqHz_65", "FreqHz_66", "FreqHz_67", "FreqHz_68", "FreqHz_69", "FreqHz_70", "FreqHz_71", "FreqHz_72", "FreqHz_73", "FreqHz_74", "FreqHz_75", "FreqHz_76", "FreqHz_77", "FreqHz_78", "FreqHz_79", "FreqHz_80", "FreqHz_81", "FreqHz_82", "FreqHz_83", "FreqHz_84", "FreqHz_85", "FreqHz_86", "FreqHz_87", "FreqHz_88", "FreqHz_89", "FreqHz_90", "FreqHz_91", "FreqHz_92", "FreqHz_93", "FreqHz_94", "FreqHz_95", "FreqHz_96", "FreqHz_97", "FreqHz_98", "FreqHz_99", "FreqHz_100", "FreqHz_101", "FreqHz_102", "FreqHz_103", "FreqHz_104", "FreqHz_105", "FreqHz_106", "FreqHz_107", "FreqHz_108", "FreqHz_109", "FreqHz_110", "FreqHz_111", "FreqHz_112", "FreqHz_113", "FreqHz_114", "FreqHz_115", "FreqHz_116", "FreqHz_117", "FreqHz_118", "FreqHz_119", "FreqHz_120", "FreqHz_121", "FreqHz_122", "FreqHz_123", "FreqHz_124", "FreqHz_125", "FreqHz_126", "FreqHz_127", "FreqHz_128", "FreqHz_129", "FreqHz_130", "FreqHz_131", "FreqHz_132", "FreqHz_133", "FreqHz_134", "FreqHz_135", "FreqHz_136", "FreqHz_137", "FreqHz_138", "FreqHz_139", "FreqHz_140", "FreqHz_141", "FreqHz_142", "FreqHz_143", "FreqHz_144", "FreqHz_145", "FreqHz_146", "FreqHz_147", "FreqHz_148", "FreqHz_149", "FreqHz_150", "FreqHz_151", "FreqHz_152", "FreqHz_153", "FreqHz_154", "FreqHz_155", "FreqHz_156", "FreqHz_157", "FreqHz_158", "FreqHz_159", "FreqHz_160", "FreqHz_161", "FreqHz_162", "FreqHz_163", "FreqHz_164", "FreqHz_165", "FreqHz_166", "FreqHz_167", "FreqHz_168", "FreqHz_169", "FreqHz_170", "FreqHz_171", "FreqHz_172", "FreqHz_173", "FreqHz_174", "FreqHz_175", "FreqHz_176", "FreqHz_177", "FreqHz_178", "FreqHz_179", "FreqHz_180", "FreqHz_181", "FreqHz_182", "FreqHz_183", "FreqHz_184", "FreqHz_185", "FreqHz_186", "FreqHz_187", "FreqHz_188", "FreqHz_189", "FreqHz_190", "FreqHz_191", "FreqHz_192", "FreqHz_193", "FreqHz_194", "FreqHz_195", "FreqHz_196", "FreqHz_197", "FreqHz_198", "FreqHz_199", "FreqHz_200", "FreqHz_201"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
s21allEthanolmag = readtable("C:\Users\javit\Documents\curso24_25\Docencia\EHU\EMIMEP\SSS\Lab\python\dataBase_Ethanol_1p2mL_1p6to3GHz\s21_allEthanol_mag.csv", opts);


%Clear temporary variables
clear opts
%%
%Set up variables and clean-up
freqAxis = s21allEthanolmag{1,2:end}; 
freqAxis = freqAxis(51:185);
yr = s21allEthanolmag{2:end,1};
Xr = s21allEthanolmag{2:end,2:end};
Xr = Xr(:,51:185);
baseline = Xr(709,:);
Xr = Xr./baseline;
rowsToRemove = [1,2,101,202,303,405,506,607,708,709];
rowsToKeep = setdiff(1:size(Xr, 1), rowsToRemove);
Xclean = Xr(rowsToKeep, :);
yclean = yr(rowsToKeep);

%%
%adjust number of samples per tag
uniqueNumbers = unique(yclean);
numUnique = numel(uniqueNumbers);
counts = histcounts(yclean, uniqueNumbers);
Xp = [];
y = [];
minCount = min(counts);
for i = 1:numel(uniqueNumbers)
    currentIndices = find(yclean == uniqueNumbers(i));    
    selectedIndices = currentIndices(1:minCount);
    Xp = [Xp; Xclean(selectedIndices, :)];
    y = [y; yclean(selectedIndices)];
end
colors = jet(length(uniqueNumbers));
legend_handles = gobjects(length(uniqueNumbers), 1);
figure;
hold on;
for i = 1:length(uniqueNumbers)
    label = uniqueNumbers(i);
    rows = Xp(y == label, :);
    for j = 1:size(rows, 1)
        h = plot(freqAxis./1e9,20.*log10(rows(j, :)), 'Color', colors(i, :));        
        if j == 1
            legend_handles(i) = h;
        end
    end
end
hold off;
xlabel("Freq (GHz)");
ylabel("20log_{10}|S_{21}|");
legend(legend_handles, arrayfun(@num2str, uniqueNumbers, 'UniformOutput', false), 'Location', 'best');
%%
% Autoscaling
X_mean = mean(Xp);
X_std = std(Xp);
X = (Xp - X_mean)./ X_std; %Con espectros al ppio no se estandariza con 
% la desviación por que amplificas  el ruido. Después cuando haces feature 
%selection si para comparar varianzas entre dos cosas que son de amplitud 
%muy distinta
%X = Xp;
%%
%PCA and plots
[coeff, score, latent, ~, explained] = pca(X);
numComponents = 20; % Limit to 20 components
coeff = coeff(:, 1:numComponents);
score = score(:, 1:numComponents);
explained = explained(1:numComponents);
%%
% Plot explained variance
figure;
bar(explained, 'FaceAlpha', 0.7);
title('Explained Variance Distribution by Principal Components');
xlabel('Principal Component');
ylabel('Explained Variance (%)');
grid on;
figure;
plot(cumsum(explained), '-o', 'LineWidth', 1.5, 'DisplayName', 'Cumulative Explained Variance');
xlabel('Principal Component');
ylabel('Explained Variance (%)');
title('Explained Variance by Principal Components');
legend;
grid on;
%%
% 2D Scatter plot: PC2 vs. PC1
figure;
scatter(score(:, 1), score(:, 2), 50, y, 'filled');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PC2 vs. PC1');
colormap('jet');
colorbar;
grid on;

% Annotate the first 5 and last 5 samples relative to each concentration
uniqueLabels = unique(y);
for label = uniqueLabels'
    % Get indices for this concentration
    idx = find(y == label);
    numSamples = length(idx);
    
    % Select first 5 and last 5 samples
    relativeNumbers = [1:5, numSamples-4:numSamples];
    selectedIdx = [idx(1:5); idx(end-4:end)];
    
    % Annotate selected samples with relative numbers
    for i = 1:length(selectedIdx)
        sampleIdx = selectedIdx(i);
        relativeNum = relativeNumbers(i); % Relative numbering
        text(score(sampleIdx, 1), score(sampleIdx, 2), ...
            num2str(relativeNum), 'FontSize', 8, 'HorizontalAlignment', 'left');
    end
end
hold off;
%%
% 2D Scatter plot: PC3 vs. PC1
figure;
scatter(score(:, 1), score(:, 3), 50, y, 'filled');
xlabel('Principal Component 1');
ylabel('Principal Component 3');
title('PC3 vs. PC1');
colormap('jet');
colorbar;
grid on;
% Annotate the first 5 and last 5 samples relative to each concentration
for label = uniqueLabels'
    % Get indices for this concentration
    idx = find(y == label);
    numSamples = length(idx);
    
    % Select first 5 and last 5 samples
    relativeNumbers = [1:5, numSamples-4:numSamples];
    selectedIdx = [idx(1:5); idx(end-4:end)];
    
    % Annotate selected samples with relative numbers
    for i = 1:length(selectedIdx)
        sampleIdx = selectedIdx(i);
        relativeNum = relativeNumbers(i); % Relative numbering
        text(score(sampleIdx, 1), score(sampleIdx, 3), ...
            num2str(relativeNum), 'FontSize', 16, 'HorizontalAlignment', 'left');
    end
end
hold off;
%%
% 2D Scatter plot: PC3 vs. PC2
figure;
scatter(score(:, 2), score(:, 3), 50, y, 'filled');
xlabel('Principal Component 2');
ylabel('Principal Component 3');
title('PC3 vs. PC2');
colormap('jet');
colorbar;
grid on;
%%
% 3D Scatter plot: PC1, PC2, and PC3
figure;
scatter3(score(:, 1), score(:, 2), score(:, 3), 50, y, 'filled');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
title('3D Plot of PC1, PC2, and PC3');
colormap('jet');
colorbar;
grid on;
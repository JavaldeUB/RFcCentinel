%% Set up the Import Options and import the data
clear all
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
s21allEthanolph = readtable("C:\Users\javit\Documents\curso24_25\Docencia\EHU\EMIMEP\SSS\Lab\python\dataBase_Ethanol_1p2mL_1p6to3GHz\s21_allEthanol_ph.csv", opts);


%Clear temporary variables
clear opts
%%
%Set up variables and clean-up
freqAxis = s21allEthanolmag{1,2:end}; 
freqAxis = freqAxis(51:185);
yr = s21allEthanolmag{2:end,1};
Xr = s21allEthanolmag{2:end,2:end}.*exp(i.*s21allEthanolph{2:end,2:end});
Xr = Xr(:,51:185);
baseline = Xr(709,:);
Xr = Xr./baseline;
rowsToRemove = [1,2,101,202,303,405,506,607,708,709];
rowsToKeep = setdiff(1:size(Xr, 1), rowsToRemove);
Xc = Xr(rowsToKeep, :);
yc = yr(rowsToKeep);
[rowsInterp, ~]=find(yc==[80,60,40]);
rowsLeft = setdiff(1:size(Xc, 1), rowsInterp);
Xinterp = Xc(rowsInterp,:);
yinterp = yc(rowsInterp);
Xclean = Xc(rowsLeft,:);
yclean = yc(rowsLeft);
%%
%adjust number of samples per tag
uniqueNumbers = unique(yclean);
numUnique = numel(uniqueNumbers);
counts = histcounts(yclean, uniqueNumbers);
Xp = [];
y = [];
minCount = min(counts)-30;
for i = 1:numel(uniqueNumbers)
    currentIndices = find(yclean == uniqueNumbers(i));    
    selectedIndices = currentIndices(1:minCount);
    Xp = [Xp; Xclean(selectedIndices, :)];
    y = [y; yclean(selectedIndices)];
end
colors = lines(length(uniqueNumbers));
legend_handles = gobjects(length(uniqueNumbers), 1);
figure;
hold on;
for i = 1:length(uniqueNumbers)
    label = uniqueNumbers(i);
    rows = Xp(y == label, :);
    for j = 1:size(rows, 1)
        h = plot(freqAxis./1e9,20.*log10(abs(rows(j, :))), 'Color', colors(i, :));        
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
%Autoscale
X_mean = mean(Xp);
X_std = std(Xp);
%X = Xp;
X = (Xp - X_mean)./ X_std; %Con espectros al ppio no se estandariza con 
% la desviación por que amplificas  el ruido. Después cuando haces feature 
%selection si para comparar varianzas entre dos cosas que son de amplitud 
%muy distinta
% figure;plot(freqAxis./1e9,Xp);
% xlabel("Freq (GHz)");
% ylabel("Standarized data");
%%
groups = repelem(1:length(uniqueNumbers), minCount)';  
% Split Data into Training and Testing
cv = cvpartition(groups, 'Holdout', 0.2);
X_train = Xclean(training(cv), :);
Y_train = yclean(training(cv), :);


groups_train = groups(training(cv));
X_test = Xclean(test(cv), :);
Y_test = yclean(test(cv), :);
%%
% LOGO Cross-Validation
uniqueGroups = unique(groups_train);
maxComponents = 41; 
mseCV = zeros(maxComponents, 1);


for nComp = 1:maxComponents
    mseGroup = zeros(length(uniqueGroups), 1);
    
    for i = 1:length(uniqueGroups)
        
        testIdx = (groups_train == uniqueGroups(i));
        trainIdx = ~testIdx;
        X_t2 = X_train(trainIdx, :);
        Y_t2 = Y_train(trainIdx, :);
        row_indices = randperm(size(X_t2, 1));
        X_train_logo = X_t2(row_indices,:);
        Y_train_logo = Y_t2(row_indices);
        X_test_logo = X_train(testIdx, :);
        Y_test_logo = Y_train(testIdx, :);
        
        [XL, YL, XS, YS, beta, PLSstats] = plsregress(X_train_logo, Y_train_logo, nComp);
        Y_pred_logo = [ones(size(X_test_logo, 1), 1), X_test_logo] * beta;
        mseGroup(i) = sqrt(mean((Y_test_logo - Y_pred_logo).^2));
    end
    
    % Average MSE Across Groups
    mseCV(nComp) = mean(mseGroup);
end
% Plot MSE Across Components
figure;
plot(1:maxComponents, mseCV, '-o');
xlabel('Number of Components');
ylabel('Cross-Validated RMSE');
title('LOGO Cross-Validation with Polynomial Features');
grid on;
%%
%Prediction
[~, optimalComponents] = min(mseCV);
disp(['Optimal Number of Components: ', num2str(optimalComponents)]);
row_indices = randperm(size(X_train, 1));
[XLF, YLF, XSF, YSF, betaFinal, PLSstatsF] = plsregress(X_train, Y_train, optimalComponents);
Y_pred_test = [ones(size(X_test, 1), 1), X_test] * betaFinal;
mseTest = sqrt(mean((Y_test - Y_pred_test).^2));
RSS = sum((Y_test - Y_pred_test).^2);
TSS = sum((Y_test - mean(Y_test)).^2);
r2 = 1 - (RSS / TSS);
disp(['Test RMSEP: ', num2str(mseTest)]);
disp(['Test R^{2}: ', num2str(r2)]);


%%
%Bland-Altman+model
% Predicted vs. Actual Test Values
figure;
scatter(Y_test, Y_pred_test, 'b', 'filled'); % Scatter plot of predicted vs actual
hold on;

% Add Perfect Agreement Line (Diagonal)
minVal = min([Y_test; Y_pred_test]);
maxVal = max([Y_test; Y_pred_test]);
plot([minVal, maxVal], [minVal, maxVal], 'k', 'LineWidth', 1.5); % Perfect agreement line

% Calculate Bias and Limits of Agreement
bias = mean(Y_pred_test - Y_test); % Mean difference
LoA = 1.96 * std(Y_pred_test - Y_test); % Limits of agreement (±1.96 * SD)

% Add Bland-Altman Lines Parallel to the Diagonal
biasLine = plot([minVal, maxVal], [minVal + bias, maxVal + bias], 'r', 'LineWidth', 1.5); % Bias line
upperLoALine = plot([minVal, maxVal], [minVal + bias + LoA, maxVal + bias + LoA], 'k-', 'LineWidth', 1.5); % Upper LoA
lowerLoALine = plot([minVal, maxVal], [minVal + bias - LoA, maxVal + bias - LoA], 'k-', 'LineWidth', 1.5); % Lower LoA

% Add Labels for LoA Lines
upperLoA = bias + LoA; % Upper LoA value
lowerLoA = bias - LoA; % Lower LoA value

% Calculate slopes of LoA lines
dx = maxVal - minVal; % Horizontal difference
dy_upper = (maxVal + upperLoA) - (minVal + upperLoA); % Vertical difference for upper LoA
dy_lower = (maxVal + lowerLoA) - (minVal + lowerLoA); % Vertical difference for lower LoA

% Angle for text rotation
angle_upper = atand(dy_upper / dx)-11; % Angle of upper LoA in degrees
angle_lower = atand(dy_lower / dx)-11; % Angle of lower LoA in degrees

% Place text parallel to Upper LoA Line
textX_upper = minVal + 0.65 * dx; % Position slightly along the line
textY_upper = (minVal + upperLoA) + 0.73 * dy_upper; % Position slightly along the line
text(textX_upper, textY_upper, sprintf('LoA = %.2f (+1.96SD)', upperLoA), ...
    'Color', 'k', 'FontSize', 10, 'Rotation', angle_upper, 'HorizontalAlignment', 'center');

% Place text parallel to Lower LoA Line
textX_lower = minVal + 0.75 * dx; % Position slightly along the line
textY_lower = (minVal + lowerLoA) + 0.67 * dy_lower; % Position slightly along the line
text(textX_lower, textY_lower, sprintf('LoA = %.2f (-1.96SD)', lowerLoA), ...
    'Color', 'k', 'FontSize', 10, 'Rotation', angle_lower, 'HorizontalAlignment', 'center');

% Customize Plot
xlabel('Actual Y (Test)');
ylabel('Predicted Y');
title('Predicted vs. Actual');
legend('Predicted vs Actual', 'Perfect Agreement', 'Bias', 'Limits of Agreement', 'Location', 'Best');
grid on;
hold off;
%%
%Calculation of the interpolation
Y_pred_interp = [ones(size(Xinterp, 1), 1), Xinterp] * betaFinal;
mseTestInterp = sqrt(mean((yinterp - Y_pred_interp).^2));
RSSinterp = sum((Y_test - Y_pred_test).^2);
TSSinterp = sum((Y_test - mean(Y_test)).^2);
r2interp = 1 - (RSSinterp / TSSinterp);
disp(['Interpolation Test RMSEP: ', num2str(mseTestInterp)]);
disp(['Interpolation Test R^{2}: ', num2str(r2)]);

%%
%Bland-Altman+model
% Predicted vs. Actual Test Values
figure;
scatter(yinterp, Y_pred_interp, 'b', 'filled'); % Scatter plot of predicted vs actual
hold on;

% Add Perfect Agreement Line (Diagonal)
minVal = min([yinterp; Y_pred_interp]);
maxVal = max([yinterp; Y_pred_interp]);
plot([minVal, maxVal], [minVal, maxVal], 'k-.', 'LineWidth', 1.5); % Perfect agreement line

% Calculate Bias and Limits of Agreement
bias = mean(Y_pred_interp - yinterp); % Mean difference
LoA = 1.96 * std(Y_pred_interp - yinterp); % Limits of agreement (±1.96 * SD)

% Add Bland-Altman Lines Parallel to the Diagonal
biasLine = plot([minVal, maxVal], [minVal + bias, maxVal + bias], 'r', 'LineWidth', 1.5); % Bias line
upperLoALine = plot([minVal, maxVal], [minVal + bias + LoA, maxVal + bias + LoA], 'k-', 'LineWidth', 1.5); % Upper LoA
lowerLoALine = plot([minVal, maxVal], [minVal + bias - LoA, maxVal + bias - LoA], 'k-', 'LineWidth', 1.5); % Lower LoA

% Add Labels for LoA Lines
upperLoA = bias + LoA; % Upper LoA value
lowerLoA = bias - LoA; % Lower LoA value

% Calculate slopes of LoA lines
dx = maxVal - minVal; % Horizontal difference
dy_upper = (maxVal + upperLoA) - (minVal + upperLoA); % Vertical difference for upper LoA
dy_lower = (maxVal + lowerLoA) - (minVal + lowerLoA); % Vertical difference for lower LoA

% Angle for text rotation
angle_upper = atand(dy_upper / dx)-15; % Angle of upper LoA in degrees
angle_lower = atand(dy_lower / dx)-15; % Angle of lower LoA in degrees

% Place text parallel to Upper LoA Line
textX_upper = minVal + 0.65 * dx; % Position slightly along the line
textY_upper = (minVal + upperLoA) + 0.73 * dy_upper; % Position slightly along the line
text(textX_upper, textY_upper, sprintf('LoA = %.2f (+1.96SD)', upperLoA), ...
    'Color', 'k', 'FontSize', 10, 'Rotation', angle_upper, 'HorizontalAlignment', 'center');

% Place text parallel to Lower LoA Line
textX_lower = minVal + 0.75 * dx; % Position slightly along the line
textY_lower = (minVal + lowerLoA) + 0.67 * dy_lower; % Position slightly along the line
text(textX_lower, textY_lower, sprintf('LoA = %.2f (-1.96SD)', lowerLoA), ...
    'Color', 'k', 'FontSize', 10, 'Rotation', angle_lower, 'HorizontalAlignment', 'center');

% Customize Plot
xlabel('Actual Y (Test)');
ylabel('Predicted Y');
title('Interpolation Predicted vs. Actual');
legend('Predicted vs Actual', 'Perfect Agreement', 'Bias', 'Limits of Agreement', 'Location', 'Best');
grid on;
hold off;




%% Set up the plotting 
clear all
plt = input("Do you wish to plot the results (0->No|1->Yes): ");

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 205);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Freq_1", "Freq_2", "Freq_3", "Freq_4", "Freq_5", "Freq_6", "Freq_7", "Freq_8", "Freq_9", "Freq_10", "Freq_11", "Freq_12", "Freq_13", "Freq_14", "Freq_15", "Freq_16", "Freq_17", "Freq_18", "Freq_19", "Freq_20", "Freq_21", "Freq_22", "Freq_23", "Freq_24", "Freq_25", "Freq_26", "Freq_27", "Freq_28", "Freq_29", "Freq_30", "Freq_31", "Freq_32", "Freq_33", "Freq_34", "Freq_35", "Freq_36", "Freq_37", "Freq_38", "Freq_39", "Freq_40", "Freq_41", "Freq_42", "Freq_43", "Freq_44", "Freq_45", "Freq_46", "Freq_47", "Freq_48", "Freq_49", "Freq_50", "Freq_51", "Freq_52", "Freq_53", "Freq_54", "Freq_55", "Freq_56", "Freq_57", "Freq_58", "Freq_59", "Freq_60", "Freq_61", "Freq_62", "Freq_63", "Freq_64", "Freq_65", "Freq_66", "Freq_67", "Freq_68", "Freq_69", "Freq_70", "Freq_71", "Freq_72", "Freq_73", "Freq_74", "Freq_75", "Freq_76", "Freq_77", "Freq_78", "Freq_79", "Freq_80", "Freq_81", "Freq_82", "Freq_83", "Freq_84", "Freq_85", "Freq_86", "Freq_87", "Freq_88", "Freq_89", "Freq_90", "Freq_91", "Freq_92", "Freq_93", "Freq_94", "Freq_95", "Freq_96", "Freq_97", "Freq_98", "Freq_99", "Freq_100", "Freq_101", "Freq_102", "Freq_103", "Freq_104", "Freq_105", "Freq_106", "Freq_107", "Freq_108", "Freq_109", "Freq_110", "Freq_111", "Freq_112", "Freq_113", "Freq_114", "Freq_115", "Freq_116", "Freq_117", "Freq_118", "Freq_119", "Freq_120", "Freq_121", "Freq_122", "Freq_123", "Freq_124", "Freq_125", "Freq_126", "Freq_127", "Freq_128", "Freq_129", "Freq_130", "Freq_131", "Freq_132", "Freq_133", "Freq_134", "Freq_135", "Freq_136", "Freq_137", "Freq_138", "Freq_139", "Freq_140", "Freq_141", "Freq_142", "Freq_143", "Freq_144", "Freq_145", "Freq_146", "Freq_147", "Freq_148", "Freq_149", "Freq_150", "Freq_151", "Freq_152", "Freq_153", "Freq_154", "Freq_155", "Freq_156", "Freq_157", "Freq_158", "Freq_159", "Freq_160", "Freq_161", "Freq_162", "Freq_163", "Freq_164", "Freq_165", "Freq_166", "Freq_167", "Freq_168", "Freq_169", "Freq_170", "Freq_171", "Freq_172", "Freq_173", "Freq_174", "Freq_175", "Freq_176", "Freq_177", "Freq_178", "Freq_179", "Freq_180", "Freq_181", "Freq_182", "Freq_183", "Freq_184", "Freq_185", "Freq_186", "Freq_187", "Freq_188", "Freq_189", "Freq_190", "Freq_191", "Freq_192", "Freq_193", "Freq_194", "Freq_195", "Freq_196", "Freq_197", "Freq_198", "Freq_199", "Freq_200", "Freq_201", "Concentration", "Vial", "Day", "Time"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
organizeddatawithbaseline = readtable("C:\Users\javit\Documents\curso24_25\Docencia\EHU\EMIMEP\SSS\Lab\matlab\organized_data_with_baseline.csv", opts);

% Convert to output type
dataBase = table2array(organizeddatawithbaseline);

% Clear temporary variables
clear organizeddatawithbaseline
clear opts

%% Import the randomList
opts = delimitedTextImportOptions("NumVariables", 2);
% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = " ";
% Specify column names and types
opts.VariableNames = ["VarName1", "Var2"];
opts.SelectedVariableNames = "VarName1";
opts.VariableTypes = ["double", "char"];
% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";
% Specify variable properties
opts = setvaropts(opts, "Var2", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var2", "EmptyFieldRule", "auto");
opts = setvaropts(opts, "VarName1", "TrimNonNumeric", true);
opts = setvaropts(opts, "VarName1", "ThousandsSeparator", ",");
% Import the data
randomList = readtable("C:\Users\javit\Documents\curso24_25\Docencia\EHU\EMIMEP\SSS\Lab\python\dataBase_EthH2O_1p2mL_1p6to3GHz_random\randomList.txt", opts);
% Convert to output type
measList = table2array(randomList);
% Clear temporary variables
clear randomList
clear opts

%% Getting data in different variables
%data
sortData = sortrows(dataBase,202);          %Ordered by label database
baseline = sortData(1:5,1:201);             %baseline data
data = sortData(6:end,1:201);               %measured data
y = sortData(6:end,202);                    %labels
freq = linspace(1.6,3,201);             %Frequency Axis
freqAxis = freq(1:201);
%% Identifying labels and generating axes for processing
% Getting the numbers of the labels
uniqueNumbers = unique(y);
numUnique = numel(uniqueNumbers);
% Counting the number of samples of each label
counts = nonzeros(histcounts(y.', 0:0.1:uniqueNumbers(end)));
uniqueNumbers = uniqueNumbers(end:-1:1);
uniqueNumbers =uniqueNumbers(1:end);
% Getting the minimum of samples to be consider of each label.
% Not all the labels were sampled the same amount of times.
minCount = min(counts);

%% Reducing the number of samples for each label to get the same for all
Xp = data;
yp = y;

%% Plotting the 20.*log10 (|s21|)
if plt==1
    colors = jet(length(uniqueNumbers));
    colors = colors(end:-1:1,:);
    legend_handles = gobjects(length(uniqueNumbers), 1);
    hS21 = figure;
    hold on;
    for i = 1:length(uniqueNumbers)
        label = uniqueNumbers(i);
        rows = Xp(yp == label, :);
        rowsb = baseline(1,:);
        for j = 1:size(rows, 1)
            %[val, idx]=min(20.*log10(abs(rows(j,:)./rowsb)));
            h = plot(freqAxis,abs(rows(j, :)-rowsb), 'Color', colors(i, :));        
            %text(freqAxis(idx), val, num2str(j), ...
            %    'HorizontalAlignment', 'left', ...
            %    'VerticalAlignment', 'middle');
            if j == 1
                legend_handles(i) = h;
            end
        end
    end
    hold off;
    xlabel("Freq (GHz)");
    ylabel("20log_{10}|S_{21}|");
    legend(legend_handles, arrayfun(@num2str, uniqueNumbers, 'UniformOutput', false), 'Location', 'best');
end

%% Transformations and preprocessing
% Baseline Correction
X_bc = zeros(size(Xp));
for i = 1:length(uniqueNumbers)
    label = uniqueNumbers(i);
    rows = Xp(yp == label, :);
    rowsb = baseline(1,:);
    for j = 1:size(rows, 1)
        X_bc((i-1)*size(rows,1)+j,:) = abs(rows(j,:)-rowsb);
    end
end
% Autoscaling only magnitude
%X_bc_wo0 = X_bc(1:end-98,:);
X_mean = mean(X_bc);
X_std = std(X_bc);
X_autoS = (X_bc);%./X_std;

%% Plotting the autoscales
if plt==1
    colors = jet(length(uniqueNumbers));
    colors = colors(end:-1:1,:);
    legend_handles = gobjects(length(uniqueNumbers), 1);
    figure;
    hold on;
    for i = 1:length(uniqueNumbers)
        label = uniqueNumbers(i);
        rows = X_autoS(yp == label, :);
        rowsb = baseline(1,:);
        for j = 1:size(rows, 1)
            %[val, idx]=min(20.*log10(abs(rows(j,:)./rowsb)));
            h = plot(freqAxis,abs(rows(j, :)), 'Color', colors(i, :));        
            %text(freqAxis(idx), val, num2str(j), ...
            %    'HorizontalAlignment', 'left', ...
            %    'VerticalAlignment', 'middle');
            if j == 1
                legend_handles(i) = h;
            end
        end
    end
    hold off;
    xlabel("Freq (GHz)");
    ylabel("Autoscaled Data");
    legend(legend_handles, arrayfun(@num2str, uniqueNumbers, 'UniformOutput', false), 'Location', 'best');
end

%% Extracting the most different spectra from the sintetic spectrum
X_aux1 = X_autoS;
X_aux2 = X_bc;
y_aux = yp;

%% Partitioning the dataset for training, test and so on
%Extracting data for external validation
[rowsEV, ~]=find(yp==[80,60,40]);
rowsLeft = setdiff(1:size(X_aux1, 1), rowsEV);
X_Test = X_aux1(rowsEV,:);
y_Test = y_aux(rowsEV);
X_Train = X_aux1(rowsLeft,:);
y_Train = yp(rowsLeft);

%Dividing training dataSet
tUniqueNumbers = uniqueNumbers(~ismember(uniqueNumbers,[80,60,40]));
groups = repelem(1:length(tUniqueNumbers), minCount)';  
% Split Data into training and internal validation
cv = cvpartition(groups, 'Holdout', 0.2);
X_train = X_Train(training(cv), :);
Y_train = y_Train(training(cv), :);
groups_train = groups(training(cv));
X_test = X_Train(test(cv), :);
Y_test = y_Train(test(cv), :);;

%%
% LOGO Cross-Validation
uniqueGroups = unique(groups_train);
maxComponents = 41; 
mseCV = zeros(maxComponents, 1);
k = input("Input the degree of the polynomial regression: ");
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
        options.display='off';
        [XL, YL, XW, XS, YS, beta, PLSstats] = polypls(X_train_logo, Y_train_logo, nComp, k, options);
        
        % Calculate normalized PLS weights
        W0 = bsxfun(@rdivide,XW,sqrt(sum(XW.^2,1)));
        % Calculate the product of summed squares of XS and YL
        sumSq = sum(XS.^2,1).*sum(YL.^2,1);
        % Calculate VIP scores for NCOMP components
        vipScores(i,:) = sqrt(size(XL,1) * sum(bsxfun(@times,sumSq,W0.^2),2) ./ sum(sumSq,2));        

        Y_pred_logo = polypred(X_test_logo,beta,XL,YL,XW,nComp);
        mseGroup(i) = sqrt(mean((Y_test_logo - Y_pred_logo).^2));
    end
    
    % Average MSE Across Groups
    mseCV(nComp) = mean(mseGroup);
    % Store Average VIP Scores Across Folds
    vipScoresMean(nComp,:) = mean(vipScores, 1);
end

%% Plotting the RMSE evolution
if plt==1
    figure;
    plot(1:maxComponents, mseCV, '-o');
    xlabel('Number of Components');
    ylabel('Cross-Validated MSE');
    title('LOGO Cross-Validation');
    grid on;
end

%% Plot VIP scores for understanding
th = 1;
optimalComp = find(mseCV == min(mseCV), 1); % Find the optimal number of components
vipOptimal = mean(vipScores,1); % VIP scores for optimal components
[~,idxVipFeat]=find(vipOptimal>=th);
if plt==1
    figure;
    fAxis = freqAxis;
    stem(fAxis,vipOptimal);
    clusters = clusterVips(idxVipFeat.');
    pink_color = [1.0, 0.75, 0.8];
    for nCls = 1:length(clusters)
        cluster = clusters{:,nCls};
        if length(cluster)>1
            % Define the rectangle coordinates
            rectangle('Position',[fAxis(min(cluster)) 0 fAxis(max(cluster))-fAxis(min(cluster)) 2],'EdgeColor','r','FaceColor',[1, 0, 0, 0.5])
        end
    end
    yline(th,"LineWidth",2)
    xlabel('\nu (GHz)');
    ylabel('Average VIP Score');
    title('Averaged VIP Scores for ');
    xlim([min(freqAxis), max(freqAxis)])
    ylim([0 2]);
    grid on;
    figure(hS21);
    for nCls = 1:length(clusters)
        cluster = clusters{:,nCls};
        if length(cluster)>1
            % Define the rectangle coordinates
            rectangle('Position',[fAxis(min(cluster)) 0 fAxis(max(cluster))-fAxis(min(cluster)) 15],'EdgeColor','r','FaceColor',[1, 0, 0, 0.5])
        end
    end
     ylim([0 15]);
end

%% Peformance Analysis
%Performance during cross-validatios
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%% PERFORMANCE DURING LOGO CROSS VALIDATION %%%%%%%%%%%%%%%%%%%');
[~, optimalComponents] = min(mseCV);
disp(['Optimal Number of Components: ', num2str(optimalComponents)]);
row_indices = randperm(size(X_train, 1));
[XLF, YLF,XWF, XSF, YSF, betaFinal, PLSstatsF] = polypls(X_train, Y_train, optimalComponents, k, options);
Y_pred_test = polypred(X_test,betaFinal,XLF,YLF,XW,optimalComponents);
mseTest = sqrt(mean((Y_test - Y_pred_test).^2));
RSS = sum((Y_test - Y_pred_test).^2);
TSS = sum((Y_test - mean(Y_test)).^2);
r2 = 1 - (RSS / TSS);
disp(['Cross-Validation RMSEP: ', num2str(mseTest)]);
disp(['Cross-Validation R^{2}: ', num2str(r2)]);
%Bland-Altman+model
minVal = min([Y_test; Y_pred_test]);
maxVal = max([Y_test; Y_pred_test]);
% Calculate Bias and Limits of Agreement
bias = mean(Y_pred_test - Y_test); % Mean difference
LoA = 1.96 * std(Y_pred_test - Y_test); % Limits of agreement (±1.96 * SD)
% Add Labels for LoA Lines
upperLoA = bias + LoA; % Upper LoA value
lowerLoA = bias - LoA; % Lower LoA value
disp(['Cross-Validation Upper LoA: ', num2str(upperLoA), ' (+1.96SD)']);
disp(['Cross-Validation Lower LoA: ', num2str(lowerLoA), ' (-1.96SD)']);
%Performance in external validation
disp('%%%% PERFORMANCE DURING EXTERNAL VALIDATION %%%%%%%%%%%%%%%%%%%%%%');
%Calculation of the prediction
Y_pred_test =  polypred(X_Test,betaFinal,XLF,YLF,XW,optimalComponents);
mseTest = sqrt(mean((y_Test - Y_pred_test).^2));
RSSinterp = sum((y_Test - Y_pred_test).^2);
TSSinterp = sum((y_Test - mean(y_Test)).^2);
r2interp = 1 - (RSSinterp / TSSinterp);
disp(['Test RMSEP: ', num2str(mseTest)]);
disp(['Test R^{2}: ', num2str(r2)]);
%Bland-Altman+model
minVal = min([y_Test; Y_pred_test]);
maxVal = max([y_Test; Y_pred_test]);
% Calculate Bias and Limits of Agreement
bias = mean(Y_pred_test - y_Test); % Mean difference
LoA = 1.96 * std(Y_pred_test - y_Test); % Limits of agreement (±1.96 * SD)
% Add Labels for LoA Lines
upperLoA = bias + LoA; % Upper LoA value
lowerLoA = bias - LoA; % Lower LoA value
disp(['Test Upper LoA: ', num2str(upperLoA), ' (+1.96SD)']);
disp(['Test Lower LoA: ', num2str(lowerLoA), ' (-1.96SD)']);
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');




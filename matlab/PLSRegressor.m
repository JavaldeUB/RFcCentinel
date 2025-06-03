classdef PLSRegressor
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        s21Mag
        s21Ph
        freqAxis
        dataTr
        labelsTr
        dataTs
        labelsTs
    end

    methods
        function obj = PLSRegressor(fileMag, filePh)
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

            %Read the tables
            obj.s21Mag = readtable(fileMag, opts);
            obj.s21Ph = readtable(filePh, opts);
            obj.freqAxis = [];
            obj.dataTr = [];
            obj.dataTs = [];
            obj.labelsTr = [];
            obj.labelsTs = [];
            clear opts;
            
        end

        function obj = selectData(obj,magOrPh)
            % Import the data
            fAxis = [obj.s21Mag{1,2:end}]; 
            obj.freqAxis = fAxis(51:185)./1e9;
            
            if magOrPh == 0
                yr = obj.s21Mag{2:end,1};
                Xr = obj.s21Mag{2:end,2:end};
                Xr = Xr(:,51:185);
                baseline = Xr(709,:);
                Xr = Xr./baseline;
                rowsToRemove = [1,2,101,202,303,405,506,607,708,709];
                rowsToKeep = setdiff(1:size(Xr, 1), rowsToRemove);
                Xc = Xr(rowsToKeep, :);
                yc = yr(rowsToKeep);
                [rowsInterp, ~]=find(yc==[80,60,40]);
                rowsLeft = setdiff(1:size(Xc, 1), rowsInterp);
                obj.dataTs = Xc(rowsInterp,:);
                obj.labelsTs = yc(rowsInterp);
                obj.dataTr = Xc(rowsLeft,:);
                obj.labelsTr = yc(rowsLeft);
                
            elseif magOrPh == 1
                yr = obj.s21Mag{2:end,1};
                XrMag = obj.s21Mag{2:end,2:end};
                XrMag = XrMag(:,51:185);
                XrPh = obj.s21Ph{2:end,2:end};
                XrPh = XrPh(:,51:185);
                baselineMg = XrMag(709,:);
                baselinePh = XrPh(709,:);
                Xr = XrMag.*exp(1i.*XrPh);
                baseline = baselineMg.*exp(1i.*baselinePh);
                
                rowsToRemove = [1,2,101,202,303,405,506,607,708,709];
                rowsToKeep = setdiff(1:size(Xr, 1), rowsToRemove);
                Xc = Xr(rowsToKeep, :);
                yc = yr(rowsToKeep);
                [rowsInterp, ~]=find(yc==[80,60,40]);
                rowsLeft = setdiff(1:size(Xc, 1), rowsInterp);
                obj.dataTs = Xc(rowsInterp,:);
                obj.labelsTs = yc(rowsInterp);
                obj.dataTr = Xc(rowsLeft,:);
                obj.labelsTr = yc(rowsLeft);
            end

        end
    end
end
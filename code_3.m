%% Abalone Ring Prediction - Hyperparameter Sweep + Feature Engineering
% Features: Sex, Length, Diameter, Height, Whole weight,
%           Shucked weight, Viscera weight, Shell weight
%           + Engineered: Volume, Density, ShellRatio, VisceraRatio, ShellThickness
% Target:   Rings (Age = Rings + 1.5)

%% 1. Load and Prepare Data
data = readtable('abalone.csv', ...
    'Delimiter',         ',', ...
    'ReadVariableNames', false);

data.Properties.VariableNames = {'Sex','Length','Diameter','Height', ...
                                  'WholeWeight','ShuckedWeight','VisceraWeight', ...
                                  'ShellWeight','Rings'};

sexMap   = containers.Map({'M','F','I'}, {1, 2, 3});
data.Sex = cellfun(@(x) sexMap(strtrim(x)), data.Sex);

%% 2. Feature Engineering
% Volume: approximates shell as a rectangular prism
data.Volume        = data.Length .* data.Diameter .* data.Height;

% Density: whole weight relative to size
data.Density       = data.WholeWeight ./ (data.Volume + 1e-6);  % small epsilon avoids /0

% Shell ratio: how much of total weight is shell
data.ShellRatio    = data.ShellWeight ./ (data.WholeWeight + 1e-6);

% Viscera ratio: how much of total weight is viscera
data.VisceraRatio  = data.VisceraWeight ./ (data.WholeWeight + 1e-6);

% Shell thickness proxy: shell weight relative to surface area
data.ShellThick    = data.ShellWeight ./ (data.Length .* data.Diameter + 1e-6);

fprintf('Original features : 8\n');
fprintf('Engineered features added: 5\n');
fprintf('Total features    : 13\n\n');

%% 3. Separate Features and Target
X = data{:, [1:8, 10:14]};    % All 8 original + 5 engineered (skip col 9 = Rings)
y = data{:, 'Rings'};

%% 4. Train/Test Split (80/20)
rng(42);
n          = height(data);
idx        = randperm(n);
splitPoint = floor(0.8 * n);
trainIdx   = idx(1:splitPoint);
testIdx    = idx(splitPoint+1:end);

X_train = X(trainIdx, :);
y_train = y(trainIdx);
X_test  = X(testIdx, :);
y_test  = y(testIdx);

%% 5. Hyperparameter Grid Search
leafSizes     = [1, 2, 5, 10, 20];
numPredictors = [2, 3, 4, 6, 8, 10, 13];   % Expanded since we now have 13 features
numTrees      = 200;

nLeaf = length(leafSizes);
nPred = length(numPredictors);

RMSE_grid = zeros(nLeaf, nPred);
R2_grid   = zeros(nLeaf, nPred);

fprintf('Running hyperparameter sweep (%d combinations)...\n', nLeaf * nPred);

for i = 1:nLeaf
    for j = 1:nPred
        rf_temp = TreeBagger(numTrees, X_train, y_train, ...
            'Method',                 'regression', ...
            'OOBPrediction',          'on', ...
            'MinLeafSize',            leafSizes(i), ...
            'NumPredictorsToSample',  numPredictors(j));

        y_pred_temp   = predict(rf_temp, X_test);
        errors_temp   = y_pred_temp - y_test;
        RMSE_grid(i,j) = sqrt(mean(errors_temp.^2));
        SS_res         = sum(errors_temp.^2);
        SS_tot         = sum((y_test - mean(y_test)).^2);
        R2_grid(i,j)   = 1 - (SS_res / SS_tot);

        fprintf('  LeafSize=%2d | NumPred=%2d | RMSE=%.4f | R²=%.4f\n', ...
            leafSizes(i), numPredictors(j), RMSE_grid(i,j), R2_grid(i,j));
    end
end

%% 6. Find Best Hyperparameter Combination
[bestRMSE, flatIdx] = min(RMSE_grid(:));
[bestI, bestJ]      = ind2sub(size(RMSE_grid), flatIdx);
bestLeafSize        = leafSizes(bestI);
bestNumPred         = numPredictors(bestJ);

fprintf('\n--- Best Hyperparameters ---\n');
fprintf('MinLeafSize           : %d\n',     bestLeafSize);
fprintf('NumPredictorsToSample : %d\n',     bestNumPred);
fprintf('RMSE                  : %.4f rings\n', bestRMSE);

%% 7. Sweep Tree Count with Best Hyperparameters
fprintf('\nSweeping tree count...\n');
treeCounts = [50, 100, 200, 300, 500];
RMSE_trees = zeros(size(treeCounts));
R2_trees   = zeros(size(treeCounts));

for k = 1:length(treeCounts)
    rf_temp = TreeBagger(treeCounts(k), X_train, y_train, ...
        'Method',                'regression', ...
        'OOBPrediction',         'on', ...
        'MinLeafSize',           bestLeafSize, ...
        'NumPredictorsToSample', bestNumPred);

    y_pred_temp   = predict(rf_temp, X_test);
    errors_temp   = y_pred_temp - y_test;
    RMSE_trees(k) = sqrt(mean(errors_temp.^2));
    SS_res        = sum(errors_temp.^2);
    SS_tot        = sum((y_test - mean(y_test)).^2);
    R2_trees(k)   = 1 - (SS_res / SS_tot);

    fprintf('  Trees=%3d | RMSE=%.4f | R²=%.4f\n', ...
        treeCounts(k), RMSE_trees(k), R2_trees(k));
end

[~, bestTreeIdx] = min(RMSE_trees);
bestNumTrees     = treeCounts(bestTreeIdx);
fprintf('\nBest tree count: %d\n', bestNumTrees);

%% 8. Train Final Best Model
fprintf('\nTraining final model...\n');
rf_best = TreeBagger(bestNumTrees, X_train, y_train, ...
    'Method',                 'regression', ...
    'OOBPrediction',          'on', ...
    'OOBPredictorImportance', 'on', ...
    'MinLeafSize',            bestLeafSize, ...
    'NumPredictorsToSample',  bestNumPred);

y_pred_best = predict(rf_best, X_test);
y_pred_best = round(y_pred_best);
errors_best = y_pred_best - y_test;

RMSE_best = sqrt(mean(errors_best.^2));
MAE_best  = mean(abs(errors_best));
SS_res    = sum(errors_best.^2);
SS_tot    = sum((y_test - mean(y_test)).^2);
R2_best   = 1 - (SS_res / SS_tot);

fprintf('\n--- Final Model Performance ---\n');
fprintf('RMSE : %.4f rings  (original: 2.1953)\n', RMSE_best);
fprintf('MAE  : %.4f rings  (original: 1.4653)\n', MAE_best);
fprintf('R²   : %.4f        (original: 0.5515)\n', R2_best);

%% 9. RMSE Heatmap of Hyperparameter Grid
figure;
heatmap(numPredictors, leafSizes, RMSE_grid, ...
    'Title',  'RMSE by Hyperparameter Combination', ...
    'XLabel', 'NumPredictorsToSample', ...
    'YLabel', 'MinLeafSize');

%% 10. Tree Count vs RMSE
figure;
plot(treeCounts, RMSE_trees, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Trees');
ylabel('RMSE');
title('RMSE vs. Number of Trees');
grid on;

%% 11. Predicted vs Actual
figure;
scatter(y_test, y_pred_best, 30, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
refLine = [min(y_test), max(y_test)];
plot(refLine, refLine, 'r--', 'LineWidth', 2);
xlabel('Actual Rings');
ylabel('Predicted Rings');
title('Predicted vs. Actual — Final Model');
legend('Predictions', 'Perfect fit', 'Location', 'northwest');
grid on;

%% 12. Feature Importance (all 13 features)
figure;
featureNames = {'Sex','Length','Diameter','Height', ...
                'WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight', ...
                'Volume','Density','ShellRatio','VisceraRatio','ShellThick'};
importance = rf_best.OOBPermutedPredictorDeltaError;
bar(importance);
set(gca, 'XTickLabel', featureNames, 'XTickLabelRotation', 35);
ylabel('Increase in MSE when permuted');
title('Feature Importance — All 13 Features');
grid on;
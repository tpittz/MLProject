%% Hyperparameter Sweep - Random Forest Regression
% Tests combinations of MinLeafSize and NumPredictorsToSample
% then re-trains the best model with increasing tree counts

%% 1. Load and Prepare Data (same as before)
data = readtable('abalone.csv', ...
    'Delimiter',         ',', ...
    'ReadVariableNames', false);

data.Properties.VariableNames = {'Sex','Length','Diameter','Height', ...
                                  'WholeWeight','ShuckedWeight','VisceraWeight', ...
                                  'ShellWeight','Rings'};

sexMap   = containers.Map({'M','F','I'}, {1, 2, 3});
data.Sex = cellfun(@(x) sexMap(strtrim(x)), data.Sex);

X = data{:, 1:8};
y = data{:, 'Rings'};

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

%% 2. Define Hyperparameter Grids
leafSizes     = [1, 2, 5, 10, 20];        % Tree depth control
numPredictors = [2, 3, 4, 6, 8];          % Features per split
numTrees      = 200;                        % Fixed during sweep (enough to be stable)

nLeaf = length(leafSizes);
nPred = length(numPredictors);

% Storage for results
RMSE_grid = zeros(nLeaf, nPred);
R2_grid   = zeros(nLeaf, nPred);

fprintf('Running hyperparameter sweep (%d combinations)...\n', nLeaf * nPred);

%% 3. Grid Search
for i = 1:nLeaf
    for j = 1:nPred
        rf_temp = TreeBagger(numTrees, X_train, y_train, ...
            'Method',                 'regression', ...
            'OOBPrediction',          'on', ...
            'MinLeafSize',            leafSizes(i), ...
            'NumPredictorsToSample',  numPredictors(j));

        y_pred_temp = predict(rf_temp, X_test);
        errors_temp = y_pred_temp - y_test;

        RMSE_grid(i,j) = sqrt(mean(errors_temp.^2));
        SS_res         = sum(errors_temp.^2);
        SS_tot         = sum((y_test - mean(y_test)).^2);
        R2_grid(i,j)   = 1 - (SS_res / SS_tot);

        fprintf('  LeafSize=%2d | NumPred=%d | RMSE=%.4f | R²=%.4f\n', ...
            leafSizes(i), numPredictors(j), RMSE_grid(i,j), R2_grid(i,j));
    end
end

%% 4. Find Best Combination
[bestRMSE, flatIdx] = min(RMSE_grid(:));
[bestI, bestJ]      = ind2sub(size(RMSE_grid), flatIdx);
bestLeafSize        = leafSizes(bestI);
bestNumPred         = numPredictors(bestJ);

fprintf('\n--- Best Hyperparameters ---\n');
fprintf('MinLeafSize           : %d\n', bestLeafSize);
fprintf('NumPredictorsToSample : %d\n', bestNumPred);
fprintf('RMSE                  : %.4f rings\n', bestRMSE);

%% 5. Sweep Tree Count Using Best Hyperparameters
fprintf('\nSweeping tree count with best hyperparameters...\n');
treeCounts  = [50, 100, 200, 300, 500];
RMSE_trees  = zeros(size(treeCounts));
R2_trees    = zeros(size(treeCounts));

for k = 1:length(treeCounts)
    rf_temp = TreeBagger(treeCounts(k), X_train, y_train, ...
        'Method',                'regression', ...
        'OOBPrediction',         'on', ...
        'MinLeafSize',           bestLeafSize, ...
        'NumPredictorsToSample', bestNumPred);

    y_pred_temp  = predict(rf_temp, X_test);
    errors_temp  = y_pred_temp - y_test;
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

%% 6. Train Final Best Model
fprintf('\nTraining final model with best hyperparameters...\n');
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
fprintf('RMSE : %.4f rings\n', RMSE_best);
fprintf('MAE  : %.4f rings\n', MAE_best);
fprintf('R²   : %.4f\n',       R2_best);

%% 7. RMSE Heatmap of Hyperparameter Grid
figure;
heatmap(numPredictors, leafSizes, RMSE_grid, ...
    'Title',      'RMSE by Hyperparameter Combination', ...
    'XLabel',     'NumPredictorsToSample', ...
    'YLabel',     'MinLeafSize', ...
    'ColorbarVisible', 'on');

%% 8. Tree Count vs RMSE Plot
figure;
plot(treeCounts, RMSE_trees, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Trees');
ylabel('RMSE');
title('RMSE vs. Number of Trees (Best Hyperparameters)');
grid on;

%% 9. Predicted vs Actual - Final Model
figure;
scatter(y_test, y_pred_best, 30, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
refLine = [min(y_test), max(y_test)];
plot(refLine, refLine, 'r--', 'LineWidth', 2);
xlabel('Actual Rings');
ylabel('Predicted Rings');
title('Predicted vs. Actual — Best Model');
legend('Predictions', 'Perfect fit', 'Location', 'northwest');
grid on;

%% 10. Feature Importance - Final Model
figure;
featureNames = {'Sex','Length','Diameter','Height', ...
                'WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight'};
importance = rf_best.OOBPermutedPredictorDeltaError;
bar(importance);
set(gca, 'XTickLabel', featureNames, 'XTickLabelRotation', 30);
ylabel('Increase in MSE when permuted');
title('Feature Importance — Best Model');
grid on;
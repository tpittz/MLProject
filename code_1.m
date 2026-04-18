%% Abalone Ring Prediction - Random Forest Regression
% Features: Sex, Length, Diameter, Height, Whole weight,
%           Shucked weight, Viscera weight, Shell weight
% Target:   Rings (Age = Rings + 1.5)

%% 1. Load Data
data = readtable('abalone.csv', ...
    'Delimiter',         ',', ...
    'ReadVariableNames', false);

data.Properties.VariableNames = {'Sex','Length','Diameter','Height', ...
                                  'WholeWeight','ShuckedWeight','VisceraWeight', ...
                                  'ShellWeight','Rings'};

%% 2. Encode Categorical Variable (Sex: M=1, F=2, I=3)
sexMap = containers.Map({'M','F','I'}, {1, 2, 3});
data.Sex = cellfun(@(x) sexMap(strtrim(x)), data.Sex);

%% 3. Separate Features and Target
X = data{:, 1:8};          % All 8 feature columns
y = data{:, 'Rings'};      % Target: ring count

%% 4. Train/Test Split (80% train, 20% test)
rng(42);                   % Seed for reproducibility
n = height(data);
idx = randperm(n);

splitPoint  = floor(0.8 * n);
trainIdx    = idx(1:splitPoint);
testIdx     = idx(splitPoint+1:end);

X_train = X(trainIdx, :);
y_train = y(trainIdx);
X_test  = X(testIdx, :);
y_test  = y(testIdx);

%% 5. Train Random Forest (100 trees)
numTrees = 100;
rf = TreeBagger(numTrees, X_train, y_train, ...
    'Method',                  'regression', ...
    'OOBPrediction',           'on', ...
    'OOBPredictorImportance',  'on', ...
    'NumPredictorsToSample',   3, ...
    'MinLeafSize',             5);

%% 6. Predict on Test Set
y_pred = predict(rf, X_test);
y_pred = round(y_pred);               % Round to nearest integer ring count

%% 7. Regression Metrics
errors  = y_pred - y_test;
RMSE    = sqrt(mean(errors.^2));
MAE     = mean(abs(errors));
SS_res  = sum(errors.^2);
SS_tot  = sum((y_test - mean(y_test)).^2);
R2      = 1 - (SS_res / SS_tot);

fprintf('\n--- Model Performance ---\n');
fprintf('RMSE : %.4f rings\n', RMSE);
fprintf('MAE  : %.4f rings\n', MAE);
fprintf('R²   : %.4f\n',       R2);
fprintf('\nAge prediction error (RMSE): %.4f years\n', RMSE);
% (Since Age = Rings + 1.5, error in rings == error in age)

%% 8. Predicted vs Actual Plot (regression analog of confusion matrix)
figure;
scatter(y_test, y_pred, 30, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
refLine = [min(y_test), max(y_test)];
plot(refLine, refLine, 'r--', 'LineWidth', 2);   % Perfect prediction line
xlabel('Actual Rings');
ylabel('Predicted Rings');
title('Predicted vs. Actual Ring Count');
legend('Predictions', 'Perfect fit', 'Location', 'northwest');
grid on;

%% 9. Out-of-Bag Error Curve (checks if 100 trees is enough)
figure;
plot(oobError(rf), 'b-', 'LineWidth', 1.5);
xlabel('Number of Trees');
ylabel('Out-of-Bag MSE');
title('OOB Error vs. Number of Trees');
grid on;

%% 10. Feature Importance
figure;
featureNames = {'Sex','Length','Diameter','Height', ...
                'WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight'};
importance = rf.OOBPermutedPredictorDeltaError;
bar(importance);
set(gca, 'XTickLabel', featureNames, 'XTickLabelRotation', 30);
ylabel('Increase in MSE when permuted');
title('Feature Importance');
grid on;
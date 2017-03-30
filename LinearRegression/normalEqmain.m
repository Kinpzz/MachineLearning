%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = csvread('save_train.csv',1,1);
X = data(:, 1:size(data,2)-1);
y = data(:, size(data,2));
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

theta = normalEqn(X, y);

RMSE = sqrt(mean((theta'*X'-y').^2));


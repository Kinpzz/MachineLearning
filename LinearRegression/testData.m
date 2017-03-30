clear ; close all; clc
fprintf('Loading data ...\n');
%% Load Data
data = csvread('save_train.csv',1,1);
X = data(:, 1:size(data,2)-1);
y = data(:, size(data,2));
m = length(y);
theta = csvread('result.csv',1,1);
%% 
% Add intercept term to X
X = [ones(m, 1) X];
%% compute RMSE
RMSE = sqrt(mean((theta'*X'-y').^2))

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

[X, mu, sigma, dim_rescale] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.03;
num_iters = 500;
lambda = 0;
indices =  crossvalind('Kfold', y, 10);
inside_RMSE = 0;
outside_RMSE = 0;
for lambda = 0:0.0000001:0.0000002
    %% ten-fold validation
    for i = 1:10
        test = (indices == i);
        train = ~test;
        train_set_x = X(train,:);
        train_set_y = y(train,:);
        test_set_x = X(test,:);
        test_set_y = y(test,:);
        % Init Theta and Run Gradient Descent 
        theta = zeros(size(X,2), 1);
        [theta, J_history] = gradientDescentMulti(train_set_x, train_set_y, theta, alpha, lambda, num_iters);
        inside_RMSE = inside_RMSE + sqrt(mean((train_set_x*theta - train_set_y).^2));
        outside_RMSE = outside_RMSE + sqrt(mean((test_set_x*theta - test_set_y).^2));
        % Plot the convergence graph
        %figure;
        %plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
        %xlabel('Number of iterations');
        %ylabel('Cost J');
        % Display gradient descent's result

    end
    inside_RMSE = inside_RMSE / 10;
    outside_RMSE = outside_RMSE / 10;
    fprintf('inside RMSE: %.5f \n', inside_RMSE);
    fprintf('outside RMSE: %.5f \n', outside_RMSE);
end
%% predict data
fprintf('Loading test set ...\n');
pred_data = csvread('save_test.csv',1,1);
X_pred = pred_data(:, 1:size(pred_data,2));
X_pred = (X_pred(:,dim_rescale) - mu(dim_rescale))./sigma(dim_rescale);
X_pred = [ones(size(X_pred,1), 1) X_pred];

y_pred = X_pred*theta;
%% write file
fid = fopen('result.csv', 'w') ;
fprintf(fid, 'id,reference\n') ;
for i = 1 : 25000
    fprintf(fid,'%d,%.15f\n',i-1,y_pred(i));
end
fclose(fid) ;

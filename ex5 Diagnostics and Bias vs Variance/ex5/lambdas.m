close all;
qq = 0:3;
offset = 2;
div = 2;
for j=qq
  lambda = offset + j/div;

[theta] = trainLinearReg(X_poly, y, lambda);
% Plot training data and fit
figure(1);
hold on;
plot(X, y, 'rx', 'MarkerSize', j, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
hold on;
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
%legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

leg1{j+1}= num2str(lambda, 'lambda %d');
leg2{j*2+1}= num2str(lambda, 'lambda %d - train');
leg2{j*2+2}= num2str(lambda, 'lambda %d - cv');
pause (6);
end
figure(2);
legend(leg2);
figure(1);
legend (leg1);

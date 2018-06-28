function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% first bet is that X, y and theta are correctly formed
% so theta n x 1 vector
% X is a m x n vector where m = number of training measurements and n is the number of features
% Y is a m x 1 vector
% disp('Cost Function for');
% X, y, theta

J = sum((X*theta-y).^2)/(2*m);

% =========================================================================

end

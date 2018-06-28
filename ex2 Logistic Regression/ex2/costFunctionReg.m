function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
% J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
polynomial = X*theta; % going straight for all xs rather than x^(i) one by one
hypothesis = sigmoid(polynomial);
cost = -y.*log(hypothesis)-(1-y).*log(1-hypothesis);

n=length(theta);
regularization=lambda/2/m*sum(theta(2:n).^2);%remember that we dont regularize theta_0 which is 1 indexed in octave

J=(mean(cost)+regularization);

gradients = X'*(hypothesis-y)./m;
grad_regularizations = lambda/m.*theta;
grad_regularizations(1) = 0; % remember we don't regularize our theta_0
grad = gradients + grad_regularizations;






% =============================================================

end

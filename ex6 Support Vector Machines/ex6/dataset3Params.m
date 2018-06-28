function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

error = [0 0 0]; % dummy row will delete this row as its a convenience

for C = Cs
  for sigma = sigmas
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    pred = svmPredict(model, Xval);
    error = [error; C sigma sum(pred~=yval)];
  end
end

error = error(2:end,:); % get rid of dummy row

[min_err min_idx] = min(error(:,3))

C = error(min_idx,1)
sigma = error(min_idx,2)

% =========================================================================

end
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
%J = 0;
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% calculate hypothesis h_theta

  bias = ones(size(X,1))(:,1); %need ones all the way down to implement weights for bias

  a1 = [bias X]; % 5000 x 401 matrix

  z2 = (Theta1 * a1')'; % 5000 x 25 matrix
  a2_nobias = sigmoid(z2); % 5000 x 25 matrix
  a2 = [bias a2_nobias]; % 5000 x 26 matrix

  z3 = (Theta2 * a2')'; % 5000 x 10 matrix
  a3_nobias = sigmoid(z3); % 5000 x 10 matrix
  a3 = [bias a3_nobias]; % 5000 x 11 matrix

  h_theta = a3_nobias; % 5000 x 10 matrix - for completeness we identify h_theta

  k = num_labels; %number of output classes k

% generate y vectors

  ys = ((y.*ones(m, k)) == [1:k]); % make 0/1 vectors for all 5000 mesaurements. 5000 x 10 matrix

% calculate cost per y and h_theta

  cost = -ys.*log(h_theta) - (1-ys).*log(1-h_theta); % 5000 x 10 matrix
  sum_ks = sum(cost,2);
  sum_ms = sum(sum_ks);

% compute regularization against all non-zero indexed theta. Remembering Octave puts that at position 1
% note this code only works for two layers but their Theta1 and 2 concepts only works for it, so its not a prob

  Theta1_squ_sum = sum(sum(Theta1(:,2:end).^2)); % remember excluding first element
  Theta2_squ_sum = sum(sum(Theta2(:,2:end).^2));
  regularization = lambda / (2*m) * (Theta1_squ_sum + Theta2_squ_sum);

% calculate cost function output J

  J = sum_ms / m + regularization;


% calculate back propogation gradients

  delta_3s = h_theta - ys; % 5000 x 10 matrix

  delta_2s = delta_3s * Theta2(:,2:end).*sigmoidGradient(z2); % 5000 x 25 matrix

  DELTA2 = delta_3s' * a2; % 10 x 26 matrix

  DELTA1 = delta_2s' * a1; % 25 x 401 matrix

  lambda_m_Theta2 = lambda / m * Theta2;
  gradreg_layer_2 = [zeros(size(Theta2,1),1) lambda_m_Theta2(:,2:end)]; % 10 26

  lambda_m_Theta1 = lambda / m * Theta1;
  gradreg_layer_1 = [zeros(size(Theta1,1),1) lambda_m_Theta1(:,2:end)]; % 25 x 401

  Theta2_grad = DELTA2/m + gradreg_layer_2;
  Theta1_grad = DELTA1/m + gradreg_layer_1;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

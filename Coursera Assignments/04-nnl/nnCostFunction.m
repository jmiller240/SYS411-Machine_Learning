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
K = num_labels;

% Add bias unit to Xs
X = [ones(m,1) X];

% Part 1: Feedforward

h1 = sigmoid(X * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

J = 0;
for row = 1:m
  a = 1:10;
  y_vector = (a == y(row))(:);
  for k = 1:K
    J = J + (((-1 * y_vector(k))*(log(h2(row,k)))) - ((1 - y_vector(k))*(log(1-h2(row,k)))));
  endfor
endfor

J = J * (1/m);

% Regularize JCOST
theta1Squared = Theta1 .^ 2;
theta2Squared = Theta2 .^ 2;

reg = sum(sum(theta1Squared(:,2:end))) + sum(sum(theta2Squared(:,2:end)));
reg = reg * (lambda/(2*m));

J = J + reg;


% Part 2: Backprop

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t = 1:m
  % 1
  % Theta1 = 25 x 401
  % Theta2 = 10 x 26
  
  a1 = X(t, :);       % 1 x 401
  
  z2 = Theta1 * a1';  % 25 x 1 
  a2 = sigmoid(z2);   % 25 x 1

  a2 = [1; a2];       % 26 x 1

  z3 = Theta2 * a2;   % 10 x 1
  a3 = sigmoid(z3);   % 10 x 1
  
##  [val idx] = max(a3);
##  
##  b = 1:num_labels;
##  results = (b == idx);
  
##  if t == 1
##    display(results);
##  endif

  % 2
  b = 1:num_labels;
  y_vector = (b == y(t))(:);
  s3 = a3 - y_vector;

  % 3
  s2 = (Theta2'*s3) .* (a2 .* (1 - a2));
  
  % 4
  s2 = s2(2:end);
  Theta2_grad = Theta2_grad + (s3*a2');
  Theta1_grad = Theta1_grad + (s2*a1);
  
endfor


% 5
Theta2_grad = (1/m) .* Theta2_grad;
Theta1_grad = (1/m) .* Theta1_grad;


% Regularize gradients
Theta2_grad = [Theta2_grad(:, 1) (Theta2_grad(:, 2:end) + ((lambda/m) .* Theta2(:, 2:end)))];
Theta1_grad = [Theta1_grad(:, 1) (Theta1_grad(:, 2:end) + ((lambda/m) .* Theta1(:, 2:end)))];


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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

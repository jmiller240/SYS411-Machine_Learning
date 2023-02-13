function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

disp(size(theta));

h = X * theta;
sum_squares = sum((h - y) .^ 2);
sum_squared_thetas = sum(theta(2:end) .^ 2);

J = (1/(2*m))*sum_squares + (lambda/(2*m))*sum_squared_thetas;

for i = 1:size(theta, 1)
  residuals = sum((h - y) .* X(:,i));
  grad(i) = (1/m)*residuals;
  if i > 1
    grad(i) = grad(i) + (lambda/m)*theta(i);
  endif
endfor

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end

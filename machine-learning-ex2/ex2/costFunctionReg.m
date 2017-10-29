function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%cost
h = @(theta, x) sigmoid( dot(theta, x) );

for i = 1 : m
	hypothesis = h( theta, X(i, :) );
	cost_for_ith_row = - y(i) * log(hypothesis) - (1 - y(i)) * log(1 - hypothesis);
	J = J + cost_for_ith_row;
end

regularization_term  = lambda / (2 * m) * dot( theta(2:end), theta'(2:end) );
J = J / m; 
J = J + regularization_term

%gradient
for i = 1 : m
	hypothesis = h( theta, X(i, :) );
	gradient_for_ith_row = ( (hypothesis - y(i)) * X(i, :) )'
	grad = grad + gradient_for_ith_row
end

regularization_term = zeros(size(theta))
regularization_term(2:end) = lambda / m * theta(2:end)
grad = grad / m + regularization_term

% =============================================================

end

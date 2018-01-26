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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

% encode y
y_temp = zeros(m, num_labels);
for i = 1 : m
	y_temp(i, y(i)) = 1;
end
y = y_temp;


cost_holder = 0;

big_delta_1 = zeros(size(Theta1));
big_delta_2 = zeros(size(Theta2));

with_bias = @(X) [1; X];

for i = 1:m
	% forwad pass
	input = with_bias(transpose(X(i, :)));

	input2 = Theta1 * input;

	hidden_layer = sigmoid(input2);

	output = sigmoid( Theta2 * with_bias(hidden_layer) );

	label = y(i, :);

	single_cost = -label * log(output) - (1 - label) * log(1 - output) ;

	cost_holder = cost_holder + single_cost;

	% backpropagation
	delta_3 = output - label.';
	delta_2 = (Theta2.' * delta_3) .* with_bias(sigmoidGradient(input2));
	delta_2 = delta_2(2 : end);
	
	big_delta_2 = big_delta_2 + delta_3 * with_bias(hidden_layer).';
	big_delta_1 = big_delta_1 + delta_2 * input.';
end

D_1 = big_delta_1 ./ m;
D_2 = big_delta_2 ./ m;

% add regularization term for gradient
updateGradient = @(D, Theta) 
for j = 2 : size(D_1)(2)
	for i = 1 : size(D_1)(1)
		D_1(i, j) = D_1(i, j) + lambda / m * Theta1(i, j);
	end
end

for j = 2 : size(D_2)(2)
	for i = 1 : size(D_2)(1)
		D_2(i, j) = D_2(i, j) + lambda / m * Theta2(i, j);
	end
end

cost_without_regularization = cost_holder / m;

r_cost = @(Theta) sum( sum(Theta .* Theta) );

without_bias = @(Theta) Theta(:, 2:size(Theta)(2));
t1 = without_bias(Theta1);
t2 = without_bias(Theta2);

regularization_term = lambda / 2 / m * ( r_cost(t1) + r_cost(t2) );

J = cost_without_regularization + regularization_term;

Theta1_grad = D_1;
Theta2_grad = D_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

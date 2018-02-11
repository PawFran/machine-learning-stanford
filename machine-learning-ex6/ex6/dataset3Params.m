function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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


params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
err = zeros(size(params) .^ 2);

curr_idx = 0;

for C = params
       for sigma = params
	       model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
	       predictions = svmPredict(model, Xval);

	       curr_idx = curr_idx + 1;
	       err(curr_idx) = mean(double(predictions ~= yval));
       end
end

err = reshape(err, [8,8])';
[minval, index] = min(err(:));
[I,J] = ind2sub([size(err,1) size(err,2)], index);

C = params(I);
sigma = params(J);

% =========================================================================

end

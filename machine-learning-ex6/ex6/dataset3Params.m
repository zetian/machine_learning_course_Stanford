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
C_opt = 0;
sigma_opt = 0;
test_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
init_error = 1000000;
for i = 1:length(test_set)
	for j = 1:length(test_set)
		C_try = test_set(i);
		sigma_try = test_set(j);
		model = svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try));
		predictions = svmPredict(model, Xval);
		error_ = mean(double(predictions ~= yval));
		if (error_ < init_error)
			init_error = error_;
			C_opt = C_try; 
			sigma_opt = sigma_try;
		end
	end
end


C = C_opt;
sigma = sigma_opt;




% =========================================================================

end

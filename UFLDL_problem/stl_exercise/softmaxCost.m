function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
m=size(data,2);
numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
M = theta * data;
M = bsxfun(@minus,M,max(M,[],1));
M = exp(M);
M1 = groundTruth .* log(bsxfun(@rdivide,M,sum(M)));
cost = -sum(sum(M1))/m + lambda*0.5*sum(sum(theta.^2));
thetagrad = -((groundTruth-bsxfun(@rdivide,M,sum(M)))*data')/m+lambda*theta;












% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end


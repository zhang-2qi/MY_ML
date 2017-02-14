function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)

    weightMatrix = reshape(theta, numFeatures, visibleSize);
    x=patches;
    cost = 0;
    grad = zeros(numFeatures, visibleSize);
    num_samples = size(patches,2);

    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------    
%     a=(weightMatrix*x).^2+epsilon;
%     cost = sum(sqrt(a))/m;
%     grad = weightMatrix*x ./ sqrt(a) *x';
a=weightMatrix'*weightMatrix*patches-patches;
b=(weightMatrix*patches).^2+epsilon;
    cost = sum(a(:).^2)./num_samples+...
            sum(sqrt(b(:)))/num_samples;
    grad = (2*weightMatrix*a*patches'+...
        2*weightMatrix*patches*a')./num_samples+...
        (weightMatrix*patches./sqrt(b))*patches'/num_samples;
    grad=grad(:);
end


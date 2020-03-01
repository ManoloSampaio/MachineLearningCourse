function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    theta = theta -(alpha/m)*somatorio(m,theta,X,y);
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
function vetor = somatorio(m,theta,X,y)
    vetor = theta
    for i=1:length(transpose(theta))
        soma = 0;
        for j=1:m
          soma = ((transpose(theta)*transpose(X(j,:))-y(j))*(X(j,i)))+soma;
        end
        vetor(i,1)=soma;
     end
end
end

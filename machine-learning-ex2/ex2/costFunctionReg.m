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
soma1=0;
soma2=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

 grad = (1/m).*somatorio(theta,y,m,X,lambda);
 J = CostFunction(m,theta,X,y,lambda);
% =============================================================
end
function vetor=somatorio(theta,y,m,X,lambda)
  vetor= theta;
  soma =0;
  for j=1:m
     soma = soma+(((sigmoid(transpose(theta)*transpose(X(j,:)))-y(j))*X(j,1)));
  end
  vetor(1)=soma;
  for i=2:length(transpose(theta))
    soma = 0;
    for j=1:m
      soma = soma+(((sigmoid(transpose(theta)*transpose(X(j,:)))-y(j))*X(j,i)));
     end
    vetor(i)=soma+(theta(i)*lambda);
  end
end
function J=CostFunction(m,theta,X,y,lambda)
soma1 =0;
soma2 =0;
for i=1:m
soma1 = ((-y(i)*log(sigmoid(transpose(theta)*transpose(X(i,:)))))-((1-y(i))*log(1-sigmoid(transpose(theta)*transpose(X(i,:))))))+soma1;
end
for i=2:length(theta)
  soma2 = soma2+(theta(i))^2;
end
J = (soma1/m)+(soma2*lambda/(2*m));
end

function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
soma1=0;
soma2=0;
grad = (1/m).*somatorio(theta,y,m,X,lambda);
J = CostFunction(m,theta,X,y,lambda);
% =============================================================
grad = grad(:);
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

% =============================================================



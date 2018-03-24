function [J, grad] = costFunction(theta, X, y)
  J = 0
  for k = 1 : size(X, 1)
    x = X(k, :)
    J += -y(k) * log(h(theta, x)) - (1 - y(k)) * log(1 - h(theta, x))
  end
  J = J / size(X, 1)
  
  grad = zeros(length(theta), 1)
  for k = 1 : size(theta, 1)
    grad(k) = gradFunc(theta, X, y, k)
  end
  grad
end

% grad
function result = gradFunc(theta, X, y,index)
  result = 0
  for k = 1 : size(X, 1)
    x = X(k, :)
    if index == 1
      result += h(theta, x) - y(k)
    else
      result += (h(theta, x) - y(k)) * x(index)
    end
  end
  result = result / size(X, 1)
end

% hypothesize
function h = h(theta, x)
  h = sigmoid(l(theta, x))
end

% linear regression
function l = l(theta, x)
  l = x * theta
end

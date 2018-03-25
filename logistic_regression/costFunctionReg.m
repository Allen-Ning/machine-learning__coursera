function [J, grad] = costFunctionReg(theta, X, y, lambda)
  J = 0
  m = size(X, 1)
  for k = 1 : m
    x = X(k, :)
    J += -y(k) * log(h(theta, x)) - (1 - y(k)) * log(1 - h(theta, x))
  end
  J = J / size(X, 1) + regularize(theta, lambda, m)

  grad = zeros(length(theta), 1)
  for k = 1 : size(theta, 1)
    grad(k) = gradFunc(theta, X, y, k, lambda, m)
  end
end

% grad
function result = gradFunc(theta, X, y,index, lambda, m)
  result = 0
  for k = 1 : size(X, 1)
    x = X(k, :)
    if index == 1
      result += h(theta, x) - y(k)
    else
      result += (h(theta, x) - y(k)) * x(index) + theta(index) * lambda / m
    end
  end

  result = result / size(X, 1)
end

function result = regularize(theta, lambda, m)
  result = 0
  for i = 2: length(theta)
    result += theta(i) ^ 2
  end
  result = lambda * result / (2 * m)
end

% hypothesize
function h = h(theta, x)
  h = sigmoid(l(theta, x))
end

% linear regression
function l = l(theta, x)
  l = x * theta
end

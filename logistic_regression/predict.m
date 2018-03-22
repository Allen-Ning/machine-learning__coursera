function p = predict(theta, X)
  p =  zeros(size(X), 1)
  for k = 1 : size(X, 1)
    x = X(k, :)
    if sigmoid(x * theta) >= 0.5
      p(k) = 1
    else
     p(k) = 0
    end
  end
end

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
% There could be another function can be used here

g = 1.0 ./ (1.0 + exp(-z));
end

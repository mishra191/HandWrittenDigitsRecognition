function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
                 
% Theta1 is first output unit

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                           
% Theta 2 is 2nd output unit..
                 
% now, theta1 and theta2 are in matrix form....

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

num_labels = size(Theta2, 1);

A = [ones(size(Theta1,1), 1)];

B = [ones(num_labels, 1)];

% Add ones to the X data matrix
X = [ones(m, 1), X];

Y= [1:num_labels];

%delta4 = zeros(size(y,1), num_labels);

for num = 1:m
A = sigmoid(Theta1 * (X(num,:))');
A = [ones(1, 1); A];
B = sigmoid(Theta2 * A);
output = (Y == y(num));
%delta4(num) = B - output;
new_sum = - [(output'.*log(B)) + ((1-output').*log(1- B))];
J = J + sum(new_sum);
end

J = J/m;

if(lambda > 0)
Theta1_size = size(Theta1,2);
Theta2_size = size(Theta2,2);
Theta1 = Theta1(:,2:Theta1_size);
Theta2 = Theta2(:,2:Theta2_size);
Theta1 = Theta1.^2;
Theta2 = Theta2.^2;
sum_of_column_Theta1 = sum(Theta1, 2);
sum_of_column_Theta2 = sum(Theta2, 2);
total_sum_Theta1 = sum(sum_of_column_Theta1);
total_sum_Theta2 = sum(sum_of_column_Theta2);
total = (total_sum_Theta1+ total_sum_Theta2);
total = (lambda*total)/(2*m);

J =  J + total;

endif

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%need to ask, how many hidden units are there

% first find small delta-4




Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
                 
% Theta1 is first output unit

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%what should be size of delta2 here?                 
DELTA1 = zeros(size(Theta1));

%what should be size of delta3 here?  
DELTA2 = zeros(size(Theta2));

for num = 1:m

% forward propagation  
%Theta1 must have the size of 
A = sigmoid(Theta1 * (X(num,:))');
A = [ones(1, 1); A];
B = sigmoid(Theta2 * A);
output = (Y == y(num));

% perform delta3 at every iteration
% delta3 is 3*1
delta3 = B - output';

% delta2 has size of 6*1
delta2 = (Theta2'* delta3).*A.*(1-A);

% removing the first index from delta2
% delta2 had size 6 *1
%This delta2 will be caluclated everytime for every row..
% now size of delta 2 is recuced to 5*1
delta2 = delta2(2: size(delta2,1));

value = X(num,:);

%something is wrong here...it should update column instead of rows..
DELTA1 = DELTA1 + delta2*value;

% it should update column instead of rows....
DELTA2 = DELTA2 + delta3*A';


end

Theta1_grad = DELTA1/m;

Theta2_grad = DELTA2/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

if(lambda > 0)

Theta1_grad_regularized = (lambda/m)*Theta1_grad(:,2:size(Theta1_grad,2));
Theta1_grad_regularized = [zeros(size(Theta1_grad_regularized,1),1),Theta1_grad_regularized];
Theta1_grad = Theta1_grad + Theta1_grad_regularized;
Theta2_grad_regularized = (lambda/m)*Theta2_grad(:,2:size(Theta2_grad,2));
Theta2_grad_regularized = [zeros(size(Theta2_grad_regularized,1),1),Theta2_grad_regularized];
Theta2_grad = Theta2_grad + Theta2_grad_regularized;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

endif

% -------------------------------------------------------------

% =========================================================================


end

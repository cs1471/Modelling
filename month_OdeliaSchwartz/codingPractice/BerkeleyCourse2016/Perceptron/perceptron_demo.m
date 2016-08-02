%%% Rosenblatt's Perceptron learning.
% This is a simple example illustrating the classic perceptron algorithm
% A linear decision function parametrized by the weigh vector "w" and bias
% paramter "b" is learned by making small adjustements to these parameters
% every time the predicted label "f_i"  mismatches the true label "y_i" of 
% an input data point "x_i".
% The predicted label corresponds to the following function:
%             f_i = sign( < w, x_i> + b),
% where "< . , . >"  denotes the dot product operation.
% The update rule is given as follows:
% if "y_i ~= f_i" (predicted label f_i different from true label y_i) then
%          w <-- w + y_i*x_i; and  
%          b <-- b + y_i;
% else continue with next data sample
% The above process is repeated over the set of samples several times.
% If data points are linearly separable the above rule is guaranteed to
% converge in a finite number of iterations.
%
% 2016 Luis G Sanchez Giraldo and Odelia Schwartz

close all
clear all
clc
%% Construct a simple data set based on MNIST images
% This is a data set of handwritten digits 0 to 9
load('data/mnist_all.mat');
% Create a simple two-class problem using images of digits 0 and 5 from
% MNIST test data set  
pos_class = 0;
neg_class = 5;

%% get samples from positive and negative classes 
pos_data = eval(strcat('test', num2str(pos_class)));   
neg_data = eval(strcat('test', num2str(neg_class)));

%% Look at some digits from the classes
% Look at different samples from each class (here plotted just the first)
figure(1);
size(pos_data)
subplot(1,2,1)
imshow((reshape(pos_data(1,:),28,28))')
size(neg_data)
subplot(1,2,2)
imshow((reshape(neg_data(1,:),28,28))')

%% Gather the samples from the two classes into one matrix X
% Note that there area 936 samples from each class, and that
% these are appended to make up 1872 samples altogether
X = double([pos_data; neg_data])/255;
size(X)

%% Label the two classes with 1 and -1 respectively
Y = [ones(size(pos_data, 1), 1); -ones(size(neg_data, 1), 1)];
size(Y)

%% Choose random samples from data. To do so:
%% permute data samples to run the learning  algorithm 
% and take just n_samples from the permuted data (here 60 samples)
n_samples = 60;  
size(X,1)
[p_idx] = randperm(size(X, 1));
X = X(p_idx(1:n_samples), :);
Y = Y(p_idx(1:n_samples));

%% Project the data onto the means of the two classes
% First look at the mean of the two class
figure(2);
subplot(1,2,1);
imshow(reshape(mean(X(Y == 1, :)),28,28)')
% mean of second class
subplot(1,2,2);
imshow(reshape(mean(X(Y == -1, :)),28,28)')
% Now project the data
V(1, :) = mean(X(Y == 1, :))' ;     
V(1, :) = V(1, :)/norm(V(1, :)); 
V(2, :) = mean(X(Y == -1, :));
V(2, :) = V(2, :)/norm(V(2, :)); 
Z = X*V';
figure(3);
gscatter(Z(:,1), Z(:,2), Y);

%% Simple Learning algorithm for Perceptron
% here we denote two classes: the positive class by label "1"  and the negative
% class by label "-1." 
% Any point in the plane colored as green will be classified as positive
% class and any point falling within the red region as negative class.
% Training samples are denoted by the green crosses (positive) and red dots
% (negative). A missclassified training point, that is "f_i ~= y_i" is 
% marked with a circle 

lr = 1; % Learning rate parameter (1 in the classic perceptron algorithm)
w = randn(size(Z, 2), 1); % Initial guess for the hyperplane parameters
b = 0;                    % bias is initially zero
max_epoch = 100;              % Number of epoch (complete loops trough all data)
epoch = 1;                 % epoch counter
h = figure();
while epoch <= max_epoch
    % loop trough all data points one time (an epoch)
    for iSmp = 1:n_samples
        z_i = Z(iSmp, :)'; 
        % compute prediction with current w and b
        f_i = sign(w'*z_i + b);
        % update w and b if missclassified
        if f_i ~= Y(iSmp)
            w = w + lr*Y(iSmp)*z_i;
            b = b + lr*Y(iSmp);
        end
    end
    % diplay current decision hyperplane (partition of the space)
    display2DPartition(Z, Y, w, b, h);
    pause(0.1)
    epoch = epoch + 1;
end

% EXERCISE: After trying the code for the given classes,
% try running the code again, but this time changing the digits of the
% positive or negative class. 
% You can do this by changing the following two lines above:
% pos_class = 0;
% neg_class = 5;
% What classes are easier to learn?


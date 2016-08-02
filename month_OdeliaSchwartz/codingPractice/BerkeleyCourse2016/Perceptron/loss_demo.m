% Plot different loss functions 
% This code illustrates different commonly used loss functions
% for training neural nets:
% Zero-one loss (perceptron)
% Square loss (regression problems)
% Hinge loss (robust loss support vector machines)
% Cross-entropy and logistic loss (logistic regression or multinomial regression)
% 2016 Luis G Sanchez Giraldo and Odelia Schwartz

close all
clear all
clc

f = -2:0.01:2;
y = 1;
figure();
hold on
leg_lab = {};

%% Zero-one loss
L_zo = sign(f) ~= y;
plot(f, L_zo, 'r-', 'LineWidth', 3)
leg_lab = cat(1, leg_lab, 'Zero-One');

%% Hinge loss
L_hinge = max(0, 1 - y*f);
plot(f, L_hinge, 'b-', 'LineWidth', 3)
leg_lab = cat(1, leg_lab, 'Hinge');

%% Square loss
L_sq = (1 - y*f).^2;
plot(f, L_sq, 'g-', 'LineWidth', 3)
leg_lab = cat(1, leg_lab, 'Square');

%% logistic loss
L_log = log(1 + exp(-y*f))/log(2);
plot(f, L_log, 'k-', 'LineWidth', 3)
leg_lab = cat(1, leg_lab, 'Logistic');

%% Cross-entropy loss
% transform y into [0, 1] range using the logistic sigmoid function
f_p = 1./(1 + exp(-f));
% full cross entropy is given by ( -((y+1)/2)*log(f_p)- ((1-y)/2)*log(1-f) )/log(2) 
L_xe = ( -((y+1)/2)*log(f_p) - ((1-y)/2)*log(1-f_p) )/log(2);
plot(f, L_xe, 'c:', 'LineWidth', 3)
leg_lab = cat(1, leg_lab, 'Cross-entropy');

% pretty plot
xlim([-2, 2])
ylim([0, 4])
legend(leg_lab)



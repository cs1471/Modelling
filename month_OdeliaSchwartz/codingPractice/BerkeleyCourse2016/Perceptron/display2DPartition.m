function h = display2DPartition(X, Y, w, b, h)
% displayPartition
% Inputs:
% -- X : 2-dimensional array of size N x 2, where N is the number of data 
%        samples.
% -- Y : column vector of length N containing the data labels
% -- w : vector of length 2 representing the decision hyperplane. this vector
%        is actually normal to the decision hyperplane.
% -- b : bias parameter of the decision hyperplane.
% -- h : (Optional) figure handle to plot. If not provided, a new figure
%        handle will be created and provided as output
% Outputs:
% -- h : figure handle for the display

if ~exist('h', 'var')
    h = figure();
end

% compute ranges of the display based on the data
x1_lim = [min(X(:,1)) max(X(:,1))];
x1_range = x1_lim(2) - x1_lim(1);
x1_lim = x1_lim + [-x1_range/6, x1_range/6];
x2_lim = [min(X(:,2)) max(X(:,2))];
x2_range = x2_lim(2) - x2_lim(1);
x2_lim = x2_lim + [-x2_range/6, x2_range/6];

% create image of partition
n_points = 100;
x1_grid = x1_lim(1) : x1_range/n_points : x1_lim(2);   
x2_grid = x2_lim(1) : x2_range/n_points : x2_lim(2); 

X1_grid = repmat(x1_grid(:)', length(x2_grid), 1);
X2_grid = repmat(x2_grid(:), 1, length(x2_grid));
X_grid = [X1_grid(:), X2_grid(:)];
Y_grid = sign(X_grid*w(:) + b);
Y_grid = reshape(Y_grid, [length(x2_grid), length(x1_grid)]);
IM_grid = ones(size(Y_grid, 1), size(Y_grid, 2), 3)*204;
IM_grid(:, :, 1) = (Y_grid == -1)*41 + 204;
IM_grid(:, :, 2) = (Y_grid >= 0)*41 + 204;
figure(h)
image(x1_grid, x2_grid, IM_grid/255);
set(gca,'YDir','normal')

% plot data points
hold on
plot(X(Y==-1, 1), X(Y==-1, 2), 'r.', 'LineWidth', 3, 'MarkerSize', 10);
plot(X(Y== 1, 1), X(Y== 1, 2), 'g+', 'LineWidth', 3, 'MarkerSize', 10);

% circle misclassified points
Y_pred = sign(X*w(:) + b);
plot(X(Y_pred == -1 & Y == 1, 1), X(Y_pred == -1 & Y == 1, 2), 'go', 'LineWidth', 3, 'MarkerSize', 10);
plot(X(Y_pred == 1 & Y == -1, 1), X(Y_pred == 1 & Y == -1, 2), 'ro', 'LineWidth', 3, 'MarkerSize', 10);

% disable hold gca
hold off


function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

[zero_rows] = find(y==0);
[one_rows] = find(y==1);

plot(X(zero_rows,1),X(zero_rows,2),'bo','MarkerSize', 6, 'MarkerFaceColor', 'b');
plot(X(one_rows,1),X(one_rows,2),'r+','MarkerSize', 9, 'LineWidth', 2);
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('zero results - not admitted', 'one results - admitted', 'location', 'northeastoutside');








% =========================================================================



hold off;

end

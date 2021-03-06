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

pos_1=find(y==1);
pos_0=find(y==0);

plot(X(pos_1,1),X(pos_1,2),'k+','LineWidth',2,'MarkerSize',7)
plot(X(pos_0,1),X(pos_0,2),'ko','MarkerSize',7,'MarkerFaceColor','y')








% =========================================================================



hold off;

end

% temp = zeros(5000, 784);
% for i = 1:5000
%     temp(i, :) = reshape(xTrainImages{i}, [784, 1]);
% end
% %init = rand(10, 784)/10;
% f = temp;
% init = vertcat(f(1,:), f(16,:), f(5,:), f(41,:), f(4,:), f(17,:), f(10,:), f(7,:), f(5,:), f(3,:));
% %init(1, 1) = init(1, 1) + 0.01;
% [centroids, clusterAssignments] = kmeans(temp,10,init);

A = (1+rand(50,10)/10);
B = (-1+rand(50,10)/10);
x = vertcat(A, B);
init = rand(2,10);
[centroids, clusterAssignments] = kmeans_imp(x,2,init);
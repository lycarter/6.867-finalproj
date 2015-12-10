temp = zeros(5000, 784);
for i = 1:5000
    temp(i, :) = reshape(xTrainImages{i}, [784, 1]);
end
%init = rand(10, 784)/10;
f = temp;
%init = vertcat(f(1,:), f(16,:), f(5,:), f(41,:), f(4,:), f(17,:), f(10,:), f(7,:), f(5,:), f(3,:));
%init(1, 1) = init(1, 1) + 0.01;
init = temp(1:10,:);
[centroids, clusterAssignments] = kmeans_imp(temp,10,init);

counts = zeros(10,10);
for i=1:5000
    for j=1:10
        if tTrain(j,i) == 1
            counts(clusterAssignments(i), j) = counts(clusterAssignments(i), j) + 1;
        end
    end
end

clusters = zeros(10,5);
for i=1:10
    [~,indices] = sort(counts(i, :), 'descend');
    clusters(i,:) = indices(1:5);
%     max = -1;
%     argmax = 1;
%     for j=1:10
%         if counts(i, j) > max
%             max = counts(i, j);
%             argmax = j;
%         end
%     end
%     clusters(i) = argmax;
end

testClusterAssignments = zeros(5000,1);
for i = 1:5000
    minDist = 999999999999;
    point = reshape(xTestImages{i}, [1, 784]);
    for j = 1:10
        curDist = norm(point-centroids(j,:))^2;
        if curDist < minDist
            minDist = curDist;
            testClusterAssignments(i) = j;
        end
    end
end

error = 0;
for i=1:5000
    flag = 0;
    for j=1:5
        if tTest(clusters(testClusterAssignments(i),j),i) == 1
            flag = 1;
        end
    end
    if flag == 0
        error = error + 1;
    end
end

% A = (1+rand(50,10)/10);
% B = (-1+rand(50,10)/10);
% x = vertcat(A, B);
% init = rand(2,10);
% [centroids, clusterAssignments] = kmeans_imp(x,2,init);
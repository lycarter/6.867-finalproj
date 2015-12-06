function [centroids, clusterAssignments] = kmeans(X,K,init)
centroids = init;
clusterAssignments = zeros(size(X,1),1);
oldCost = 999999999999;
newCost = 999999999998;
[n,d] = size(X);
z = 0;
while(oldCost > newCost+.001)
    oldCost = newCost;
    z = z + 1;
    %assign clusters
    for i = 1:n
        minDist = 999999999999;
        for j = 1:K
            curDist = square_distance(X(i,:),centroids(j,:));
            if curDist < minDist
                minDist = curDist;
                clusterAssignments(i) = j;
            end
        end
    end
    
    %update centers
    centroids = zeros(K,d);
    centroidCount = zeros(K, 1);
    for i = 1:n
        centroids(clusterAssignments(i),:) = centroids(clusterAssignments(i), :) + X(i,:);
        centroidCount(clusterAssignments(i)) = centroidCount(clusterAssignments(i)) + 1.0;
    end
    for i = 1:K
        centroids(i,:) = centroids(i,:) / centroidCount(i);
    end
    %calculate cost
    newCost = 0;
    for i = 1:n
        newCost = newCost + square_distance(X(i,:),centroids(clusterAssignments(i),:));
    end
    disp(z)
end
end

function [distance] = square_distance(x, y)
    distance = norm(x-y)^2;
end
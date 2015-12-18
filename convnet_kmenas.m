% data = images.data(:,:,:,50001:60000);
% labels = images.labels(50001:60000);
% 
% feat = zeros(10000,64);
% 
% for i = 1:10000
%     temp = vl_simplenn(netfeat,data(:,:,:,i));
%     feat(i,:) = temp(12).x;
% end
% 
% [centroids,clusterAssignments] = kmeans_imp(feat,10,rand(10,64));
clusterAssignments = res;
counts = zeros(10,10);
for i = 1:10000
    for j = 1:10
        if labels(i) == j
            counts(clusterAssignments(i),j) = counts(clusterAssignments(i),j) + 1;
        end
    end
end

corr = zeros(10,2);
for i = 1:10
    corr(i,1) = i;
    [M,Index] = max(counts(i,:));
    corr(i,2) = Index;
end

counts
corr

temp = zeros(10,10000);
temp2 = zeros(10,10000);
for i = 1:10000
    temp(labels(i), i) = 1;
    temp2(corr(clusterAssignments(i),2), i) = 1;
end

a = 1

plotconfusion(temp,temp2)












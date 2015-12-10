% Load the training data into memory
% xTrainImages = tr(:, 1:784);
xTrainImages = num2cell(traindata, 2);
tTrain = trainlabels;

for i = 1:size(xTrainImages)
    r = double(reshape(xTrainImages{i}(1:1024), 32, 32));
    g = double(reshape(xTrainImages{i}(1025:2048), 32, 32));
    b = double(reshape(xTrainImages{i}(2049:3072), 32, 32));
    xTrainImages{i} = (r+g+b)/(3.0*255);
end
tTrainTemp = zeros(size(tTrain,1), 10);

for i = 1:size(tTrain)
    temp = zeros(10,1);
    temp(tTrain(i)+1) = 1;
    tTrainTemp(i, :) = temp;
end

xTestImages = num2cell(testdata, 2);
tTest = testlabels;

for i = 1:size(xTestImages)
    r = double(reshape(xTestImages{i}(1:1024), 32, 32));
    g = double(reshape(xTestImages{i}(1025:2048), 32, 32));
    b = double(reshape(xTestImages{i}(2049:3072), 32, 32));
    xTestImages{i} = (r+g+b)/(3.0*255);
end
tTestTemp = zeros(size(tTest,1), 10);

for i = 1:size(tTest)
    temp = zeros(10,1);
    temp(tTest(i)+1) = 1;
    tTestTemp(i, :) = temp;
end

% 
% tTrain = tTrainTemp;
% Display some of the training images
clf
for i = 1:20
    subplot(4,5,i);
    imshow(xTrainImages{i});
end
rng('default')
hiddenSize1 = 200;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15);
view(autoenc1)
plotWeights(autoenc1);
feat1 = encode(autoenc1,xTrainImages);
result1 = predict(autoenc1, xTrainImages);
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1);
view(autoenc2)
feat2 = encode(autoenc2,feat1);
f = transpose(feat2);
%init = vertcat(f(1,:), f(16,:), f(5,:), f(41,:), f(4,:), f(17,:), f(10,:), f(7,:), f(5,:), f(3,:));
init = f(1:10,:);
[centroids, clusterAssignments] = kmeans_imp(transpose(feat2),10,init);

[idx, C] = kmeans(transpose(feat2),10);

counts = zeros(10,10);
for i=1:size(xTrainImages)
    counts(clusterAssignments(i), labels(i)+1) = counts(clusterAssignments(i), labels(i)+1) + 1;
end

clusters = zeros(10);
for i=1:10
    max = -1;
    argmax = 1;
    for j=1:10
        if counts(i, j) > max
            max = counts(i, j);
            argmax = j;
        end
    end
    clusters(i) = argmax;
end
error = 0;
for i=1:size(xTrainImages)
    if clusters(clusterAssignments(i)) ~= labels(i)+1
        error = error + 1;
    end
end

softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
view(softnet)
view(autoenc1)
view(autoenc2)
view(softnet)
deepnet = stack(autoenc1,autoenc2,softnet);
view(deepnet)
% Get the number of pixels in each image
imageWidth = 32;
imageHeight = 32;
inputSize = imageWidth*imageHeight;

% Load the test images
% xTestImages = te(:, 1:784);
% xTestImages = num2cell(xTestImages, 2);
% tTest = te(:, 785);

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
y = deepnet(xTest);
plotconfusion(tTest,y);
% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);
y = deepnet(xTest);
plotconfusion(tTest,y);
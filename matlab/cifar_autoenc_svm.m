projdir = 'C:\Users\lycarter\Documents\MIT\6.867\';
datadir = strcat(projdir,'6.867-finalproj\matlab\cifar-10-batches-mat\');

% load training data
load(strcat(datadir,'data_batch_1.mat'));
xTrainImages = data;
tTrain = labels;
% load(strcat(datadir,'data_batch_2.mat'));
% xTrainImages = cat(1,xTrainImages,data);
% tTrain = cat(1,tTrain,labels);
% load(strcat(datadir,'data_batch_3.mat'));
% xTrainImages = cat(1,xTrainImages,data);
% tTrain = cat(1,tTrain,labels);

% load validation data
load(strcat(datadir,'data_batch_4.mat'));
xValImages = data;
tVal = labels;
load(strcat(datadir,'data_batch_5.mat'));
xValImages = cat(1,xValImages,data);
tVal = cat(1,tVal,labels);

clear data;
clear labels;
clear batch_label;

% reshape data and transform to grayscale
xTrainImages = reshape(xTrainImages,size(xTrainImages,1),[],3);
xValImages = reshape(xValImages,size(xValImages,1),[],3);
xTrainImages = mean(xTrainImages,3);
xValImages = mean(xValImages,3);

xTrainImages = xTrainImages';
xValImages = xValImages';
tTrain = tTrain';
tVal = tVal';

tTrainVect = zeros(10,size(tTrain,2));
for i = 1:size(tTrain,2)
    tTrainVect(tTrain(i)+1,i) = 1;
end
tValVect = zeros(10,size(tVal,2));
for i = 1:size(tVal,2)
    tValVect(tVal(i)+1,i) = 1;
end

xTrainMatrix = xTrainImages;
xValMatrix = xValImages;

xTrainImages = reshape(xTrainImages,32,32,size(xTrainImages,2));
xValImages = reshape(xValImages,32,32,size(xValImages,2));

xTrainImages = num2cell(xTrainImages,[1 2]);
xValImages = num2cell(xValImages,[1 2]);

hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
'MaxEpochs',100, ...
'L2WeightRegularization',0.004, ...
'SparsityRegularization',4, ...
'SparsityProportion',0.15, ...
'ScaleData', false);

% view(autoenc1)
plotWeights(autoenc1);
feat1 = encode(autoenc1,xTrainImages);

%layer 2
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
'MaxEpochs',100, ...
'L2WeightRegularization',0.002, ...
'SparsityRegularization',4, ...
'SparsityProportion',0.1, ...
'ScaleData', false);

% view(autoenc2);

feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2, tTrainVect, 'maxEpochs', 400);
% view(softnet);

deepnet = stack(autoenc1,autoenc2,softnet);

view(deepnet);

y = deepnet(xValMatrix);


plotconfusion(tValVect,y);

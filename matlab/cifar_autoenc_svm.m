projdir = 'E:\Documents\Landon\MIT\6.867\';
datadir = strcat(projdir,'6.867-finalproj\matlab\cifar-10-batches-mat\');

% load training data
load(strcat(datadir,'data_batch_1.mat'));
xTrainImages = data;
tTrainImages = labels;
load(strcat(datadir,'data_batch_2.mat'));
xTrainImages = cat(1,xTrainImages,data);
tTrain = cat(1,tTrain,labels);
load(strcat(datadir,'data_batch_3.mat'));
xTrainImages = cat(1,xTrainImages,data);
tTrain = cat(1,tTrain,labels);

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

hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
'MaxEpochs',400, ...
'L2WeightRegularization',0.004, ...
'SparsityRegularization',4, ...
'SparsityProportion',0.15, ...
'ScaleData', false);

view(autoenc1)
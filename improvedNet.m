clc;
import nnet.layer.*
digitDatasetPath='D:\discern\dataset';
imds=imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
%% 数据展示
figure('Name','数据集部分图像展示','NumberTitle','off');
numImages=1020;
perm=randperm(numImages,20);
figure(1)
for i=1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
%% 数据集划分
[imgTrain,imgTest]=splitEachLabel(imds,0.8,'randomize');
% 加载ResNet50（ImageNet上的预训练网络）
net = resnet50;
numClasses = 3;
fcLayer = fullyConnectedLayer(numClasses, 'Name', 'fc');
net = replaceLayer(net, 'fc1000', fcLayer);
net = addLayer(net, classificationLayer('Name', 'output'));

% 定义注意力机制
attentionLayer = [
    globalAveragePooling2dLayer('Name', 'global_avg_pool')
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(2048, 'Name', 'fc2')
    sigmoidLayer('Name', 'sigmoid')
    multiplyLayer(2,'Name', 'multiply')
    ];

% 添加注意力层到ResNet50
lgraph = layerGraph(net);
output = net.Layers(end).OutputNames;
lgraph = addLayers(lgraph, attentionLayer);
lgraph = connectLayers(lgraph, output, 'global_avg_pool');

% 连接层
lgraph = connectLayers(lgraph, 'multiply', 'res5a_relu');

% 用SVM代替softmax
lgraph = removeLayers(lgraph, 'ClassificationLayer_predictions');
lgraph = addLayers(lgraph, fullyConnectedLayer(numClasses, 'Name', 'fc_final'));
lgraph = addLayers(lgraph, svmLayer('Name', 'svm'));
lgraph = connectLayers(lgraph, 'avg_pool', 'fc_final');
lgraph = connectLayers(lgraph, 'fc_final', 'svm');

% 设置训练参数
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 20, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 5, ...
    'L2Regularization', 0.0005, ...
    'ExecutionEnvironment','gpu', ...
    'ValidationData', imgTest, ...
    'ValidationFrequency', 50, ...
    'Verbose', false,...
    'Plots', 'training-progress');
% 训练模型
[net,Info] = trainNetwork(imgTrain, lgraph, options);
save("trainedInfo_resnet50.mat","info");






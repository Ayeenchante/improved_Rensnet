%% 识别图像类型
clc;
clear all;
load('newCNNTrainmini(Resnet50).mat','net');

net.Layers;
[file,path]=uigetfile('*');
image=fullfile(path,file);
I=imresize(imread(image),[300,300]);
file

sz=net.Layers(1).InputSize;
I=I(1:sz(1),1:sz(2),1:sz(3));
tic
[label,scores]=classify(net,I);
toc
labelnew=str2num(char(label))+1;
classNames = net.Layers(end).ClassNames
[~,name]=xlsread('labelname.xlsx');

figure('Name','识别结果','NumberTitle','off');
imshow(I);
condition=name(labelnew,1);
C=char(condition);
title(string(C)+"[准确率："+num2str(100*scores(classNames == label))+"%]");
% title(['\bf',condition]);
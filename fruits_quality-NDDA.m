% Copyright 2020 by José Naranjo-Torres (https://www.researchgate.net/profile/Jose_Naranjo_Torres2) and LITRP (www.litrp.cl)
% All rights reserved.
% 
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License GPLv3 as published by
% the Free Software Foundation.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  Please see the
% GNU General Public License for more details: https://www.gnu.org/licenses/gpl-3.0.html
% 
% __author__ = "José Naranjo-Torres"
% __copyright__ = "Copyright (C) 2020 José Naranjo-Torres"
% __license__ = "GNU General Public License GPLv3"
% __version__ = "1.0"

%%
%Directory path of the apple dataset, download from: https://github.com/OlafenwaMoses/AppleDetection/releases/download/v1/apple_detection_dataset.zip
%(Ismail, A.; Idris, M.Y.I.; Ayub, M.N.; Por, L.Y. Investigation of Fusion Features for Apple Classification
%in Smart manufacturing. Symmetry2019,11, 1194.  doi:10.3390/sym11101194.)
appleDataTrainPath = fullfile('C:\','apple_quality_dataset','train','segmented'); % Training Data
appleDataTestPath = fullfile('C:\','apple_quality_dataset','validation','segmented');% Test Data


% The function "imageDatastore" labels the images automatically
% in the folder name and stores the data as an object
appleDataTrain = imageDatastore(appleDataTrainPath,'IncludeSubfolders', ...
    true,'LabelSource','foldernames');
 
appleDataTest = imageDatastore(appleDataTestPath,'IncludeSubfolders', ...
    true,'LabelSource','foldernames');
%%
%%%%%%%%%%%%%%%%%%%%% OPTIONAL %%%%%%%%%%%%%%%%%%%%%
%If the number of images for each category in the training set is not equal,
%it is equal to the minimum number of images in a category randomly.
countLabel = appleDataTrain.countEachLabel;
minSet = min(countLabel{:,2});
appleDataTrain = splitEachLabel(appleDataTrain,minSet,'randomize');
%%
% The images in this database do not have the same dimensions, so they are resized
% and the data augmentation procedure is applied in one step.

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandXReflection',true, ...
    'RandYReflection',true);

imageSize = [32 32 3];
augAppleDataTrain = augmentedImageDatastore(imageSize,appleDataTrain,...
    'DataAugmentation',imageAugmenter);

augAppleDataTest = augmentedImageDatastore(imageSize,appleDataTest);

%%
%%%%%%%%%%%%%%%%%%%%% OPTIONAL %%%%%%%%%%%%%%%%%%%%%
%If the number of images for each category in the training set is not equal,
%it is equal to the minimum number of images in a category randomly.

%  countLabel = appleDataTrain.countEachLabel;
%  minSet = min(countLabel{:,2});
%  appleDataTrain = splitEachLabel(appleDataTrain,minSet,'randomize');

%%
%Parameters settings
filtersNumbers = [8, 16, 32];
filtersSize = [5,5;,4,4;3,3];
poolSize = [2, 2];

%%
% Define the "Architecture of the convolutional neural network".

layers = [imageInputLayer([imageSize(1) imageSize(2) imageSize(3)])
          convolution2dLayer(filtersSize(1,1:2),filtersNumbers(1)) 
          reluLayer
          batchNormalizationLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer(filtersSize(2,1:2),filtersNumbers(2)) 
          reluLayer
          batchNormalizationLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer(filtersSize(3,1:2),filtersNumbers(3)) 
          reluLayer
          batchNormalizationLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(2)
          softmaxLayer
          classificationLayer()];
 %%
% Training options
options = trainingOptions('sgdm',...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.5,...
      'L2Regularization',0.0001,...
      'InitialLearnRate',0.001,... 
      'MaxEpochs',36,... 
      'MiniBatchSize',32, ...
      'Plots','training-progress');
%%
% Training the network
convnet = trainNetwork(augAppleDataTrain,layers,options);
%%
%The trained network is tested with the test data
Pred = classify(convnet,augAppleDataTest);
Validation = appleDataTest.Labels;

%%
%the accuracy is calculated.
accuracy = sum(Pred == Validation)/numel(Validation);
fprintf('Test Acuracy: %4.2f%% \n',accuracy*100) 
figure, plotconfusion(Validation,Pred)

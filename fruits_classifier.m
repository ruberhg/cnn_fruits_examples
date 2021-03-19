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
%Directory path of the fruit dataset, download from: https://github.com/Horea94/Fruit-Images-Dataset  
%(Horea Muresan, Mihai Oltean, Fruit recognition from images using deep learning, Acta Univ. Sapientiae,
%Informatica Vol. 10, Issue 1, pp. 26-42, 2018.)
fruitDataTrainPath = fullfile('C:\','fruits-360_dataset','fruits-360','Training'); % Training Data
fruitDataTestPath = fullfile('C:\','fruits-360_dataset','fruits-360','Test');% Test Data

%Select the categories to study in the example
Categ = {'Apple Golden 1','Apple Pink Lady','Apple Red 1','Pear Red','Pear Williams','Pear Monster'};
numCat = length(Categ);

% The function "imageDatastore" labels the images automatically
% in the folder name and stores the data as an object
fruitDataTrain = imageDatastore(fullfile(fruitDataTrainPath,Categ), ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
 
fruitDataTest = imageDatastore(fullfile(fruitDataTestPath,Categ), ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

%Image dimension (If all the images have the same dimensions) 
imSize = size(readimage(fruitDataTrain,1));
%%
%%%%%%%%%%%%%%%%%%%%% OPTIONAL %%%%%%%%%%%%%%%%%%%%%
%If the number of images for each category in the training set is not equal,
%it is equal to the minimum number of images in a category randomly.
countLabel = fruitDataTrain.countEachLabel;
minSet = min(countLabel{:,2});
fruitDataTrain = splitEachLabel(fruitDataTrain,minSet,'randomize');

%%
%Parameters settings
filtersNumbers = [8, 16, 32];
filtersSize = [5,5;,4,4;3,3];
poolSize = [2, 2];

%%
% Define the "Architecture of the convolutional neural network".
layers = [imageInputLayer([imSize(1) imSize(2) imSize(3)])
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
          fullyConnectedLayer(numCat)
          softmaxLayer
          classificationLayer()];
%%
% Training options
options = trainingOptions('sgdm',...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.5,...
      'L2Regularization',0.0005,...
      'InitialLearnRate',0.001,... 
      'MaxEpochs',10,... 
      'MiniBatchSize',32, ...
      'Plots','training-progress');
%%
% Training the network
classiNet = trainNetwork(fruitDataTrain,layers,options);
%%
%The trained network is tested with the test data
Pred = classify(classiNet,fruitDataTest);
Validation = fruitDataTest.Labels;

%%
% Accuracy
accuracy = sum(Pred == Validation)/numel(Validation);
fprintf('Test Acuracy: %4.2f%% \n',accuracy*100) 
figure, plotconfusion(Validation,Pred)



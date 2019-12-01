clear all; close all;
%%
load red_wine.mat
%load white_wine.mat

% feature standardization (zero-mean and unit variance)
%data(:,1:end-1) = bsxfun(@rdivide,bsxfun(@minus,data(:,1:end-1),mean(data(:,1:end-1),1)),std(data(:,1:end-1),0,1));
x = data(:,1:end-1);
t = data(:,end);
%% ************ create or load training, validation and test indices **********
train_ratio = 0.7;
val_ratio = 0.15;
test_ratio = 1-train_ratio-val_ratio;
no_of_sample = size(data,1);
no_train_sample = round(no_of_sample*train_ratio);
no_val_sample = round(no_of_sample*val_ratio);
no_test_sample = no_of_sample - no_train_sample - no_val_sample;
if exist('train70val15test15\train_index.mat','file') && exist('train70val15test15\val_index.mat','file') && exist('train70val15test15\test_index.mat','file')
    load('train70val15test15\train_index.mat');load('train70val15test15\val_index.mat');load('train70val15test15\test_index.mat');
end
%% Linear Regression

trainX = x([train_index val_index],:);
trainT = t([train_index val_index]);

testX = x(test_index,:);
testT = t(test_index);

w = trainX\trainT;

trainY = round(trainX*w);
testY = round(testX*w);

fprintf('Train set - Percentage Correct Classification   : %f%%\n', 100*mean(trainT==trainY));
fprintf('Train set - Percentage Incorrect Classification : %f%%\n', 100*mean(trainT~=trainY));
fprintf('Test set - Percentage Correct Classification   : %f%%\n', 100*mean(testT==testY));
fprintf('Test set - Percentage Incorrect Classification : %f%%\n', 100*mean(testT~=testY));

testY1K = full(sparse(1:numel(testY),testY,ones(numel(testY),1)));
testT1K = full(sparse(1:numel(testT),testT,ones(numel(testT),1)));

if size(testY1K,2) < size(testT1K,2)
    testY1K = [testY1K zeros(size(testY1K,1),size(testT1K,2)-size(testY1K,2))];
elseif size(testT1K,2) < size(testY1K,2)
    testT1K = [testT1K zeros(size(testT1K,1),size(testY1K,2)-size(testT1K,2))];
end
    
% plotconfusion(testT1K',testY1K')
[c,cm] = confusion(testT1K',testY1K')

fprintf('Final Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Final Percentage Incorrect Classification : %f%%\n', 100*c);
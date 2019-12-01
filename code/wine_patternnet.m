clear all; close all;
%%
load red_wine.mat
%load white_wine.mat

if 0
    figure,
    hist(data(:,end),unique(data(:,end))), xlabel('Sensory preference'),
    ylabel('Frequency (wine samples)'), title('Red wine')
end

% feature standardization (zero-mean and unit variance)
if 0
    data(:,1:end-1) = bsxfun(@rdivide,bsxfun(@minus,data(:,1:end-1),mean(data(:,1:end-1),1)),std(data(:,1:end-1),0,1));
end
x = data(:,1:end-1);
t = full(sparse(1:size(x,1),data(:,end),ones(size(x,1),1)));    % create 1-K labels
%t = [zeros(size(data,1),1) dummyvar(data(:,end)) zeros(size(data,1),2)];
%t = dummyvar(data(:,end));

x = x';
t = t';

k = 5;  % number of CV-folds
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
else
    rand_index = randperm(no_of_sample);
    train_index = rand_index(1:no_train_sample);
    val_index = rand_index(no_train_sample+1 : no_train_sample+no_val_sample);
    test_index = rand_index(no_train_sample+no_val_sample+1:end);
    save('train70val15test15\train_index','train_index');
    save('train70val15test15\val_index','val_index');
    save('train70val15test15\test_index','test_index');
end

if exist('train70val15test15\cv_tr_ind.mat') && exist('train70val15test15\cv_te_ind.mat')   % load cross-validation train and test indices
    load('train70val15test15\cv_tr_ind.mat');   load('train70val15test15\cv_te_ind.mat');
else
    [cv_tr_ind,cv_te_ind] = make_cvsplit(no_train_sample,k,train_index);
    save('train70val15test15\cv_tr_ind','cv_tr_ind');
    save('train70val15test15\cv_te_ind','cv_te_ind');
end
%% ****************** MODEL SELECTION BASED ON CROSS-VALIDATION *************************
setdemorandstream(391418381)    % random seed
hiddenNeurons = 5:25;   % hyper-parameter of NN

for n=1:numel(hiddenNeurons)  % loop over the neurons in the hidden layer
    for foldIdx=1:k         % loop over CV fold index
        
        net = patternnet(hiddenNeurons(n),'trainscg','crossentropy');
        net.layers{1}.transferFcn = 'logsig';
        % net.divideFcn = '';   % no early stopping
        % net.trainParam.epochs = 100;
        net.divideFcn = 'divideind';
        net.trainParam.showWindow = 0;
        % Setup Division of Data for Training, Validation, Testing
        %         net.divideParam.trainRatio = 70/100;
        %         net.divideParam.valRatio = 15/100;
        %         net.divideParam.testRatio = 15/100;
        
        
        [net.divideParam.trainInd,net.divideParam.valInd,net.divideParam.testInd] = ...
            divideind(no_of_sample,cv_tr_ind{foldIdx},val_index,cv_te_ind{foldIdx});
        
        %         view(net)
        
        [net,tr] = train(net,x,t);
        nntraintool
        
        testX = x(:,tr.testInd);
        testT = t(:,tr.testInd);
        
        testY = net(testX);
        testIndices = vec2ind(testY);
        perf = perform(net,testT,testY);
        
        
        % plotconfusion(testT,testY)
        
        [c,cm] = confusion(testT,testY)
        
        fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
        fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
        
        testCC(n,foldIdx) = 100*(1-c);
        fprintf('neuron=%d/%d, foldIdx=%d/%d\n',n,numel(hiddenNeurons),foldIdx,k);
    end
end

[maxCC,ind] = max(mean(testCC,2));
best_hiddenneuronsize = hiddenNeurons(ind);
%% *********** Train & Test the NN model with best model parameter found with CV *********
net = patternnet(best_hiddenneuronsize,'trainscg','crossentropy');
net.layers{1}.transferFcn = 'tansig';
net.divideFcn = 'divideind';
net.divideParam.trainInd = train_index;
net.divideParam.valInd = val_index;
net.divideParam.testInd = test_index;
[net,tr] = train(net,x,t);
nntraintool

testX = x(:,tr.testInd);
testT = t(:,tr.testInd);

testY = net(testX);
testIndices = vec2ind(testY);
perf = perform(net,testT,testY);

% plotconfusion(testT,testY)
[c,cm] = confusion(testT,testY)

fprintf('Test set - Final Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Test set - Final Percentage Incorrect Classification : %f%%\n', 100*c);

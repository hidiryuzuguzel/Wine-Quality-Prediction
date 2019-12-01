%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%          ASE5016 - PROJECT      %%%%%%%%%%%%%
%%%%%%%%         Hidir Yuzuguzel, 244904                 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;
%%
load red_wine.mat

% feature standardization (zero-mean and unit variance)
data(:,1:end-1) = bsxfun(@rdivide,bsxfun(@minus,data(:,1:end-1),mean(data(:,1:end-1),1)),std(data(:,1:end-1),0,1));
data_x = data(:,1:end-1);
data_y = [zeros(size(data_x,1),1) dummyvar(data(:,end)) zeros(size(data_x,1),2)];

% ******** partition the whole dataset into training and test data sets ********
no_of_samples = size(data,1);   % Number of samples

hist(data(:,end),unique(data(:,end))),
xlabel('Sensory preference'), ylabel('Frequency (wine samples)')
title('Red wine histogram')

shuffle = randperm(no_of_samples);
train_ratio = 0.7;
test_ratio = 1 - train_ratio;
no_of_train_samples = round(train_ratio*no_of_samples);
no_of_test_samples = no_of_samples - no_of_train_samples;

trainx = data(shuffle(1:no_of_train_samples),1:end-1);
trainy = data(shuffle(1:no_of_train_samples),end);

testx = data(shuffle(no_of_train_samples+1:end),1:end-1);
testy = data(shuffle(no_of_train_samples+1:end),end);

% ****** training ****************
[trainyhat,net] = train_mlp(trainx',trainy',20);
% ******* test **************
[testyhat, ~] = forward(net.W1, net.B1, net.W2, net.B2, testx', net.parameters);

out_train = round(trainyhat);
out_test = round(testyhat);
% Miss classification rate (MCR)
fprintf('MCR for trainmlp.mat train set=%f, test set=%f\n',mean(trainy'~=out_train), mean(testy'~=out_test))
% ****** plot data ************
figure
subplot(211)
plot(1:no_of_train_samples,trainy,'ro',1:no_of_train_samples,out_train,'bx'), title('Training'), legend('Actual','Predicted')
subplot(212)
plot(1:no_of_test_samples,testy,'ro',1:no_of_test_samples,out_test,'bx'), title('Test'), legend('Actual','Predicted')






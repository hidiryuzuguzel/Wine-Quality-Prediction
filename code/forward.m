function [out, p] = forward(W1, B1, W2, B2, data, parameters)
% FORWARD forward propogation phase in MLP training
% parameters: necessary params., such as activation func. to use
% at the hidden layers and output layers
% out: 1-by-N row vector containing f(x) for each of the N inputs
% p: m-by-N matrix of outputs from the hidden layer of MLP

p = parameters.hiddenactfun(bsxfun(@plus,W1*data,B1));
out = parameters.outactfun(W2*p+B2);

end


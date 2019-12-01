function [y,net] = train_mlp(data, target, m)

n = size(data,1);   % input dimension
[ W1, B1, W2, B2 ] = initiate( m, n, 0.1 );
w_old = [W1(:);B1(:);W2(:);B2(:)];
w_new = 999*ones(size(w_old));
lambda = 10^-2;
MAX_ITER = 1000;
I = eye(numel(w_old));
tol = 10^-6;
% ****** activation funcs. & derivatives **********
sigactfun = @(x) logsig(x);
dsigactfun = @(x) (1-logsig(x)).*logsig(x);
tanhactfun = @(x) tanh(x);
dtanhactfun = @(x) 1 - tanh(x).^2;
linactfun = @(x) x;
dlinactfun = @(x) 1;
reluactfun = @(x) double((x>0).*x);
dreluactfun = @(x) double(x>0)
% *************************************************
parameters.hiddenactfun = reluactfun;
parameters.dhiddenactfun = dreluactfun;

parameters.outactfun = linactfun;
parameters.doutactfun = dlinactfun;

turn_GN = true;

k = 1;
while k <= MAX_ITER
   
    if turn_GN
        if k == 1
            w = w_old;
        else
            w = w_new;
        end
        [J,E] = backward(w, m, data, target, parameters);
        fprintf('Iteration=%d, SSQE=%f\n',k,sum(E.^2))
        plot(k,norm(E),'d'), xlabel('Iteration index k'), ylabel('SSQE'), hold on, drawnow
    end
    dw = (J'*J + lambda*I)\(J'*E);
    w_tmp = w + dw;    % compute cost with w_tmp
    [~,Etmp] = backward(w_tmp, m, data, target, parameters);
    
    if norm(Etmp) < norm(E)     % cost decrease (Gauss-Newton)
        w_old = w;
        w_new = w_tmp;
        lambda = lambda/10;
        turn_GN = true;
        k = k + 1;
    else                        % no cost decrease (SD)
        lambda = 10*lambda;
        turn_GN = false;
    end
    
    %fprintf('norm(J*E)=%f, norm(w_next-w_prev) =%f\n',norm(J'*E),norm(w_new-w_old));
    if norm(J'*E) < tol || norm(w_new-w_old) < tol    % termination
        break;
    end
    
    % inspect how training is proceeding
%     [W1,B1,W2,B2] = isolate(w, m, n);
%     [out,~] = forward(W1, B1, W2, B2, data, parameters);
%     plot(1:numel(target),target,'o',1:numel(target),out,'x'); drawnow;
    
    
            
end

[W1,B1,W2,B2] = isolate(w_new, m, n);
[y,~] = forward(W1, B1, W2, B2, data, parameters);
net = struct('W1',W1,'B1',B1,'W2',W2,'B2',B2,'parameters',parameters);

end


function [ W1, B1, W2, B2 ] = initiate( m, n, gain )
%INITIATE initializes the network parameters for a given number
% of hidden neurons m and input dimensionality n. Draw the params.
% from a normal distribution, where gain corresponds to the 
% standard deviation, e.g. gain*randn(10,5)

W1 = gain*randn(m,n);   % input weights
B1 = gain*randn(m,1);   % input bias
W2 = gain*randn(1,m);   % output weights
B2 = gain*randn;        % output bias

end

function [W1,B1,W2,B2] = isolate(w, m, n)
% ISOLATE seperates the NN parameters from a combined vector
% defined through w = [W1(:);B1(:),W2(:),B2(:)]

W1 = reshape(w(1:m*n),m,n);
B1 = reshape(w(m*n+1:m*n+m),m,1);
W2 = reshape(w(m*n+m+1:m*n+2*m),1,m);
B2 = w(end);

end


function [J,E] = backward(w, m, data, target, parameters)

n = size(data,1);
[W1,B1,W2,B2] = isolate(w, m, n);
[p2,p1] = forward(W1, B1, W2, B2, data, parameters);

E = target - p2;    % error
z2 = ones(size(E));
%z2 = E;

ah_in = bsxfun(@plus,W1*data,B1);
ah_out = W2*parameters.hiddenactfun(ah_in)+B2;

d2 = z2 .* parameters.doutactfun(ah_out);
z1 = W2'*d2;    % m-by-N
d1 = z1 .* parameters.dhiddenactfun(ah_in);


% compute gradient
J = zeros(numel(target),numel(w));  % init. Jacobian matrix (N-by-L)


for i=1:n   % loop over input dimension
     J(:,(i-1)*m+1:i*m) =  (bsxfun(@times,d1,data(i,:)))';
end

J(:,m*n+1:m*n+m) = d1';

J(:,m*n+m+1:m*n+2*m) = (bsxfun(@times,d2,p1))';

J(:,end) = d2';

E = E';
end



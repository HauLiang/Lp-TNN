function Lp_TNN_recon = Lp_TNN(mask_image, mask, p, r, gamma, mu, rho, maxIter, tol)
%
% This code implements the Lp-TNN algorithm 
%
% Inputs:
%    mask_image:  sampled image
%    mask:  sampled set
%    p:     p-value of lp-norm
%    r:     truncated parameter of truncated nuclear norm
%    gamma: regularization parameter
%    mu:    penalty parameter
%    rho:   step constant
%    maxIter:  the maximum allowable iterations
%    tol:     tolerance of convergence criterion
% Outputs:
%    Lp_TNN_recon:  recovered image, obtained by Lp-TNN
%
% Authors: Hao Liang (haoliang@stu.xmu.edu.cn) 
% Last modified by: 22/03/10
%
%

% Initialization
[nx,ny] = size(mask_image); X = mask_image.*mask; 
idx = find(mask==1);     % sampled matrix index
idx1 = find(mask~=1);    % unsampled matrix index
Sigma = zeros(nx,ny); Lamda = zeros(nx,ny); 
E = zeros(nx,ny); W = mask_image; XMed = zeros(nx,ny);

for i = 1:maxIter   
    
    % Update X^{(k)}
    M = E+mask_image+Lamda/mu; N = W-Sigma/mu; 
    XMed(idx) = (N(idx)+M(idx))/2; XMed(idx1) = N(idx1);
    
    % Update X^{(k+1)}
    [U,S,V] = svd(XMed,'econ');
    S = sign(S).*max(abs(S)-1/mu,0);
    Xtemp = X; X = U*S*V';
  
    % Update E_{\Omega}^{(k+1)}
    H = X-mask_image-Lamda/mu;
    for j = 1:length(idx)
        E(idx(j)) = solve_subproblem_17(H(idx(j)), 2*gamma/mu, p);
    end
    
    % Update A^{(k+1)} and B^{(k+1)}
    [U1,~,V1] = svd(W);
    A = (U1(:,1:r)).';
    B = (V1(:,1:r )).';
    
    % Update W^{(k+1)}
    Wtemp = W;
    W = X+(1/mu)*Sigma+(1/mu)*(A.'*B);
    W(idx) = Wtemp(idx);
    
    % Update Lamda^{(k+1)}
    Lamda = Lamda + mu*(E-X+mask_image);
    
    % Update ¡Æ^{(k+1)}
    Sigma = Sigma + mu*(X-W);
    
    % Update ¦Ì
    mu = mu*rho;
    
    % Stopping criteria
    TOLL = norm(X-Xtemp,'fro')/max(norm(X,'fro'),1);
    if TOLL<=tol
        break;
    end
    
end

Lp_TNN_recon = X;

end


function x = solve_subproblem_17(alpha, lambda, p)

% Solving the subproblem (17), i.e.,
%    min_{x}  (x-alpha)^2 + lambda*|x|^p

a = (lambda*p*(1-p)/2)^(1/(2-p))+eps;
b = 2*a-2*alpha+lambda*p*a^(p-1);
if b < 0
    x = 2*a;
    for i = 1:10
        f = 2*x-2*alpha+lambda*p*x^(p-1);
        g = 2+lambda*p*(p-1)*x^(p-2);
        x = x-f/g;
    end
    sigma1 = x;
    ob1 = x^2-2*x*alpha+lambda*abs(x)^p;
else
    sigma1 = 1;ob1=inf;
end

a = -a;
b = 2*a-2*alpha+lambda*p*abs(a)^(p-1);
if b > 0
    x = 2*a;
    for i = 1:10
        f = 2*x-2*alpha-lambda*p*abs(x)^(p-1);
        g = 2+lambda*p*(p-1)*abs(x)^(p-2);
        x = x-f/g;
    end
    sigma2 = x;
    ob2 = x^2-2*x*alpha+lambda*abs(x)^p;
else
    sigma2 = 1;ob2=inf;
end
sigma_can = [0,sigma1,sigma2];
[~,idx] = min([0,ob1,ob2]);
x = sigma_can(idx);

end
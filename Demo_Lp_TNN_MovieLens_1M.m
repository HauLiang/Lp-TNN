% ------------------ MovieLens-1M Demo for Lp-TNN -------------------------
% 
% This is a simple example to test the Lp-TNN algorithm in real-world
% dataset -- MovieLens-1M Dataset
%
% -- A Robust Low-Rank Matrix Completion Based on Truncated Nuclear Norm and Lp-norm
%
% This code is based on the FGSR code available from:
%    https://github.com/ImKeTT/Low-rank_Matrix-completion/blob/master/FGSR-for-effecient-low-rank-matrix-recovery/test_MC_movielens1M.m 
%
% from the following paper:
%    [1] Fan, Jicong and Ding, Lijun and Chen, Yudong and Udell, Madeleine. 
%    Factor group-sparse regularization for efficient low-rank matrix recovery.
%    Advances in Neural Information Processing Systems, 2019.
%
% Please check the accompanying license and the license of [1] before using. 
%
% Author: Hao Liang 
% Last modified by: 22/03/11
%

clc; clear; close all;

%% Pre-processing, please refer to [1]

load('MovieLens1M.mat');    % load data

% Pre-process
M_org = double(X~=0);
maxdim = sum(M_org(:))/prod(size(M_org))*min(size(M_org));
sm = sum(M_org'); idsm = find(sm>=5);
M_org = M_org(idsm,:);
X = X(idsm,:); X(M_org==0) = mean(X(M_org==1));
[nr,nc] = size(X); M_t = ones(nr,nc);

% Random mask
missrate = 0.5;    % missing rate 
for i=1:nc
    idx = find(M_org(:,i)==1);
    lidx = length(idx);
    temp = randperm(lidx,ceil(lidx*missrate));
    temp = idx(temp);
    M_t(temp,i) = 0;
end
M = M_t.*M_org; Xm = X.*M; [nx, ny] = size(Xm); 

% Initialization indicator
NMAE = zeros(1,6); RMSE = NMAE; 

disp('Data finish!')

% Parameter setting
maxIter = 100; tol = 1e-2;


%% SVT
tao = sqrt(nx*ny); step = 1.2*missrate;  

tic;
[Xr{1}] = SVT(Xm, M, tao, step, maxIter, tol);
t1 = toc;

disp('SVT finish!')

%% SVP
step = 1/missrate/sqrt(maxIter); k = 26;
tic;
[Xr{2}] = SVP(Xm, M, step, k, maxIter, tol);
t2 = toc;

disp('SVP finish!')

%% TNNR
beta = 1; rank_r = 20;
tic;
[Xr{3}] = TNNR_ADMM(Xm, M, beta, rank_r, maxIter, tol);
t3 = toc;

disp('TNNR finish!')

%% Sp-lp
gamma = 1; p1 = 1; p2 = 1;

tic;
[Xr{4}] = Sp_lp_new(Xm, M, gamma, p1, p2, maxIter, tol);
t4 = toc;

disp('Sp-lp finish!')


%% FGSR 
options.d = 20;
options.regul_B = 'L21';
options.maxiter = maxIter;
options.lambda = 0.007;
options.tol = tol;

tic;
[Xr{5}] = MC_FGSR_PALM(Xm, M, options);
t5 = toc;

disp('FGSR finish!')

%% Lp-TNN
p = 0.2; r = 10; gamma = 1.1; mu = 0.001; rho = 1.3; 

tic;
[Xr{6}] = Lp_TNN(Xm, M, p, r, gamma, mu, rho, maxIter, tol);
t6 = toc;

disp('Lp-TNN finish!')


%% Record the NMAEs and RMSEs
for i = 1:length(Xr)
    if ~isempty(Xr{i})
        MM = (~M).*M_org;
        E = (Xr{i}-X).*MM;
        NMAE(i) = sum(abs(E(:)))/sum(MM(:))/4;
        RMSE(i) = norm(E,'fro')/norm(X.*MM,'fro');
    end
end
Time = [t1,t2,t3,t4,t5,t6];

NMAE
RMSE
Time


% ------------------------ Demo for Lp-TNN --------------------------------
% 
% This is a simple example to test the Lp-TNN algorithm 
% -- A Robust Low-Rank Matrix Completion Based on Truncated Nuclear Norm and Lp-norm
%
% Author: Hao Liang 
% Last modified by: 22/03/10
%


%% Experiment setup
clc; clear; close all;

% Load RGB figure
image = imread('demo_figure.jpg');

% Normalize R, G, and B channels, respectively
image_double = double(image);
img_R = image_double(:,:,1); img_G = image_double(:,:,2); img_B = image_double(:,:,3);
xm_R = min(img_R(:)); Io_R = img_R-xm_R; img_R = Io_R/max(Io_R(:));
xm_G = min(img_G(:)); Io_G = img_G-xm_G; img_G = Io_G/max(Io_G(:)); 
xm_B = min(img_B(:)); Io_B = img_B-xm_B; img_B = Io_B/max(Io_B(:));

% Parameters setting
[nx,ny,~] = size(image_double);

% Random mask
mask = zeros(nx,ny); samp_rate = 0.4;  % sampling rate 
chosen = randperm(nx*ny,round(samp_rate*nx*ny)); mask(chosen) = 1 ;

% Masked image
mask_R = img_R.*mask; mask_G = img_G.*mask; mask_B = img_B.*mask;
mask_image = cat(3,mask_R,mask_G,mask_B);


%% Lp-TNN
p = 0.2; r = 0; gamma = 1.1; mu = 0.1; rho = 1.3; maxIter = 1000; tol = 1e-4;
Lp_TNN_recon_R = Lp_TNN(mask_R, mask, p, r, gamma, mu, rho, maxIter, tol);
Lp_TNN_recon_G = Lp_TNN(mask_G, mask, p, r, gamma, mu, rho, maxIter, tol);
Lp_TNN_recon_B = Lp_TNN(mask_B, mask, p, r, gamma, mu, rho, maxIter, tol);
Lp_TNN_image = cat(3,Lp_TNN_recon_R,Lp_TNN_recon_G,Lp_TNN_recon_B);


%% Experimental results
figure; imshow(image,[]); title('Original image','FontSize',15,'FontName','Times New Roman'); 
figure; imshow(mask_image,[]); title(['Masked image (sampling rate = ', num2str(samp_rate),')'],'FontSize',15,'FontName','Times New Roman'); 
figure; imshow(Lp_TNN_image,[]); title('Recovered image by Lp-TNN','FontSize',15,'FontName','Times New Roman'); 

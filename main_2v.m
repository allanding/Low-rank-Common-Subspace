clc
clear all
close all
%% load 2-view data
load 2view.mat
K = 2;

%% test data of two views
%% Xs1 and Xt1 are the training and testing data for view 1
%% Xs2 and Xt2 are the training and testing data for view 2

Xt = [Xt1;Xt2]';
Yt = [Yt1;Yt2];

%% training data of two views
Xs = [Xs1;Xs2]';
Ys = [Ys1;Ys2];

%% Stack to Acehive Big Matrix Xs and Xt
s1 = size(Xs1,2);
s2 = size(Xs2,2);
t2 = size(Xs2,2);
[n,t1] = size(Xs1); %% high dim
Xss = [Xs1, zeros(size(Xs2));
    zeros(size(Xs1)), Xs2];

Xtt = [Xs1, Xs2];
Xtt = [Xtt;Xtt];
s = s1+s2;
t = t1+t2;
d = 200;

%% call low-rank common subspace function
P = LRCS(Xtt,Xss,t,s,n,K,d);

%% Calculate the recognition rate
Zs = P'*Xs;
Zt = P'*Xt;
Cls = knnclassify(Zt',Zs',Ys,1);
acc = length(find(Cls==Yt))/length(Yt);
fprintf('Results+NN=%0.4f\n',acc);


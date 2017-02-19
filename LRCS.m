function Pn = LRCS(Xt,Xs,t,s,n,K,d)

%% |Z|_* + |Pt|_*+lambda1|E|_2,1+lambda2|Est|_1, s.t. Ps'*Xs = Pt'*Xt*Z+E, Ps = Pt + Est
%% for the detail solution, please refer to our ICDM 14 paper
%% @inproceedings{ding2014low,
%%   title={Low-Rank Common Subspace for Multi-view Learning},
%%   author={Ding, Zhengming and Fu, Yun},
%%   booktitle={2014 IEEE International Conference on Data Mining (ICDM)},
%%   pages={110--119},
%%   year={2014},
%%   organization={IEEE}
%% }

Xt = Xt/K;

%% initialize Ps and Pt

options.ReducedDim = d;
Ps = PCA(Xs',options);
Pt = Ps;

%% initialize others
Qt = zeros(size(Pt));
Z = zeros(t,s);
J = zeros(t,s);
E = zeros(d,s);
Est = zeros(K*n,d);

%% laglange multipliers
Y1 = zeros(d,s);
Y2 = zeros(K*n,d);
Y3 = zeros(t,s);
Y4 = zeros(K*n,d);

%% parameters initialization
lambda1 = 1e-1; %% error term ||E||_{2,1}
lambda2 = 1e-2; %% error term ||Est||_1
maxiter = 200; %% maximum iteration
max_mu = 1e6;
rho = 1.2;
mu = 1e-5;
tol = 1e-6;
warning off
for iter = 1:maxiter
    disp(iter)
    %% update J
    temp = Z + Y3/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    %% update Z
    Z1 = Xt'*Pt*Pt'*Xt + eye(t);
    Z2 = Xt'*Pt*(Ps'*Xs-E)+J+(Xt'*Pt*Y1-Y3)/mu;
    Z = Z1\Z2;
    
    %% update Qt
    temp = Pt + Y4/mu;
    [Qu,Qs,Qv] = svd(temp,'econ');
    Qs = diag(Qs);
    svp = length(find(Qs>1/mu));
    if svp>=1
        Qs = Qs(1:svp)-1/mu;
    else
        svp = 1;
        Qs = 0;
    end
    Qt = Qu(:,1:svp)*diag(Qs)*Qv(:,1:svp)';
    
    if iter>1
        %% update Ps
        Ps1 = Xs*Xs' + eye(K*n);
        Ps2 = Xs*(Pt'*Xt*Z-E)'+Pt+Est - (Xs*Y1'+Y2)/mu;
        Ps = Ps1\Ps2;
        
        %%update Pt
        Pt1 = Xt*Z*Z'*Xt' + 2*eye(K*n);
        Pt2 = Xt*Z*(Ps'*Xs-E)'+ Ps -Est + Qt + (Xt*Z*Y1'+Y2-Y4)/mu;
        Pt = Pt1\Pt2;
        Pt = sqrt(K)*orth(Pt/sqrt(K));
        
        %% update Est
        temp = Ps - Pt+Y2/mu;
        Est = max(0,temp - lambda2/mu)+min(0,temp + lambda2/mu);
    end
    
    %% update E
    temp = Ps'*Xs -Pt'*Xt*Z+Y1/mu;
    E = solve_l1l2(temp,lambda1/mu);
    
    %% update the multiplies
    leq1 = Ps'*Xs -Pt'*Xt*Z -E;
    leq2 = Ps-Pt-Est;
    leq3 = Z-J;
    leq4 = Pt-Qt;
    
    %% check convergence
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max(max(max(abs(leq3))),stopC);
    stopC = max(max(max(abs(leq4))),stopC);
    
    disp(stopC)
    if stopC<tol
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        Y4 = Y4 + mu*leq4;
        mu = min(max_mu,mu*rho);
    end
end

%% decompose the stacked projection into the small one
Pn = zeros(n,d);
for ii=0:K-1
    Pn = Pn+Pt(ii*n+1:(ii+1)*n,:);
end

end
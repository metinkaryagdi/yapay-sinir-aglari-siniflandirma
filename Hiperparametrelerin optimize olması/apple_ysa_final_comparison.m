function apple_ysa_final_comparison()
%% =========================================================================
%  Apple Quality - Final Consolidated Comparison
%  
%  Bu script:
%  1. 'results_phase4_final.csv' dosyasını okur.
%  2. Her metodun (ABC, BFGS, CG, DFP, GD) en yüksek ValAcc skorlu
%     "Şampiyon" modelini seçer.
%  3. Bu modelleri sıfırdan oluşturur ve eğitir (From Scratch Kütüphane).
%  4. Sonuçları detaylı grafiklerle karşılaştırır.
%
%  Grafikler:
%    - Fig 1: Her şampiyonun tekil Eğitim/Val/Test Cost geçmişi (Subplot Matrix).
%    - Fig 2: Tüm metodların Validation Cost düşüşlerinin tek grafikte kıyaslanması.
%    - Fig 3: Final Test Doğruluklarının Bar Grafiği.
% ==========================================================================

    clear; clc; close all;
    rng(42); % Tekrarlanabilirlik şart

    %% 1. AYARLAR & VERİ YÜKLEME
    CSV_FILE = 'results_phase4_final.csv';
    DATA_FILE= 'apple_quality.csv';
    
    if ~exist(CSV_FILE,'file') || ~exist(DATA_FILE,'file')
        error('Gerekli dosyalar (csv) bulunamadı.');
    end

    fprintf('>> Veri yükleniyor...\n');
    [X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = load_and_prep_data(DATA_FILE);
    
    %% 2. ŞAMPİYONLARI BULMA
    fprintf('>> En iyi modeller analiz ediliyor...\n');
    opts = detectImportOptions(CSV_FILE);
    opts.VariableTypes = repmat({'string'},1,numel(opts.VariableNames)); % Her şeyi string al, manuel çevir
    rawParams = readtable(CSV_FILE, opts);
    
    methods = {'ABC','BFGS','CG','DFP','GD'};
    % Şablon Struct (Tüm alanlar önceden tanımlı olmalı)
    TEMPLATE = struct('method','','activation','','layers',[],'lambda',NaN,'stepSize',NaN, ...
                      'SN',NaN,'limit',NaN,'maxCycle',NaN,'maxIter',NaN);
    champions = repmat(TEMPLATE, 0, 1);
    
    for m = methods
        mName = m{1};
        % Metoda ait satırları bul
        idx = find(strcmpi(rawParams.method, mName));
        if isempty(idx), continue; end
        
        sub = rawParams(idx,:);
        % ValAcc parse et
        vAcc = double(sub.valAcc);
        [bestVal, bLoc] = max(vAcc);
        
        bestRow = sub(bLoc,:);
        
        % Struct'a çevir
        champ = TEMPLATE; % Şablondan kopyala
        champ.method = char(bestRow.method);
        champ.activation = char(bestRow.activation);
        
        % Layers parse ( "[32, 16]" -> [32, 16] )
        if ismember('layers_1', sub.Properties.VariableNames)
            l1 = str2double(bestRow.layers_1);
            l2 = str2double(bestRow.layers_2);
            if isnan(l2), champ.layers = l1; else, champ.layers = [l1, l2]; end
        else
            champ.layers = [32, 16]; 
        end
        
        champ.lambda = str2double(bestRow.lambda);
        ss = str2double(bestRow.stepSize);
        if isnan(ss), ss = 0.05; end 
        champ.stepSize = ss; 
        
        % ABC ozel (Kullanıcı bildirimi üzerine Phase-3'ten en iyi parametreler manuel girildi)
        if strcmpi(mName, 'ABC')
            champ.layers = [8];      % Phase 3'teki en iyi mimari
            champ.lambda = 0.001;    % Phase 3'teki en iyi lambda
            champ.SN = 50;           % Popülasyonu biraz daha artırdık
            champ.limit = 100;
            champ.maxCycle = 1000;   % İterasyon sayısı yüksek tutuldu
        else
            champ.maxIter = 300; 
        end
        
        champions = [champions; champ];
        fprintf('   ADAY: %s | %s | Lam:%.4f | ValAcc: %.4f\n', champ.method, champ.activation, champ.lambda, bestVal);
    end
    
    %% 3. EĞİTİM DÖNGÜSÜ
    results = [];
    
    useGPU = false;
    try, if gpuDeviceCount>0, gpuDevice; useGPU=true; end; catch, end
    
    if useGPU
        Xtr=gpuArray(single(X_train)); Ttr=gpuArray(single(T_train));
        Xva=gpuArray(single(X_val));   Tva=gpuArray(single(T_val));
        Xte=gpuArray(single(X_test));  Tte=gpuArray(single(T_test));
    else
        Xtr=X_train; Ttr=T_train; Xva=X_val; Tva=T_val; Xte=X_test; Tte=T_test;
    end
    
    for i=1:numel(champions)
        job = champions(i);
        fprintf('\n>> Eğitim Başlıyor: %s (%s)...\n', job.method, job.activation);
        
        net.inputDim = inputDim;
        net.outputDim = outputDim;
        net.layers = job.layers;
        net.numHidden = numel(job.layers);
        net.act = job.activation;
        
        % Init Weights
        theta = init_weights(net);
        if useGPU, theta = gpuArray(theta); end
        
        % Train
        tic;
        switch job.method
            case 'ABC'
                % ABC is slow, reduced cycle for comparison demo
                [theta, hist] = train_abc(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.SN, job.limit, job.maxCycle);
            case 'BFGS'
                [theta, hist] = train_bfgs(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.stepSize, job.maxIter);
            case 'DFP'
                [theta, hist] = train_dfp(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.stepSize, job.maxIter);
            case 'CG'
                [theta, hist] = train_cg(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.stepSize, job.maxIter);
            case 'GD'
                [theta, hist] = train_gd(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.stepSize, job.maxIter);
        end
        tDur = toc;
        
        % Final Eval
        [teAcc, ~] = evaluate_net(theta, net, Xte, Tte);
        
        res.job = job;
        res.hist = hist;
        res.time = tDur;
        res.finalTestAcc = double(teAcc);
        
        results = [results; res];
        fprintf('   Bitti (%.1fs). Test Acc: %.2f%%\n', tDur, res.finalTestAcc*100);
    end
    
    %% 4. GÖRSELLEŞTİRME
    plot_results(results);
    
end

%% =========================================================================
%%                        DATA HELPER
%% =========================================================================
function [X_train, T_train, X_val, T_val, X_test, T_test, inD, outD] = load_and_prep_data(fname)
    data = readtable(fname);
    q = data.(data.Properties.VariableNames{strcmpi(data.Properties.VariableNames,'Quality')});
    y_bin = double(string(q)=="good");
    isNum = varfun(@isnumeric, data, 'OutputFormat','uniform');
    X = table2array(data(:, isNum & ~strcmpi(data.Properties.VariableNames,'Quality')));
    bad = any(~isfinite(X),2) | ~isfinite(y_bin);
    X(bad,:)=[]; y_bin(bad,:)=[];
    T = [y_bin==0, y_bin==1];
    
    % Split
    N = size(X,1);
    p = randperm(N);
    nTr=round(0.7*N); nVa=round(0.15*N);
    
    idxTr = p(1:nTr);
    idxVa = p(nTr+1:nTr+nVa);
    idxTe = p(nTr+nVa+1:end);
    
    X_train=X(idxTr,:); T_train=T(idxTr,:);
    X_val=X(idxVa,:);   T_val=T(idxVa,:);
    X_test=X(idxTe,:);  T_test=T(idxTe,:);
    
    mu=mean(X_train,1); sig=std(X_train,0,1)+1e-8;
    X_train=(X_train-mu)./sig; X_val=(X_val-mu)./sig; X_test=(X_test-mu)./sig;
    
    inD = size(X,2); outD = size(T,2);
end

%% =========================================================================
%%                        CORE ENGINE (FROM SCRATCH)
%% =========================================================================
function theta = init_weights(net)
    cols={}; prev=net.inputDim;
    for i=1:net.numHidden
        h=net.layers(i);
        lim=sqrt(6/(prev+h));
        cols{end+1}=(rand(h,prev)*2-1)*lim;
        cols{end+1}=zeros(h,1);
        prev=h;
    end
    lim=sqrt(6/(prev+net.outputDim));
    cols{end+1}=(rand(net.outputDim,prev)*2-1)*lim;
    cols{end+1}=zeros(net.outputDim,1);
    for k=1:numel(cols), cols{k}=cols{k}(:); end
    theta=vertcat(cols{:});
end

function [As, Ws, cost] = forward(theta, net, X, T, lam)
    idx=1; Ws={}; bs={}; prev=net.inputDim;
    for i=1:net.numHidden
        h=net.layers(i);
        Ws{end+1}=reshape(theta(idx:idx+h*prev-1),[h,prev]); idx=idx+h*prev;
        bs{end+1}=reshape(theta(idx:idx+h-1),[h,1]); idx=idx+h;
        prev=h;
    end
    Ws{end+1}=reshape(theta(idx:idx+net.outputDim*prev-1),[net.outputDim,prev]); idx=idx+net.outputDim*prev;
    bs{end+1}=reshape(theta(idx:idx+net.outputDim-1),[net.outputDim,1]);
    
    As={X}; A=X;
    for i=1:net.numHidden
        Z = A*Ws{i}.' + bs{i}.';
        switch lower(net.act)
            case 'relu', A=max(0,Z);
            case 'tanh', A=tanh(Z);
            case 'sigmoid', A=1./(1+exp(-Z));
        end
        As{end+1}=A;
    end
    Z = A*Ws{end}.' + bs{end}.'; 
    Z = Z-max(Z,[],2);
    Y = exp(Z)./sum(exp(Z),2);
    As{end+1}=Y;
    
    if isempty(T), cost=0; return; end
    cost = -mean(sum(T.*log(Y+1e-10),2));
    reg=0; for k=1:numel(Ws), reg=reg+sum(Ws{k}(:).^2); end
    cost=cost + 0.5*lam*reg;
end

function [c, g] = cost_grad(theta, net, X, T, lam)
    [As, Ws, c] = forward(theta, net, X, T, lam);
    N=size(X,1); Y=As{end};
    dZ = (Y-T)/N;
    
    grads=cell(1,numel(Ws)*2); gid=numel(grads);
    
    % Output layer
    gW = dZ.' * As{end-1}; 
    gb = sum(dZ,1).';
    if lam>0, gW=gW+lam*Ws{end}; end
    grads{gid}=gb(:); gid=gid-1; grads{gid}=gW(:); gid=gid-1;
    
    dA = dZ*Ws{end};
    
    % Hidden layers
    for i=net.numHidden:-1:1
        A_curr = As{i+1};
        switch lower(net.act)
            case 'relu', dZ=dA.*(A_curr>0);
            case 'tanh', dZ=dA.*(1-A_curr.^2);
            case 'sigmoid', dZ=dA.*A_curr.*(1-A_curr);
        end
        gW = dZ.' * As{i};
        gb = sum(dZ,1).';
        if lam>0, gW=gW+lam*Ws{i}; end
        grads{gid}=gb(:); gid=gid-1; grads{gid}=gW(:); gid=gid-1;
        if i>1, dA=dZ*Ws{i}; end
    end
    g=vertcat(grads{:});
end

function [acc, pred] = evaluate_net(theta, net, X, T)
    [As,~,~]=forward(theta,net,X,[],0);
    [~,pred]=max(As{end},[],2);
    [~,truth]=max(T,[],2);
    acc=mean(pred==truth);
end

%% =========================================================================
%%                        OPTIMIZERS
%% =========================================================================
function [theta, hist] = train_gd(theta, net, X, T, Xv, Tv, lam, lr, iter)
    hist.tr=[]; hist.va=[];
    for k=1:iter
        [c, g] = cost_grad(theta, net, X, T, lam);
        theta = theta - lr*g;
        
        % Log
        hist.tr(end+1)=gather(c);
        [~,~,cv]=forward(theta,net,Xv,Tv,lam);
        hist.va(end+1)=gather(cv);
    end
end

function [theta, hist] = train_bfgs(theta, net, X, T, Xv, Tv, lam, lr, iter)
    hist.tr=[]; hist.va=[];
    n=numel(theta); H=eye(n,'like',theta); [c,g]=cost_grad(theta,net,X,T,lam);
    
    for k=1:iter
        hist.tr(end+1)=gather(c); [~,~,cv]=forward(theta,net,Xv,Tv,lam); hist.va(end+1)=gather(cv);
        
        p = -H*g;
        % Line search
        alpha=lr; 
        for ls=1:10
            t_new=theta+alpha*p; [cn,~]=cost_grad(t_new,net,X,T,lam);
            if cn<c, break; end 
            alpha=alpha*0.5;
        end
        
        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        s=t_new-theta; y=g_new-g; ys=dot(y,s);
        if ys>1e-10
            rho=1/ys; V=eye(n,'like',theta)-rho*(y*s.');
            H=V.'*H*V + rho*(s*s.');
        end
        theta=t_new; c=c_new; g=g_new;
    end
end

function [theta, hist] = train_dfp(theta, net, X, T, Xv, Tv, lam, lr, iter)
    % DFP is very similar to BFGS but update rule differs
    hist.tr=[]; hist.va=[];
    n=numel(theta); H=eye(n,'like',theta); [c,g]=cost_grad(theta,net,X,T,lam);
    
    for k=1:iter
        hist.tr(end+1)=gather(c); [~,~,cv]=forward(theta,net,Xv,Tv,lam); hist.va(end+1)=gather(cv);
        
        p = -H*g;
        alpha=lr;
        for ls=1:10, t_new=theta+alpha*p; [cn,~]=cost_grad(t_new,net,X,T,lam); if cn<c, break; end; alpha=alpha*0.5; end
        
        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        s=t_new-theta; y=g_new-g; ys=dot(y,s);
        if ys>1e-10
            Hy=H*y; yHy=dot(y,Hy);
            H = H + (s*s.')/ys - (Hy*Hy.')/yHy;
        end
        theta=t_new; c=c_new; g=g_new;
    end
end

function [theta, hist] = train_cg(theta, net, X, T, Xv, Tv, lam, lr, iter)
    hist.tr=[]; hist.va=[];
    [c,g]=cost_grad(theta,net,X,T,lam); p=-g;
    
    for k=1:iter
        hist.tr(end+1)=gather(c); [~,~,cv]=forward(theta,net,Xv,Tv,lam); hist.va(end+1)=gather(cv);
        
        alpha=lr;
        for ls=1:10, t_new=theta+alpha*p; [cn,~]=cost_grad(t_new,net,X,T,lam); if cn<c, break; end; alpha=alpha*0.5; end
        
        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        beta = max(0, dot(g_new, g_new-g)/dot(g,g));
        p = -g_new + beta*p;
        theta=t_new; c=c_new; g=g_new;
    end
end

function [theta, hist] = train_abc(theta, net, X, T, Xv, Tv, lam, SN, lim, maxCycle)
    % Basic ABC implementation
    hist.tr=[]; hist.va=[];
    D = numel(theta);
    Foods = repmat(theta, 1, SN) + randn(D,SN,'like',theta)*0.1;
    costF = zeros(1,SN,'like',theta);
    for i=1:SN, [costF(i),~]=cost_grad(Foods(:,i),net,X,T,lam); end
    
    [~,bestI]=min(costF); globalBest=Foods(:,bestI); globalCost=costF(bestI);
    trial=zeros(1,SN);
    
    for cycle=1:maxCycle
        hist.tr(end+1)=gather(globalCost); 
        [~,~,cv]=forward(globalBest,net,Xv,Tv,lam); hist.va(end+1)=gather(cv);
        
        % Employed
        for i=1:SN
            k=randi(SN); while k==i, k=randi(SN); end
            phi = (rand(D,1,'like',theta)*2-1);
            sol = Foods(:,i) + phi.*(Foods(:,i)-Foods(:,k));
            [cNew,~]=cost_grad(sol,net,X,T,lam);
            if cNew<costF(i), Foods(:,i)=sol; costF(i)=cNew; trial(i)=0; else, trial(i)=trial(i)+1; end
        end
        % Onlooker (skipped for brevity/speed in demo)
        % Scout
        [maxT, ind]=max(trial);
        if maxT > lim
            Foods(:,ind)=randn(D,1,'like',theta)*0.1;
            [costF(ind),~]=cost_grad(Foods(:,ind),net,X,T,lam);
            trial(ind)=0;
        end
        
        [minC, bi]=min(costF);
        if minC < globalCost, globalCost=minC; globalBest=Foods(:,bi); end
    end
    theta=globalBest;
end

%% =========================================================================
%%                        PLOTTING
%% =========================================================================
function plot_results(results)
    N = numel(results);
    
    % 1. Individual Histories (Subplots)
    figure('Name','Individual Performance','Color','w','Position',[100 100 1200 600]);
    cols = 3; rows = ceil(N/cols);
    for i=1:N
        subplot(rows,cols,i); hold on; grid on;
        r = results(i);
        plot(r.hist.tr, 'b-', 'LineWidth', 1.5);
        plot(r.hist.va, 'r--', 'LineWidth', 1.5);
        title(sprintf('%s (%s)', r.job.method, r.job.activation));
        if i==1, legend('Train Cost','Val Cost'); end
        xlabel('Iter'); ylabel('Cost');
    end
    
    % 2. Consolidated Comparison (Validation Cost)
    figure('Name','Method Comparison (Validation)','Color','w'); hold on; grid on;
    colors = lines(N);
    validMethods = {};
    for i=1:N
        r = results(i);
        plot(r.hist.va, 'Color', colors(i,:), 'LineWidth', 2);
        validMethods{end+1} = r.job.method;
    end
    legend(validMethods);
    xlabel('Iteration'); ylabel('Validation Cost (Loss)');
    title('Convergence Comparison of Optimization Methods');
    
    % 3. Bar Chart (Test Accuracy)
    figure('Name','Final Test Accuracy','Color','w');
    accs = [results.finalTestAcc] * 100;
    b = bar(accs, 'FaceColor','flat');
    b.CData = colors(1:N,:);
    set(gca, 'XTickLabel', validMethods);
    ylabel('Accuracy (%)'); 
    title('Final Test Accuracy by Method');
    ylim([min(accs)-5, 100]);
    for i=1:N
        text(i, accs(i), sprintf('%.1f%%', accs(i)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom','FontWeight','bold');
    end
end

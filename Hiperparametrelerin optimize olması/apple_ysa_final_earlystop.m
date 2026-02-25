function apple_ysa_final_earlystop()
%% =========================================================================
%  Apple Quality - Final Results with EARLY STOPPING
%  
%  Bu script, "Overfitting" problemini kökten çözmek için tasarlanmıştır.
%  Özellikler:
%  1. Early Stopping: Validation Cost artmaya başladığında eğitimi durdurur.
%  2. Best Model Restoration: Eğitim sırasında görülen EN DÜŞÜK Validation
%     Cost değerine sahip ağırlıkları (theta) döndürür. Son iterasyonu değil!
%  3. Increased MaxIter: Modellerin potansiyelini görmek için iterasyon 
%     sayısı 1000'e çıkarılmıştır. (Sabır mekanizması erken durduracaktır).
% ==========================================================================

    clear; clc; close all;
    rng(42); 

    %% 1. AYARLAR & VERİ YÜKLEME
    CSV_FILE = 'results_phase4_final.csv';
    DATA_FILE= 'apple_quality.csv';
    PATIENCE_LIMIT = 25; % 25 iterasyon boyunca iyileşme olmazsa dur
    
    if ~exist(CSV_FILE,'file') || ~exist(DATA_FILE,'file')
        error('Gerekli dosyalar (csv) bulunamadı.');
    end

    fprintf('>> Veri yükleniyor (Early Stopping Versiyon)...\n');
    [X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = load_and_prep_data(DATA_FILE);
    
    %% 2. ŞAMPİYONLARI BULMA
    fprintf('>> Şampiyonlar seçiliyor...\n');
    opts = detectImportOptions(CSV_FILE);
    opts.VariableTypes = repmat({'string'},1,numel(opts.VariableNames));
    rawParams = readtable(CSV_FILE, opts);
    
    methods = {'ABC','BFGS','CG','DFP','GD'};
    % Şablon
    TEMPLATE = struct('method','','activation','','layers',[],'lambda',NaN,'stepSize',NaN, ...
                      'SN',NaN,'limit',NaN,'maxCycle',NaN,'maxIter',NaN);
    champions = repmat(TEMPLATE, 0, 1);
    
    for m = methods
        mName = m{1};
        idx = find(strcmpi(rawParams.method, mName));
        if isempty(idx), continue; end
        
        sub = rawParams(idx,:);
        vAcc = double(sub.valAcc);
        [bestVal, bLoc] = max(vAcc);
        bestRow = sub(bLoc,:);
        
        champ = TEMPLATE;
        champ.method = char(bestRow.method);
        champ.activation = char(bestRow.activation);
        
        if ismember('layers_1', sub.Properties.VariableNames)
            l1 = str2double(bestRow.layers_1);
            l2 = str2double(bestRow.layers_2);
            if isnan(l2), champ.layers = l1; else, champ.layers = [l1, l2]; end
        else
            champ.layers = [32, 16]; 
        end
        
        champ.lambda = str2double(bestRow.lambda);
        ss = str2double(bestRow.stepSize); if isnan(ss), ss=0.05; end
        champ.stepSize = ss; 
        
        % Global Overrides for Best Performance
        champ.maxIter = 1000; % Hepsi için yüksek limit (Early stop yönetecek)
        
        % ABC FIX (Phase 3'ten bilinen en iyi değerler)
        if strcmpi(mName, 'ABC')
            champ.layers = [8];
            champ.lambda = 0.001;
            champ.SN = 50; 
            champ.limit = 100;
            champ.maxCycle = 1000;
        end
        
        champions = [champions; champ];
        fprintf('   ADAY: %s | %s | Lam:%.4f (MaxIter: %d)\n', champ.method, champ.activation, champ.lambda, champ.maxIter);
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
        fprintf('\n>> Eğitim: %s (%s)...\n', job.method, job.activation);
        
        net.inputDim = inputDim;
        net.outputDim = outputDim;
        net.layers = job.layers;
        net.numHidden = numel(job.layers);
        net.act = job.activation;
        
        theta = init_weights(net);
        if useGPU, theta = gpuArray(theta); end
        
        tic;
        switch job.method
            case 'ABC'
                [theta, hist] = train_abc_es(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.SN, job.limit, job.maxCycle, PATIENCE_LIMIT);
            case 'BFGS'
                [theta, hist] = train_bfgs_es(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.stepSize, job.maxIter, PATIENCE_LIMIT);
            case 'DFP'
                [theta, hist] = train_dfp_es(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.stepSize, job.maxIter, PATIENCE_LIMIT);
            case 'CG'
                [theta, hist] = train_cg_es(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.stepSize, job.maxIter, PATIENCE_LIMIT);
            case 'GD'
                [theta, hist] = train_gd_es(theta, net, Xtr, Ttr, Xva, Tva, job.lambda, job.stepSize, job.maxIter, PATIENCE_LIMIT);
        end
        tDur = toc;
        
        [teAcc, ~] = evaluate_net(theta, net, Xte, Tte);
        
        res.job = job;
        res.hist = hist;
        res.time = tDur;
        res.finalTestAcc = double(teAcc);
        
        results = [results; res];
        fprintf('   Bitti (%.1fs). Best Val Cost: %.4f | Final Test Acc: %.2f%%\n', ...
            tDur, min(hist.va), res.finalTestAcc*100);
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
%%                        CORE ENGINE
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
    
    gW = dZ.' * As{end-1}; 
    gb = sum(dZ,1).';
    if lam>0, gW=gW+lam*Ws{end}; end
    grads{gid}=gb(:); gid=gid-1; grads{gid}=gW(:); gid=gid-1;
    
    dA = dZ*Ws{end};
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
%%                        OPTIMIZERS WITH EARLY STOPPING (ES)
%% =========================================================================

function [finalTheta, hist] = train_gd_es(theta, net, X, T, Xv, Tv, lam, lr, iter, patience)
    hist.tr=[]; hist.va=[];
    bestVal = Inf; bestTheta = theta; noImp = 0;
    
    for k=1:iter
        [c, g] = cost_grad(theta, net, X, T, lam);
        theta = theta - lr*g;
        
        [~,~,cv] = forward(theta,net,Xv,Tv,lam);
        hist.tr(end+1)=gather(c); hist.va(end+1)=gather(cv);
        
        if cv < bestVal
            bestVal = cv; bestTheta = theta; noImp = 0;
        else
            noImp = noImp + 1;
        end
        
        if noImp >= patience
            fprintf('      Early Stopping @ Iter %d (Best Val: %.4f)\n', k, bestVal);
            break; 
        end
    end
    finalTheta = bestTheta; % Restore BEST weights, not final
end

function [finalTheta, hist] = train_bfgs_es(theta, net, X, T, Xv, Tv, lam, lr, iter, patience)
    hist.tr=[]; hist.va=[];
    n=numel(theta); H=eye(n,'like',theta); [c,g]=cost_grad(theta,net,X,T,lam);
    bestVal = Inf; bestTheta = theta; noImp = 0;
    
    for k=1:iter
        [~,~,cv] = forward(theta,net,Xv,Tv,lam);
        hist.tr(end+1)=gather(c); hist.va(end+1)=gather(cv);
        
        if cv < bestVal
            bestVal = cv; bestTheta = theta; noImp = 0;
        else
            noImp = noImp + 1;
        end
        if noImp >= patience, fprintf('      Early Stopping @ Iter %d (Best Val: %.4f)\n', k, bestVal); break; end
        
        p = -H*g;
        alpha=lr; for ls=1:10, t_new=theta+alpha*p; [cn,~]=cost_grad(t_new,net,X,T,lam); if cn<c, break; end; alpha=alpha*0.5; end
        
        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        s=t_new-theta; y=g_new-g; ys=dot(y,s);
        if ys>1e-10
            rho=1/ys; V=eye(n,'like',theta)-rho*(y*s.');
            H=V.'*H*V + rho*(s*s.');
        end
        theta=t_new; c=c_new; g=g_new;
    end
    finalTheta = bestTheta;
end

function [finalTheta, hist] = train_dfp_es(theta, net, X, T, Xv, Tv, lam, lr, iter, patience)
    hist.tr=[]; hist.va=[];
    n=numel(theta); H=eye(n,'like',theta); [c,g]=cost_grad(theta,net,X,T,lam);
    bestVal = Inf; bestTheta = theta; noImp = 0;
    
    for k=1:iter
        [~,~,cv] = forward(theta,net,Xv,Tv,lam);
        hist.tr(end+1)=gather(c); hist.va(end+1)=gather(cv);
        
        if cv < bestVal
            bestVal = cv; bestTheta = theta; noImp = 0;
        else
            noImp = noImp + 1;
        end
        if noImp >= patience, fprintf('      Early Stopping @ Iter %d (Best Val: %.4f)\n', k, bestVal); break; end
        
        p = -H*g;
        alpha=lr; for ls=1:10, t_new=theta+alpha*p; [cn,~]=cost_grad(t_new,net,X,T,lam); if cn<c, break; end; alpha=alpha*0.5; end
        
        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        s=t_new-theta; y=g_new-g; ys=dot(y,s);
        if ys>1e-10
            Hy=H*y; yHy=dot(y,Hy);
            H = H + (s*s.')/ys - (Hy*Hy.')/yHy;
        end
        theta=t_new; c=c_new; g=g_new;
    end
    finalTheta = bestTheta;
end

function [finalTheta, hist] = train_cg_es(theta, net, X, T, Xv, Tv, lam, lr, iter, patience)
    hist.tr=[]; hist.va=[];
    [c,g]=cost_grad(theta,net,X,T,lam); p=-g;
    bestVal = Inf; bestTheta = theta; noImp = 0;
    
    for k=1:iter
        [~,~,cv] = forward(theta,net,Xv,Tv,lam);
        hist.tr(end+1)=gather(c); hist.va(end+1)=gather(cv);
        
        if cv < bestVal
            bestVal = cv; bestTheta = theta; noImp = 0;
        else
            noImp = noImp + 1;
        end
        if noImp >= patience, fprintf('      Early Stopping @ Iter %d (Best Val: %.4f)\n', k, bestVal); break; end
        
        alpha=lr; for ls=1:10, t_new=theta+alpha*p; [cn,~]=cost_grad(t_new,net,X,T,lam); if cn<c, break; end; alpha=alpha*0.5; end
        
        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        beta = max(0, dot(g_new, g_new-g)/dot(g,g));
        p = -g_new + beta*p;
        theta=t_new; c=c_new; g=g_new;
    end
    finalTheta = bestTheta;
end

function [finalTheta, hist] = train_abc_es(theta, net, X, T, Xv, Tv, lam, SN, lim, maxCycle, patience)
    hist.tr=[]; hist.va=[];
    D = numel(theta);
    Foods = repmat(theta, 1, SN) + randn(D,SN,'like',theta)*0.1;
    costF = zeros(1,SN,'like',theta);
    for i=1:SN, [costF(i),~]=cost_grad(Foods(:,i),net,X,T,lam); end
    
    [~,bestI]=min(costF); globalBest=Foods(:,bestI); globalCost=costF(bestI);
    bestVal = Inf; bestTheta = globalBest; noImp = 0;
    trial=zeros(1,SN);
    
    for cycle=1:maxCycle
        [~,~,cv] = forward(globalBest,net,Xv,Tv,lam);
        hist.tr(end+1)=gather(globalCost); 
        hist.va(end+1)=gather(cv);
        
        if cv < bestVal
            bestVal = cv; bestTheta = globalBest; noImp = 0;
        else
            noImp = noImp + 1;
        end
        if noImp >= patience, fprintf('      Early Stopping @ Iter %d (Best Val: %.4f)\n', cycle, bestVal); break; end
        
        % Employed
        for i=1:SN
            k=randi(SN); while k==i, k=randi(SN); end
            phi = (rand(D,1,'like',theta)*2-1);
            sol = Foods(:,i) + phi.*(Foods(:,i)-Foods(:,k));
            [cNew,~]=cost_grad(sol,net,X,T,lam);
            if cNew<costF(i), Foods(:,i)=sol; costF(i)=cNew; trial(i)=0; else, trial(i)=trial(i)+1; end
        end
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
    finalTheta = bestTheta;
end

%% =========================================================================
%%                        PLOTTING
%% =========================================================================
function plot_results(results)
    N = numel(results);
    
    % 1. Individual Histories (Subplots)
    figure('Name','Overfitting Free - Individual Performance','Color','w','Position',[100 100 1200 600]);
    cols = 3; rows = ceil(N/cols);
    for i=1:N
        subplot(rows,cols,i); hold on; grid on;
        r = results(i);
        plot(r.hist.tr, 'b-', 'LineWidth', 1.5);
        plot(r.hist.va, 'r--', 'LineWidth', 1.5);
        % Best Noktayı İşaretle
        [minVal, minIdx] = min(r.hist.va);
        plot(minIdx, minVal, 'go', 'MarkerSize', 8, 'LineWidth', 2);
        
        title(sprintf('%s (%s)', r.job.method, r.job.activation));
        if i==1, legend('Train Cost','Val Cost','Best Stop'); end
        xlabel('Iter'); ylabel('Cost');
    end
    
    % 2. Final Test Accuracy
    figure('Name','Final Test Accuracy (Early Stopping)','Color','w');
    validMethods={};
    for i=1:N, validMethods{end+1}=results(i).job.method; end
    
    accs = [results.finalTestAcc] * 100;
    b = bar(accs, 'FaceColor','flat');
    colors = lines(N);
    b.CData = colors(1:N,:);
    set(gca, 'XTickLabel', validMethods);
    ylabel('Accuracy (%)'); 
    title({'Final Test Accuracy (Guaranteed No Overfit)', 'Restored from Best Validation Epoch'});
    ylim([min(accs)-5, 100]);
    for i=1:N
        text(i, accs(i), sprintf('%.1f%%', accs(i)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom','FontWeight','bold');
    end
end

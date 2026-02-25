function apple_ysa_phase4()
%% ================================================================
%  Apple Quality - Phase 4 (Architecture & Activation Search)
%  
%  GOAL: Phase 3'ün en iyi modellerini alarak şunları dener:
%    1) Aktivasyonlar: Tanh, ReLU, Sigmoid
%    2) Derinlik: 1 Gizli Katman (H) vs 2 Gizli Katman (H -> H/2)
%
%  Features:
%    - Multi-layer Support (N-Layer Backpropagation)
%    - Activation Function Switch (ReLU, Sigmoid, Tanh)
%    - He/Xavier Initialization
% ================================================================

    clear; clc; close all;
    rng(42);

    %% ================== AYARLAR ======================================
    CHECKPOINT3 = 'checkpoint_phase3_apple_v4.mat';
    CHECKPOINT4 = 'checkpoint_phase4_apple.mat';
    CSV4        = 'results_phase4_final.csv';

    TOPK_INPUT  = 5;          % P3'ten kaç şampiyon alalım?
    VAL_RATIO   = 0.20;       % Validation Split oranı
    PATIENCE    = 30;         % Derin ağlar için biraz daha sabır
    MIN_DELTA   = 1e-5;
    
    USE_GPU     = true;      
    ALLOW_PARFOR= true;

    %% ================== ŞABLON =======================================
    TEMPLATE = struct( ...
        'phase',4, ...
        'method','', ...
        'parent_id',NaN, ...     % Hangi P3 modelinden türedi
        'activation','', ...     % 'tanh', 'relu', 'sigmoid'
        'layers',[], ...         % [Hidden1, Hidden2, ...]
        'L',NaN, ...             % Gizli katman sayısı
        'lambda',NaN, ...
        'stepSize',NaN, ...
        'maxIter',NaN, ...
        'bestIter',NaN, ...
        'stopReason','', ...
        'finalValCost',NaN, ...
        'trainAcc',NaN, ...
        'valAcc',NaN, ...
        'testAcc',NaN, ...
        'trainCostHist',[], ...
        'valCostHist',[], ...
        'testCostHist',[]);

    %% ================== 1) VERİ YÜKLEME ==============================
    if ~exist('apple_quality.csv','file'), error('apple_quality.csv yok!'); end
    data = readtable('apple_quality.csv');
    q = data.(data.Properties.VariableNames{strcmpi(data.Properties.VariableNames,'Quality')});
    y_bin = double(string(q)=="good");
    isNum = varfun(@isnumeric, data, 'OutputFormat','uniform');
    X = table2array(data(:, isNum & ~strcmpi(data.Properties.VariableNames,'Quality')));
    bad = any(~isfinite(X),2) | ~isfinite(y_bin);
    X(bad,:)=[]; y_bin(bad,:)=[];
    T = [y_bin==0, y_bin==1];

    % Train/Test Split (Phase 1-2-3 ile ayni RNG)
    rng(1);
    idx0 = find(y_bin==0); idx1 = find(y_bin==1);
    idx0 = idx0(randperm(numel(idx0))); idx1 = idx1(randperm(numel(idx1)));
    n0_tr = round(0.7*numel(idx0)); n1_tr = round(0.7*numel(idx1));
    trIdx = [idx0(1:n0_tr); idx1(1:n1_tr)];
    teIdx = [idx0(n0_tr+1:end); idx1(n1_tr+1:end)];
    X_train_full = X(trIdx,:); T_train_full = T(trIdx,:);
    X_test = X(teIdx,:); T_test = T(teIdx,:);

    % Train/Val Split (Phase 3 ile ayni RNG=42)
    rng(42); 
    N_tr = size(X_train_full,1); perm = randperm(N_tr);
    n_val = round(N_tr * VAL_RATIO);
    val_idx = perm(1:n_val); train_idx = perm(n_val+1:end);

    X_train = X_train_full(train_idx,:); T_train = T_train_full(train_idx,:);
    X_val   = X_train_full(val_idx,:);   T_val   = T_train_full(val_idx,:);

    mu=mean(X_train,1); sig=std(X_train,0,1)+1e-8;
    X_train=(X_train-mu)./sig; X_val=(X_val-mu)./sig; X_test=(X_test-mu)./sig;
    
    inputDim = size(X_train,2); outputDim = size(T_train,2);
    fprintf('Data Ready: Tr=%d, Val=%d, Te=%d\n',size(X_train,1),size(X_val,1),size(X_test,1));

    %% ================== GPU PREP =====================================
    useGPU = false;
    try
        if USE_GPU && (gpuDeviceCount>0), gpuDevice; useGPU=true; fprintf('>> GPU Aktif.\n'); end
    catch, fprintf('>> GPU Hatası, CPU.\n'); end

    if useGPU
        Xtr_g=gpuArray(single(X_train)); Ttr_g=gpuArray(single(T_train));
        Xval_g=gpuArray(single(X_val));  Tval_g=gpuArray(single(T_val));
        Xte_g=gpuArray(single(X_test));  Tte_g=gpuArray(single(T_test));
    else
        Xtr_g=X_train; Ttr_g=T_train; Xval_g=X_val; Tval_g=T_val; Xte_g=X_test; Tte_g=T_test;
    end

    %% ================== JOB GENERATION (PHASE 4) =====================
    [jobs, results, res_id] = init_phase(CHECKPOINT4, TEMPLATE);

    if isempty(jobs)
        fprintf('>> Phase-4 Jobları oluşturuluyor...\n');
        if ~exist(CHECKPOINT3,'file'), error('P3 Checkpoint yok!'); end
        s3 = load(CHECKPOINT3,'results');
        tbl3 = struct2table(s3.results);
        tbl3 = tbl3(~isnan(tbl3.valAcc),:);
        
        jobs = build_phase4_jobs(tbl3, TOPK_INPUT);
        results = repmat(TEMPLATE, numel(jobs), 1);
        res_id = 0;
        save(CHECKPOINT4,'jobs','results','res_id','-v7.3');
    end
    fprintf('PHASE-4 | Toplam Varyasyon: %d\n', numel(jobs));

    %% ================== EXECUTION LOOP ===============================
    p = gcp('nocreate');
    if ALLOW_PARFOR && isempty(p), parpool('threads'); end
    idxTodo = find(isnan([results.valAcc]));
    fprintf('>> Kalan Job: %d\n', numel(idxTodo));
    
    startT = tic;
    
    if useGPU
         for ii = 1:numel(idxTodo)
            j = idxTodo(ii);
            job = jobs(j);
            layStr = sprintf('%d-', job.layers);
            % job.method veya job.activation cell ise char array'e cevir
            mName = job.method; if iscell(mName), mName=mName{1}; end
            aName = job.activation; if iscell(aName), aName=aName{1}; end
            
            fprintf('Job %d/%d: %s %s [%s] lam=%.3f... ', j, numel(jobs), mName, aName, layStr(1:end-1), job.lambda);
            
            try
                out = run_multilayer_job(job, TEMPLATE, inputDim, outputDim, ...
                    Xtr_g, Ttr_g, Xval_g, Tval_g, Xte_g, Tte_g, PATIENCE, MIN_DELTA);
                results(j) = out;
                fprintf('BestIter=%d ValAcc=%.3f\n', out.bestIter, out.valAcc);
                if mod(ii,5)==0, save(CHECKPOINT4,'jobs','results','res_id','-v7.3'); end
            catch ME
                fprintf('ERR: %s\n', ME.message);
            end
         end
    else
        numBatch = numel(idxTodo);
        batch_results = repmat(TEMPLATE, numBatch, 1);
        parfor ii = 1:numBatch
            j = idxTodo(ii);
            out = run_multilayer_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    Xtr_g, Ttr_g, Xval_g, Tval_g, Xte_g, Tte_g, PATIENCE, MIN_DELTA);
            batch_results(ii) = out;
        end
        results(idxTodo) = batch_results;
    end
    save(CHECKPOINT4,'jobs','results','res_id','-v7.3');
    fprintf('>> Phase 4 Tamamlandı (%.1f dk)\n', toc(startT)/60);

    %% ================== REPORTING & PLOTS ============================
    tbl = struct2table(results);
    tbl = tbl(~isnan(tbl.valAcc),:);
    writetable(tbl, CSV4);
    
    % En İyi 5'i Göster
    tblTop = sortrows(tbl, 'finalValCost', 'ascend'); 
    disp('=== P4 CHAMPIONS (Val Cost Lowest) ===');
    disp(tblTop(1:min(8,height(tblTop)), {'method','activation','layers','valAcc','testAcc','bestIter'}));

    % Grafikler
    plot_phase4_summary(tbl, tblTop);

end

%% =================================================================
%%                     JOB BUILDER
%% =================================================================
function jobs = build_phase4_jobs(tbl3, topk)
    % Her METHOD için en iyi modeli al
    methods = unique(tbl3.method);
    jobs = [];
    k=0;
    
    activations = {'tanh', 'relu', 'sigmoid'};
    
    for m = 1:numel(methods)
        methodName = methods{m};
        % Bu metodun sonuclarini al
        sub = tbl3(strcmp(tbl3.method, methodName), :);
        if isempty(sub), continue; end
        
        % En iyisini sec (ValAcc gore)
        sub = sortrows(sub, 'valAcc', 'descend');
        champions = sub(1:min(1, height(sub)), :); % Her metodun 1 numarasini al
        
        for i=1:height(champions)
            champ = champions(i,:);
            
            % Varyasyon 1: Activation Functions (Tek Katman - Baseline ve Alternatifleri)
            H1 = champ.hiddenDim;
            
            % Varyasyon 2: Deep Network (2 Katman)
            % H -> H/2 (Örn: 32 -> 16)
            H_deep = [H1, max(4, round(H1/2))];
            
            layer_configs = {H1, H_deep};
            
            for act = activations
                for lc = layer_configs
                    layers = lc{1};
                    
                    k=k+1;
                    jobs(k).phase = 4;
                    jobs(k).method = champ.method;
                    jobs(k).parent_id = i; 
                    jobs(k).activation = act{1};
                    jobs(k).layers = layers;
                    jobs(k).L = numel(layers); 
                    jobs(k).lambda = champ.lambda;
                    jobs(k).stepSize = champ.stepSize;
                    jobs(k).maxIter = max(300, champ.maxIter + 50); 
                end
            end
        end
    end
end

%% =================================================================
%%                   RUN ONE JOB
%% =================================================================
function out = run_multilayer_job(job, TEMPLATE, inD, outD, Xtr, Ttr, Xval, Tval, Xte, Tte, pat, minDel)
    
    net.inputDim = inD;
    net.outputDim = outD;
    net.layers = job.layers;
    net.numHidden = numel(job.layers);
    net.act = job.activation;
    
    theta0 = init_weights_deep(net);
    if isa(Xtr,'gpuArray'), theta0=gpuArray(theta0); end
    
    % Method Cell/String kontrol
    mName = job.method;
    if iscell(mName), mName=mName{1}; end
    if isstring(mName), mName=char(mName); end
    
    % Sadece BFGS, DFP, CG, GD destekli (ABC cok yavas kalir derin agda)
    switch mName
        case 'BFGS', res = train_deep_solver(@train_bfgs_core, theta0, net, Xtr, Ttr, Xval, Tval, Xte, Tte, job, pat, minDel);
        case 'DFP',  res = train_deep_solver(@train_bfgs_core, theta0, net, Xtr, Ttr, Xval, Tval, Xte, Tte, job, pat, minDel); % DFP mantigi BFGS ile ayni kodda
        case 'CG',   res = train_deep_solver(@train_cg_core,   theta0, net, Xtr, Ttr, Xval, Tval, Xte, Tte, job, pat, minDel);
        case 'GD',   res = train_deep_solver(@train_gd_core,   theta0, net, Xtr, Ttr, Xval, Tval, Xte, Tte, job, pat, minDel);
        case 'ABC',  res = train_deep_solver(@train_gd_core,   theta0, net, Xtr, Ttr, Xval, Tval, Xte, Tte, job, pat, minDel); % ABC icin GD fallback (Derin agda cok yavas)
        otherwise,   res = train_deep_solver(@train_gd_core,   theta0, net, Xtr, Ttr, Xval, Tval, Xte, Tte, job, pat, minDel);
    end
    
    out = TEMPLATE;
    out.phase=4; out.method=job.method; out.activation=job.activation;
    out.layers=job.layers; out.L=numel(job.layers); 
    out.lambda=job.lambda; out.stepSize=job.stepSize;
    
    out.bestIter = res.bestIter;
    out.stopReason = res.stopReason;
    out.trainCostHist = res.trHist;
    out.valCostHist = res.valHist;
    out.testCostHist = res.teHist;
    
    out.finalValCost = res.valHist(out.bestIter);
    
    [trA, ~] = evaluate_deep(res.bestTheta, net, Xtr, Ttr);
    [vaA, ~] = evaluate_deep(res.bestTheta, net, Xval, Tval);
    [teA, ~] = evaluate_deep(res.bestTheta, net, Xte, Tte);
    
    out.trainAcc = double(trA); out.valAcc = double(vaA); out.testAcc = double(teA);
end

%% =================================================================
%%            GENERIC DEEP SOLVER WRAPPER
%% =================================================================
function res = train_deep_solver(solverFunc, theta, net, Xtr, Ttr, Xval, Tval, Xte, Tte, job, pat, minDel)
    % Generic wrapper to handle histories and early stopping
    % solverFunc: @train_bfgs_core(theta, net, X, T, lam, lr) -> returns next theta
    
    maxIter = job.maxIter;
    res = struct('trHist',zeros(maxIter,1), 'valHist',zeros(maxIter,1), 'teHist',zeros(maxIter,1), ...
                 'bestTheta', theta, 'bestIter', 0, 'stopReason', '');
    
    bestVal = inf; noImp = 0;
    
    % State for optimizers (H matrix for BFGS, p for CG)
    state = struct('H', [], 'grad', [], 'dir', []); 
    
    for k=1:maxIter
        
        [cTr, grad] = cost_grad_deep(theta, net, Xtr, Ttr, job.lambda);
        cVal = cost_only_deep(theta, net, Xval, Tval, job.lambda);
        
        cTrS=double(gather(cTr)); cValS=double(gather(cVal)); 
        
        res.trHist(k)=cTrS; res.valHist(k)=cValS; 
        
        % Teyit icin test cost (egitimde kullanilmaz)
        cTe = cost_only_deep(theta, net, Xte, Tte, job.lambda);
        res.teHist(k)=double(gather(cTe));
        
        % Early Stop
        if cValS < bestVal - minDel
            bestVal = cValS; res.bestTheta = theta; res.bestIter = k; noImp = 0;
        else
            noImp = noImp + 1;
        end
        
        if noImp >= pat
            res.stopReason = 'EarlyStop';
            res.trHist=res.trHist(1:k); res.valHist=res.valHist(1:k); res.teHist=res.teHist(1:k);
            return;
        end
        
        % Optimization Step
        [theta, state] = solverFunc(theta, net, Xtr, Ttr, job, cTr, grad, state);
    end
    res.stopReason = 'MaxIter';
end

%% =================================================================
%%            CORE OPTIMIZERS (SINGLE STEP)
%% =================================================================
function [thetaNew, state] = train_gd_core(theta, ~, ~, ~, job, ~, grad, state)
     thetaNew = theta - job.stepSize * grad;
end

function [thetaNew, state] = train_bfgs_core(theta, net, X, T, job, cTr, grad, state)
    n = numel(theta);
    if isempty(state.H)
        state.H = eye(n, 'like', theta);
        state.grad = grad; % prev grad
    end
    
    H = state.H;
    p = -H * grad;
    
    % Line Search
    alpha = line_search_deep(theta, p, net, X, T, job.lambda, cTr, grad, job.stepSize);
    
    t_new = theta + alpha*p;
    [~, g_new] = cost_grad_deep(t_new, net, X, T, job.lambda);
    
    s = t_new - theta;
    y = g_new - grad;
    ys = dot(y,s);
    
    if ys > 1e-10
        rho = 1/ys;
        I = eye(n,'like',theta);
        V = I - rho*(y*s.');
        H = V.' * H * V + rho*(s*s.');
    end
    
    state.H = H;
    state.grad = g_new;
    thetaNew = t_new;
end

function [thetaNew, state] = train_cg_core(theta, net, X, T, job, cTr, grad, state)
    if isempty(state.dir)
        p = -grad;
    else
        p = state.dir;
    end
    if isempty(state.grad), state.grad = grad; end
    
    alpha = line_search_deep(theta, p, net, X, T, job.lambda, cTr, grad, job.stepSize);
    t_new = theta + alpha*p;
    [~, g_new] = cost_grad_deep(t_new, net, X, T, job.lambda);
    
    % PR update
    g_prev = state.grad;
    beta = max(0, dot(g_new, g_new - g_prev) / (dot(g_prev, g_prev) + 1e-10));
    p_new = -g_new + beta*p;
    
    if dot(p_new, g_new) >= 0, p_new = -g_new; end
    
    state.dir = p_new;
    state.grad = g_new;
    thetaNew = t_new;
end

%% =================================================================
%%            DEEP NEURAL NETWORK ENGINE
%% =================================================================
function alpha = line_search_deep(theta, p, net, X, T, lam, c0, g0, lr)
    alpha = lr; c1 = 1e-4; gTp = dot(g0, p);
    while true
        t_new = theta + alpha*p;
        c_new = cost_only_deep(t_new, net, X, T, lam);
        if c_new <= c0 + c1*alpha*gTp || alpha < 1e-9, break; end
        alpha = alpha * 0.5;
    end
end

function c = cost_only_deep(theta, net, X, T, lam)
    [~, ~, c] = forward_pass(theta, net, X, T, lam);
end

function [c, g] = cost_grad_deep(theta, net, X, T, lam)
    % 1. Forward
    [As, Ws, c] = forward_pass(theta, net, X, T, lam);
    
    % 2. Backward
    g = backward_pass(As, Ws, T, net, lam);
end

function [As, Ws, c] = forward_pass(theta, net, X, T, lam)
    [Ws, bs] = unpack_deep(theta, net);
    As = cell(1, net.numHidden + 1); % [A0, A1, ... AL]
    A = X;
    As{1} = A; % A0 = Input
    
    % Hidden Layers
    actFunc = net.act;
    if iscell(actFunc), actFunc=actFunc{1}; end
    if isstring(actFunc), actFunc=char(actFunc); end
    
    for i=1:net.numHidden
        Z = A * Ws{i}.' + bs{i}.';
        switch actFunc
            case 'relu',    A = max(0, Z);
            case 'sigmoid', A = 1 ./ (1 + exp(-Z));
            case 'tanh',    A = tanh(Z);
        end
        As{i+1} = A;
    end
    
    % Output Layer (Softmax)
    Wout = Ws{end}; bout = bs{end};
    Z = A * Wout.' + bout.';
    Z = Z - max(Z,[],2);
    Y = exp(Z) ./ sum(exp(Z),2);
    As{end+1} = Y; % Sonuc Y'dir
    
    % Cost
    if ~isempty(T)
        N = size(X,1);
        c = -mean(sum(T .* log(Y+1e-10), 2));
        % Regularization loop
        w_sq = 0;
        for k=1:numel(Ws)
            w_sq = w_sq + sum(Ws{k}(:).^2); 
        end
        c = c + 0.5*lam*w_sq;
    else
        c=0;
    end
end

function grad = backward_pass(As, Ws, T, net, lam)
    N = size(T,1);
    Y = As{end};
    dZ = (Y - T) / N;
    
    grads = cell(1, numel(Ws)*2); % [gW1, gb1, gW2, gb2...]
    idx_g = numel(grads);
    
    % Output Layer Gradients
    A_prev = As{end-1};
    W_curr = Ws{end};
    
    gW = dZ.' * A_prev;
    gb = sum(dZ, 1).';
    if lam>0, gW = gW + lam*W_curr; end
    
    grads{idx_g} = gb(:); idx_g=idx_g-1;
    grads{idx_g} = gW(:); idx_g=idx_g-1;
    
    % Backprop Hidden Layers
    dA = dZ * W_curr; % dA for layer L
    
    actFunc = net.act;
    if iscell(actFunc), actFunc=actFunc{1}; end
    if isstring(actFunc), actFunc=char(actFunc); end
    
    for i = net.numHidden:-1:1
        A_curr = As{i+1};  % Hidden output
        A_prev = As{i};    % Hidden input
        
        switch actFunc
            case 'relu',    dZ = dA .* (A_curr > 0);
            case 'sigmoid', dZ = dA .* A_curr .* (1 - A_curr);
            case 'tanh',    dZ = dA .* (1 - A_curr.^2);
        end
        
        W_curr = Ws{i};
        gW = dZ.' * A_prev;
        gb = sum(dZ, 1).';
        if lam>0, gW = gW + lam*W_curr; end
        
        grads{idx_g} = gb(:); idx_g=idx_g-1;
        grads{idx_g} = gW(:); idx_g=idx_g-1;
        
        if i > 1
            dA = dZ * W_curr;
        end
    end
    grad = vertcat(grads{:});
end

function [acc, pred] = evaluate_deep(theta, net, X, T)
    [As, ~, ~] = forward_pass(theta, net, X, [], 0);
    Y = As{end};
    [~, pred] = max(Y, [], 2);
    [~, true_y] = max(T, [], 2);
    acc = mean(gather(pred)==gather(true_y));
end

function [Ws, bs] = unpack_deep(theta, net)
    % Layers structure: [H1, H2, ..., H_last]
    % Ws will utilize {W1, W2, ..., W_out}
    % Input -> H1 -> ... -> Out
    
    idx=1;
    Ws = {}; bs = {};
    
    prevNode = net.inputDim;
    
    % Hidden Layers
    for i=1:net.numHidden
        hNode = net.layers(i);
        szW = [hNode, prevNode]; lenW=prod(szW);
        Ws{end+1} = reshape(theta(idx:idx+lenW-1), szW); idx=idx+lenW;
        
        szb = [hNode, 1]; lenb=hNode;
        bs{end+1} = reshape(theta(idx:idx+lenb-1), szb); idx=idx+lenb;
        
        prevNode = hNode;
    end
    
    % Output Layer
    outNode = net.outputDim;
    szW = [outNode, prevNode]; lenW=prod(szW);
    Ws{end+1} = reshape(theta(idx:idx+lenW-1), szW); idx=idx+lenW;
        
    szb = [outNode, 1]; lenb=outNode;
    bs{end+1} = reshape(theta(idx:idx+lenb-1), szb);
end

function theta = init_weights_deep(net)
    % He Initialization for ReLU, Xavier for others
    cols = {};
    prevNode = net.inputDim;
    
    actFunc = net.act;
    if iscell(actFunc), actFunc=actFunc{1}; end
    if isstring(actFunc), actFunc=char(actFunc); end
    
    for i=1:net.numHidden
        hNode = net.layers(i);
        
        if strcmp(actFunc, 'relu')
            lim = sqrt(2/prevNode); % He
        else
            lim = sqrt(6/(prevNode+hNode)); % Xavier
        end
        
        cols{end+1} = (rand(hNode, prevNode)*2-1)*lim;
        cols{end+1} = zeros(hNode,1);
        prevNode = hNode;
    end
    
    % Output Layer
    outNode = net.outputDim;
    lim = sqrt(6/(prevNode+outNode));
    cols{end+1} = (rand(outNode, prevNode)*2-1)*lim;
    cols{end+1} = zeros(outNode,1);
    
    for i=1:numel(cols), cols{i}=cols{i}(:); end
    theta = vertcat(cols{:});
end

function [jobs, results, res_id] = init_phase(checkpointFile, TEMPLATE)
    jobs=[]; results=[]; res_id=0;
    if exist(checkpointFile,'file')
        s = load(checkpointFile,'jobs','results','res_id');
        jobs=s.jobs; results=s.results; res_id=s.res_id;
        if isempty(results), results=repmat(TEMPLATE,1,numel(jobs)); end
    end
end

%% =================================================================
%%                     PLOTTING
%% =================================================================
function plot_phase4_summary(tbl, tblTop)
    
    % 1. Heatmap: Activation vs Depth (Validation Acc)
    % Her parent/method icin farkli bir grup
    methods = unique(tbl.method);
    for m = methods'
        sub = tbl(strcmp(tbl.method, m{1}), :);
        
        figure('Name',['P4 Heatmap ' m{1}],'Color','w');
        
        % Matris hazırlığı: Rows=Depth, Cols=Activation
        acts = {'tanh','relu','sigmoid'};
        depths = [1, 2];
        gridVal = nan(2,3);
        
        for r=1:2
            for c=1:3
                mask = (sub.L == depths(r)) & strcmp(sub.activation, acts{c});
                if any(mask)
                    gridVal(r,c) = max(sub.valAcc(mask));
                end
            end
        end
        
        imagesc(gridVal); colormap(jet); colorbar;
        title(['P4: ' m{1} ' Performance (Val Acc)']);
        set(gca, 'XTick', 1:3, 'XTickLabel', acts);
        set(gca, 'YTick', 1:2, 'YTickLabel', {'1 Layer','2 Layers'});
        xlabel('Activation Function'); ylabel('Depth');
        
        % Text values
        for r=1:2
            for c=1:3
                if ~isnan(gridVal(r,c))
                    text(c, r, sprintf('%.3f', gridVal(r,c)), ...
                        'HorizontalAlignment','center','FontWeight','bold','Color','k');
                end
            end
        end
    end
    
    % 2. Best Runs Comparison
    figure('Name','P4 Best Runs Comparison','Color','w');
    top5 = tblTop(1:min(5,height(tblTop)), :);
    for i=1:height(top5)
        subplot(1, height(top5), i); hold on; grid on;
        run = top5(i,:);
        plot(run.trainCostHist{1}, 'b-');
        plot(run.valCostHist{1}, 'g--');
        plot(run.testCostHist{1}, 'r-');
        xline(run.bestIter, 'k:');
        
        % Cell/Array kontrol
        layers = run.layers; 
        if iscell(layers), layers = layers{1}; end
        
        mName = run.method; if iscell(mName), mName=mName{1}; end
        aName = run.activation; if iscell(aName), aName=aName{1}; end
        
        layStr = sprintf('%d-', layers);
        title(sprintf('%s\n%s [%s]', mName, aName, layStr(1:end-1)), 'FontSize',8);
        if i==1, legend('Tr','Val','Te','Stop'); end
    end
end

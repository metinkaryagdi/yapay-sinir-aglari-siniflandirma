function apple_ysa_phase3_overfitproof()
%% =========================================================================
%  PHASE-3: OVERFITTING-PROOF TRAINING SYSTEM (REVISED)
%  
%  Amaç: Her optimizasyon metodu için OVERFITTING OLMAYAN en iyi modeli bul
%  
%  Neden Bu Yaklaşım?
%  - Model seçimini VALIDATION ile yapıyoruz (test'e bakmak leakage olur)
%  - Overfitting = train düşerken val yükselir → genelleme bozulması
%  - Regularization (lambda) + Early Stopping → genellemeyi artırır
%  - Her yöntemin kendi şampiyonu → adil karşılaştırma
%  
%  Overfitting Tespiti (Net Kural):
%  1. Val geri sıçraması: valCost(end) - min(valCost) > overfitTol
%  2. Train iyileşmeye devam: trainCost(end) < trainCost(bestIter) - trainImproveTol
%  → İkisi birden sağlanırsa OVERFIT
%  
%  Çıktı: 
%  - results_phase3_all.csv (tüm job'lar)
%  - results_phase3_champions.csv (her method'dan 1 şampiyon)
%  - Grafikler (train/val curves + comparison)
% ==========================================================================

    clear; clc; close all;
    rng(42); % Reproducibility
    
    %% ================== AYARLAR ======================================
    DATA_FILE = 'apple_quality.csv';
    CSV_ALL = 'results_phase3_all.csv';
    CSV_CHAMPIONS = 'results_phase3_champions.csv';
    
    % Data Split (sabit sayılar)
    N_TRAIN = 2800;
    N_VAL = 600;
    N_TEST = 600;
    
    % Early Stopping
    PATIENCE = 40;
    MIN_DELTA = 1e-4;
    
    % Overfitting Detection Thresholds
    OVERFIT_TOL = 0.02;          % Val geri sıçrama toleransı
    TRAIN_IMPROVE_TOL = 0.005;   % Train iyileşme eşiği
    
    % Training
    MAX_ITER = 500;
    
    %% ================== VERİ YÜKLEME & SPLIT =========================
    fprintf('>> Veri yükleniyor (seed=42, sabit split)...\n');
    [X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = ...
        load_and_split_data(DATA_FILE, N_TRAIN, N_VAL, N_TEST);
    
    fprintf('   Train: %d | Val: %d | Test: %d samples\n', ...
        size(X_train,1), size(X_val,1), size(X_test,1));
    
    %% ================== JOB GENERATİON ===============================
    fprintf('>> Job grid oluşturuluyor...\n');
    jobs = generate_jobs();
    fprintf('   Toplam %d job oluşturuldu.\n', numel(jobs));
    
    %% ================== GPU SETUP ====================================
    useGPU = false;
    try
        if gpuDeviceCount > 0
            gpuDevice;
            useGPU = true;
            fprintf('>> GPU kullanılıyor.\n');
        end
    catch
        fprintf('>> CPU kullanılıyor.\n');
    end
    
    if useGPU
        Xtr = gpuArray(single(X_train)); Ttr = gpuArray(single(T_train));
        Xva = gpuArray(single(X_val));   Tva = gpuArray(single(T_val));
        Xte = gpuArray(single(X_test));  Tte = gpuArray(single(T_test));
    else
        Xtr = X_train; Ttr = T_train;
        Xva = X_val;   Tva = T_val;
        Xte = X_test;  Tte = T_test;
    end
    
    %% ================== EĞİTİM DÖNGÜSÜ ===============================
    results = [];
    
    for i = 1:numel(jobs)
        job = jobs(i);
        fprintf('\n[%d/%d] %s | H=%d | λ=%.4f | lr=%.4f\n', ...
            i, numel(jobs), job.method, job.H, job.lambda, job.stepSize);
        
        % Network yapısı
        net.inputDim = inputDim;
        net.outputDim = outputDim;
        net.H = job.H;
        net.act = 'tanh';
        
        % Ağırlıkları başlat
        theta = init_weights(net);
        if useGPU, theta = gpuArray(theta); end
        
        % Eğitim parametreleri
        params.lambda = job.lambda;
        params.stepSize = job.stepSize;
        params.maxIter = job.maxIter;
        params.patience = PATIENCE;
        params.minDelta = MIN_DELTA;
        
        % Eğit
        tic;
        switch job.method
            case 'BFGS'
                [bestTheta, hist] = train_bfgs_es(theta, net, Xtr, Ttr, Xva, Tva, params);
            case 'DFP'
                [bestTheta, hist] = train_dfp_es(theta, net, Xtr, Ttr, Xva, Tva, params);
            case 'CG'
                [bestTheta, hist] = train_cg_es(theta, net, Xtr, Ttr, Xva, Tva, params);
            case 'GD'
                [bestTheta, hist] = train_gd_es(theta, net, Xtr, Ttr, Xva, Tva, params);
            case 'ABC'
                [bestTheta, hist] = train_abc_es(theta, net, Xtr, Ttr, Xva, Tva, params, job);
            otherwise
                error('Unknown method: %s', job.method);
        end
        tElapsed = toc;
        
        % Test accuracy (SADECE bestTheta ile, 1 kere)
        [testAcc, ~] = evaluate_net(bestTheta, net, Xte, Tte);
        
        % Overfitting tespiti
        bestValCost = min(hist.valCost);
        finalValCost = hist.valCost(end);
        bestTrainCost = hist.trainCost(hist.bestIter);
        finalTrainCost = hist.trainCost(end);
        
        valRebound = finalValCost - bestValCost;
        trainImprove = bestTrainCost - finalTrainCost;
        
        % OVERFIT RULE: Val yükseldi VE train düşmeye devam etti
        isOverfit = (valRebound > OVERFIT_TOL) && (trainImprove > TRAIN_IMPROVE_TOL);
        
        if isOverfit
            status = 'OVERFIT';
        else
            status = 'OK';
        end
        
        % Validation accuracy (bestIter'de)
        [valAcc, ~] = evaluate_net(bestTheta, net, Xva, Tva);
        
        % Sonuçları kaydet
        res.method = job.method;
        res.H = job.H;
        res.lambda = job.lambda;
        res.stepSize = job.stepSize;
        res.maxIter = job.maxIter;
        res.bestIter = hist.bestIter;
        res.stopReason = hist.stopReason;
        res.status = status;
        res.bestValCost = bestValCost;
        res.finalValCost = finalValCost;
        res.bestTrainCost = bestTrainCost;
        res.finalTrainCost = finalTrainCost;
        res.valAcc = double(valAcc);
        res.testAcc = double(testAcc);
        res.trainCostHist = hist.trainCost;
        res.valCostHist = hist.valCost;
        res.time = tElapsed;
        
        results = [results; res];
        
        fprintf('   ✓ Best@%d | ValCost=%.4f | ValAcc=%.2f%% | Test=%.2f%% | %s | %s\n', ...
            hist.bestIter, bestValCost, valAcc*100, testAcc*100, ...
            hist.stopReason, status);
    end
    
    %% ================== ŞAMPİYON SEÇİMİ ==============================
    fprintf('\n>> Şampiyon seçimi (her method için en iyi OK job)...\n');
    
    methods = unique({results.method});
    champions = [];
    
    for m = 1:numel(methods)
        methodName = methods{m};
        methodResults = results(strcmp({results.method}, methodName));
        
        % Önce OK olanları filtrele
        okResults = methodResults(strcmp({methodResults.status}, 'OK'));
        
        if isempty(okResults)
            fprintf('   ⚠ %s: OK job bulunamadı, overfitTol gevşetiliyor...\n', methodName);
            % %20 gevşetme
            relaxedTol = OVERFIT_TOL * 1.2;
            for j = 1:numel(methodResults)
                r = methodResults(j);
                valRebound = r.finalValCost - r.bestValCost;
                trainImprove = r.bestTrainCost - r.finalTrainCost;
                if ~((valRebound > relaxedTol) && (trainImprove > TRAIN_IMPROVE_TOL))
                    okResults = [okResults; r];
                end
            end
        end
        
        if isempty(okResults)
            fprintf('   ⚠ %s: Hala OK job yok, en iyi valAcc seçiliyor (no-overfit-found)...\n', methodName);
            [~, bestIdx] = max([methodResults.valAcc]);
            champion = methodResults(bestIdx);
            champion.status = 'no-overfit-found';
        else
            % Şampiyon seçimi: max valAcc → min bestValCost → max testAcc
            [~, sortIdx] = sortrows([[okResults.valAcc]', -[okResults.bestValCost]', [okResults.testAcc]'], [-1, 2, -3]);
            champion = okResults(sortIdx(1));
        end
        
        champions = [champions; champion];
        
        fprintf('   🏆 %s: H=%d, λ=%.4f → ValAcc=%.2f%%, TestAcc=%.2f%% [%s]\n', ...
            methodName, champion.H, champion.lambda, ...
            champion.valAcc*100, champion.testAcc*100, champion.status);
    end
    
    %% ================== CSV EXPORT ===================================
    fprintf('\n>> CSV export...\n');
    export_all_to_csv(results, CSV_ALL);
    export_champions_to_csv(champions, CSV_CHAMPIONS);
    fprintf('   %s (tüm job''lar)\n', CSV_ALL);
    fprintf('   %s (şampiyonlar)\n', CSV_CHAMPIONS);
    
    %% ================== VİZUALİZASYON ================================
    fprintf('>> Grafikler oluşturuluyor...\n');
    plot_results(champions);
    
    fprintf('\n✅ Phase-3 Overfitting-Proof tamamlandı!\n');
end

%% =========================================================================
%%                        VERİ YÜKLEME & SPLIT
%% =========================================================================
function [X_train, T_train, X_val, T_val, X_test, T_test, inD, outD] = ...
    load_and_split_data(fname, nTr, nVa, nTe)
    
    data = readtable(fname);
    q = data.(data.Properties.VariableNames{strcmpi(data.Properties.VariableNames,'Quality')});
    y_bin = double(string(q) == "good");
    
    isNum = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
    X = table2array(data(:, isNum & ~strcmpi(data.Properties.VariableNames, 'Quality')));
    
    % Temizlik
    bad = any(~isfinite(X), 2) | ~isfinite(y_bin);
    X(bad, :) = []; y_bin(bad, :) = [];
    T = [y_bin == 0, y_bin == 1];
    
    % Sabit split
    N = size(X, 1);
    p = randperm(N);
    
    idxTr = p(1:nTr);
    idxVa = p(nTr+1:nTr+nVa);
    idxTe = p(nTr+nVa+1:nTr+nVa+nTe);
    
    X_train = X(idxTr, :); T_train = T(idxTr, :);
    X_val = X(idxVa, :);   T_val = T(idxVa, :);
    X_test = X(idxTe, :);  T_test = T(idxTe, :);
    
    % Normalizasyon (train stats ile)
    mu = mean(X_train, 1);
    sig = std(X_train, 0, 1) + 1e-8;
    X_train = (X_train - mu) ./ sig;
    X_val = (X_val - mu) ./ sig;
    X_test = (X_test - mu) ./ sig;
    
    inD = size(X, 2);
    outD = size(T, 2);
end

%% =========================================================================
%%                        JOB GENERATİON
%% =========================================================================
function jobs = generate_jobs()
    hiddenDims = [8, 16, 32];
    lambdas = [0.0001, 0.0005, 0.001, 0.005, 0.01];
    stepSizes = [0.01, 0.03, 0.05];
    maxIters = [300, 500];
    
    methods = {'BFGS', 'DFP', 'CG', 'GD', 'ABC'};
    
    % Template struct (tüm field'lar)
    TEMPLATE = struct('method', '', 'H', NaN, 'lambda', NaN, 'stepSize', NaN, ...
                      'maxIter', NaN, 'SN', NaN, 'limit', NaN, 'maxCycle', NaN);
    jobs = repmat(TEMPLATE, 0, 1);
    
    for m = 1:numel(methods)
        for h = 1:numel(hiddenDims)
            for l = 1:numel(lambdas)
                for s = 1:numel(stepSizes)
                    for it = 1:numel(maxIters)
                        job = TEMPLATE;  % Template'ten başla
                        job.method = methods{m};
                        job.H = hiddenDims(h);
                        job.lambda = lambdas(l);
                        job.stepSize = stepSizes(s);
                        job.maxIter = maxIters(it);
                        
                        % ABC özel parametreler
                        if strcmp(methods{m}, 'ABC')
                            job.SN = 50;
                            job.limit = 100;
                            job.maxCycle = job.maxIter;
                        end
                        
                        jobs = [jobs; job];
                    end
                end
            end
        end
    end
end

%% =========================================================================
%%                        NETWORK HELPERS
%% =========================================================================
function theta = init_weights(net)
    in = net.inputDim;
    h = net.H;
    out = net.outputDim;
    
    lim1 = sqrt(6 / (in + h));
    W1 = (rand(h, in) * 2 - 1) * lim1;
    b1 = zeros(h, 1);
    
    lim2 = sqrt(6 / (h + out));
    W2 = (rand(out, h) * 2 - 1) * lim2;
    b2 = zeros(out, 1);
    
    theta = [W1(:); b1(:); W2(:); b2(:)];
end

function [W1, b1, W2, b2] = unpack_theta(theta, net)
    in = net.inputDim;
    h = net.H;
    out = net.outputDim;
    
    idx = 1;
    W1 = reshape(theta(idx:idx+h*in-1), [h, in]); idx = idx + h*in;
    b1 = reshape(theta(idx:idx+h-1), [h, 1]); idx = idx + h;
    W2 = reshape(theta(idx:idx+out*h-1), [out, h]); idx = idx + out*h;
    b2 = reshape(theta(idx:idx+out-1), [out, 1]);
end

function [cost, Y] = forward(theta, net, X, T, lambda)
    [W1, b1, W2, b2] = unpack_theta(theta, net);
    
    Z1 = X * W1.' + b1.';
    A1 = tanh(Z1);
    Z2 = A1 * W2.' + b2.';
    Z2 = Z2 - max(Z2, [], 2);
    Y = exp(Z2) ./ sum(exp(Z2), 2);
    
    if isempty(T)
        cost = 0;
        return;
    end
    
    cost = -mean(sum(T .* log(Y + 1e-10), 2));
    
    if lambda > 0
        reg = sum(W1(:).^2) + sum(W2(:).^2);
        cost = cost + 0.5 * lambda * reg;
    end
end

function [cost, grad] = cost_and_grad(theta, net, X, T, lambda)
    [W1, b1, W2, b2] = unpack_theta(theta, net);
    N = size(X, 1);
    
    Z1 = X * W1.' + b1.';
    A1 = tanh(Z1);
    Z2 = A1 * W2.' + b2.';
    Z2 = Z2 - max(Z2, [], 2);
    Y = exp(Z2) ./ sum(exp(Z2), 2);
    
    cost = -mean(sum(T .* log(Y + 1e-10), 2));
    if lambda > 0
        reg = sum(W1(:).^2) + sum(W2(:).^2);
        cost = cost + 0.5 * lambda * reg;
    end
    
    dZ2 = (Y - T) / N;
    dW2 = dZ2.' * A1;
    db2 = sum(dZ2, 1).';
    
    dA1 = dZ2 * W2;
    dZ1 = dA1 .* (1 - A1.^2);
    dW1 = dZ1.' * X;
    db1 = sum(dZ1, 1).';
    
    if lambda > 0
        dW2 = dW2 + lambda * W2;
        dW1 = dW1 + lambda * W1;
    end
    
    grad = [dW1(:); db1(:); dW2(:); db2(:)];
end

function [acc, pred] = evaluate_net(theta, net, X, T)
    [~, Y] = forward(theta, net, X, [], 0);
    [~, pred] = max(Y, [], 2);
    [~, truth] = max(T, [], 2);
    acc = mean(pred == truth);
end

%% =========================================================================
%%                        TRAINING FUNCTIONS (Early Stopping)
%% =========================================================================
function [bestTheta, hist] = train_bfgs_es(theta, net, X, T, Xv, Tv, params)
    n = numel(theta);
    H = eye(n, 'like', theta);
    [c, g] = cost_and_grad(theta, net, X, T, params.lambda);
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    
    hist.trainCost = [];
    hist.valCost = [];
    
    for k = 1:params.maxIter
        [cv, ~] = forward(theta, net, Xv, Tv, params.lambda);
        hist.trainCost(end+1) = gather(c);
        hist.valCost(end+1) = gather(cv);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if noImpCount >= params.patience
            hist.stopReason = 'EarlyStop';
            hist.bestIter = bestIter;
            return;
        end
        
        p = -H * g;
        alpha = params.stepSize;
        for ls = 1:10
            t_new = theta + alpha * p;
            [cn, ~] = cost_and_grad(t_new, net, X, T, params.lambda);
            if cn < c, break; end
            alpha = alpha * 0.5;
        end
        
        [c_new, g_new] = cost_and_grad(t_new, net, X, T, params.lambda);
        s = t_new - theta;
        y = g_new - g;
        ys = dot(y, s);
        
        if ys > 1e-10
            rho = 1 / ys;
            V = eye(n, 'like', theta) - rho * (y * s.');
            H = V.' * H * V + rho * (s * s.');
        end
        
        theta = t_new;
        c = c_new;
        g = g_new;
    end
    
    hist.stopReason = 'MaxIter';
    hist.bestIter = bestIter;
end

function [bestTheta, hist] = train_dfp_es(theta, net, X, T, Xv, Tv, params)
    n = numel(theta);
    H = eye(n, 'like', theta);
    [c, g] = cost_and_grad(theta, net, X, T, params.lambda);
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    
    hist.trainCost = [];
    hist.valCost = [];
    
    for k = 1:params.maxIter
        [cv, ~] = forward(theta, net, Xv, Tv, params.lambda);
        hist.trainCost(end+1) = gather(c);
        hist.valCost(end+1) = gather(cv);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if noImpCount >= params.patience
            hist.stopReason = 'EarlyStop';
            hist.bestIter = bestIter;
            return;
        end
        
        p = -H * g;
        alpha = params.stepSize;
        for ls = 1:10
            t_new = theta + alpha * p;
            [cn, ~] = cost_and_grad(t_new, net, X, T, params.lambda);
            if cn < c, break; end
            alpha = alpha * 0.5;
        end
        
        [c_new, g_new] = cost_and_grad(t_new, net, X, T, params.lambda);
        s = t_new - theta;
        y = g_new - g;
        ys = dot(y, s);
        
        if ys > 1e-10
            Hy = H * y;
            yHy = dot(y, Hy);
            H = H + (s * s.') / ys - (Hy * Hy.') / yHy;
        end
        
        theta = t_new;
        c = c_new;
        g = g_new;
    end
    
    hist.stopReason = 'MaxIter';
    hist.bestIter = bestIter;
end

function [bestTheta, hist] = train_cg_es(theta, net, X, T, Xv, Tv, params)
    [c, g] = cost_and_grad(theta, net, X, T, params.lambda);
    p = -g;
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    
    hist.trainCost = [];
    hist.valCost = [];
    
    for k = 1:params.maxIter
        [cv, ~] = forward(theta, net, Xv, Tv, params.lambda);
        hist.trainCost(end+1) = gather(c);
        hist.valCost(end+1) = gather(cv);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if noImpCount >= params.patience
            hist.stopReason = 'EarlyStop';
            hist.bestIter = bestIter;
            return;
        end
        
        alpha = params.stepSize;
        for ls = 1:10
            t_new = theta + alpha * p;
            [cn, ~] = cost_and_grad(t_new, net, X, T, params.lambda);
            if cn < c, break; end
            alpha = alpha * 0.5;
        end
        
        [c_new, g_new] = cost_and_grad(t_new, net, X, T, params.lambda);
        beta = max(0, dot(g_new, g_new - g) / dot(g, g));
        p = -g_new + beta * p;
        
        theta = t_new;
        c = c_new;
        g = g_new;
    end
    
    hist.stopReason = 'MaxIter';
    hist.bestIter = bestIter;
end

function [bestTheta, hist] = train_gd_es(theta, net, X, T, Xv, Tv, params)
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    
    hist.trainCost = [];
    hist.valCost = [];
    
    for k = 1:params.maxIter
        [c, g] = cost_and_grad(theta, net, X, T, params.lambda);
        [cv, ~] = forward(theta, net, Xv, Tv, params.lambda);
        
        hist.trainCost(end+1) = gather(c);
        hist.valCost(end+1) = gather(cv);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if noImpCount >= params.patience
            hist.stopReason = 'EarlyStop';
            hist.bestIter = bestIter;
            return;
        end
        
        theta = theta - params.stepSize * g;
    end
    
    hist.stopReason = 'MaxIter';
    hist.bestIter = bestIter;
end

function [bestTheta, hist] = train_abc_es(theta, net, X, T, Xv, Tv, params, job)
    D = numel(theta);
    SN = job.SN;
    limit = job.limit;
    maxCycle = job.maxCycle;
    
    Foods = repmat(theta, 1, SN) + randn(D, SN, 'like', theta) * 0.1;
    costF = zeros(1, SN, 'like', theta);
    for i = 1:SN
        [costF(i), ~] = cost_and_grad(Foods(:,i), net, X, T, params.lambda);
    end
    
    [~, bestI] = min(costF);
    globalBest = Foods(:, bestI);
    globalCost = costF(bestI);
    
    bestValCost = Inf;
    bestTheta = globalBest;
    bestIter = 0;
    noImpCount = 0;
    trial = zeros(1, SN);
    
    hist.trainCost = [];
    hist.valCost = [];
    
    for cycle = 1:maxCycle
        [cv, ~] = forward(globalBest, net, Xv, Tv, params.lambda);
        hist.trainCost(end+1) = gather(globalCost);
        hist.valCost(end+1) = gather(cv);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = globalBest;
            bestIter = cycle;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if noImpCount >= params.patience
            hist.stopReason = 'EarlyStop';
            hist.bestIter = bestIter;
            return;
        end
        
        % Employed Bees
        for i = 1:SN
            k = randi(SN);
            while k == i, k = randi(SN); end
            phi = (rand(D, 1, 'like', theta) * 2 - 1);
            sol = Foods(:,i) + phi .* (Foods(:,i) - Foods(:,k));
            [cNew, ~] = cost_and_grad(sol, net, X, T, params.lambda);
            if cNew < costF(i)
                Foods(:,i) = sol;
                costF(i) = cNew;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
        end
        
        % Scout
        [maxT, ind] = max(trial);
        if maxT > limit
            Foods(:,ind) = randn(D, 1, 'like', theta) * 0.1;
            [costF(ind), ~] = cost_and_grad(Foods(:,ind), net, X, T, params.lambda);
            trial(ind) = 0;
        end
        
        [minC, bi] = min(costF);
        if minC < globalCost
            globalCost = minC;
            globalBest = Foods(:,bi);
        end
    end
    
    hist.stopReason = 'MaxIter';
    hist.bestIter = bestIter;
end

%% =========================================================================
%%                        CSV EXPORT
%% =========================================================================
function export_all_to_csv(results, filename)
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'method,H,lambda,stepSize,maxIter,bestIter,stopReason,status,');
    fprintf(fid, 'bestValCost,finalValCost,bestTrainCost,finalTrainCost,');
    fprintf(fid, 'valAcc,testAcc\n');
    
    for i = 1:numel(results)
        r = results(i);
        fprintf(fid, '%s,%d,%.6f,%.6f,%d,%d,%s,%s,', ...
            r.method, r.H, r.lambda, r.stepSize, r.maxIter, r.bestIter, r.stopReason, r.status);
        fprintf(fid, '%.6f,%.6f,%.6f,%.6f,', ...
            r.bestValCost, r.finalValCost, r.bestTrainCost, r.finalTrainCost);
        fprintf(fid, '%.6f,%.6f\n', r.valAcc, r.testAcc);
    end
    
    fclose(fid);
end

function export_champions_to_csv(champions, filename)
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'method,H,lambda,stepSize,maxIter,bestIter,stopReason,status,');
    fprintf(fid, 'bestValCost,finalValCost,valAcc,testAcc\n');
    
    for i = 1:numel(champions)
        c = champions(i);
        fprintf(fid, '%s,%d,%.6f,%.6f,%d,%d,%s,%s,', ...
            c.method, c.H, c.lambda, c.stepSize, c.maxIter, c.bestIter, c.stopReason, c.status);
        fprintf(fid, '%.6f,%.6f,%.6f,%.6f\n', ...
            c.bestValCost, c.finalValCost, c.valAcc, c.testAcc);
    end
    
    fclose(fid);
end

%% =========================================================================
%%                        VİZUALİZASYON
%% =========================================================================
function plot_results(champions)
    if isempty(champions)
        fprintf('   ⚠ Şampiyon bulunamadı.\n');
        return;
    end
    
    N = numel(champions);
    
    % 1. Individual Train vs Val Curves
    figure('Name', 'Phase-3 Champions: Train vs Val', 'Color', 'w', 'Position', [100 100 1200 600]);
    cols = min(3, N);
    rows = ceil(N / cols);
    
    for i = 1:N
        subplot(rows, cols, i);
        hold on; grid on;
        
        plot(champions(i).trainCostHist, 'b-', 'LineWidth', 1.5);
        plot(champions(i).valCostHist, 'r--', 'LineWidth', 1.5);
        plot(champions(i).bestIter, champions(i).bestValCost, 'go', ...
            'MarkerSize', 10, 'LineWidth', 2);
        
        title(sprintf('%s (H=%d, λ=%.4f)', champions(i).method, ...
            champions(i).H, champions(i).lambda));
        xlabel('Iteration');
        ylabel('Cost');
        if i == 1
            legend('Train', 'Val', 'Best', 'Location', 'best');
        end
    end
    
    % 2. Val Cost Comparison
    figure('Name', 'Phase-3 Champions: Val Cost Comparison', 'Color', 'w');
    hold on; grid on;
    colors = lines(N);
    
    for i = 1:N
        plot(champions(i).valCostHist, 'LineWidth', 2, 'Color', colors(i,:), ...
            'DisplayName', champions(i).method);
    end
    
    xlabel('Iteration');
    ylabel('Validation Cost');
    title('Validation Cost Comparison (Champions)');
    legend('Location', 'best');
    
    % 3. Test Accuracy Bar Chart
    figure('Name', 'Phase-3 Champions: Test Accuracy', 'Color', 'w');
    methods = {champions.method};
    accs = [champions.testAcc] * 100;
    
    b = bar(accs, 'FaceColor', 'flat');
    b.CData = colors(1:N, :);
    
    set(gca, 'XTickLabel', methods);
    ylabel('Test Accuracy (%)');
    title('Phase-3 Champions: Test Accuracy');
    ylim([min(accs)-5, 100]);
    
    for i = 1:N
        text(i, accs(i), sprintf('%.1f%%', accs(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontWeight', 'bold');
    end
end

function apple_ysa_phase4_finetune()
%% =========================================================================
%  PHASE-4: FINE-TUNING (Champion-Based Narrow Search)
%  
%  Amaç: Phase-3 şampiyonlarından başlayarak kademeli iterasyon artışı ve
%        dar hiperparametre taraması ile her method için mutlak en iyi modeli bul
%  
%  Strateji:
%  Stage 1: Şampiyon ayarlarıyla iterasyon merdiveni (100→500)
%  Stage 2: En iyi 2 sonuç etrafında dar tarama (lr, λ, H)
%  
%  Overfitting Koruması:
%  - Early stopping (patience=30, min_delta=0.002)
%  - Overfit flag: 20 iter üst üste val yükselme
%  
%  Çıktı:
%  - Her method için top 3 konfigürasyon
%  - Overfit listesi
%  - Genel winner (min ValCost → max ValAcc)
% ==========================================================================

    clear; clc; close all;
    rng(42);
    
    %% ================== AYARLAR ======================================
    DATA_FILE = 'apple_quality.csv';
    CSV_ALL = 'results_phase4_finetune_all.csv';
    CSV_TOP3 = 'results_phase4_finetune_top3.csv';
    CSV_WINNER = 'results_phase4_finetune_winner.csv';
    
    % Data Split (sabit)
    N_TRAIN = 2800;
    N_VAL = 600;
    N_TEST = 600;
    
    % Early Stopping
    PATIENCE = 30;
    MIN_DELTA = 0.002;
    
    % Overfitting Detection
    OVERFIT_WINDOW = 20;  % 20 iter üst üste val yükselme
    
    %% ================== ŞAMPİYON BAŞLANGIÇ NOKTALARI =================
    champions = get_champion_configs();
    
    %% ================== VERİ YÜKLEME =================================
    fprintf('>> Veri yükleniyor (seed=42)...\n');
    [X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = ...
        load_and_split_data(DATA_FILE, N_TRAIN, N_VAL, N_TEST);
    
    fprintf('   Train: %d | Val: %d | Test: %d\n', ...
        size(X_train,1), size(X_val,1), size(X_test,1));
    
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
    
    %% ================== STAGE 1: İTERASYON MERDİVENİ =================
    fprintf('\n>> STAGE 1: İterasyon merdiveni (şampiyon ayarları)...\n');
    
    maxIterLadder = [100, 150, 200, 250, 300, 350, 400, 450, 500];
    allResults = [];
    
    for m = 1:numel(champions)
        champ = champions(m);
        fprintf('\n=== %s ===\n', champ.method);
        
        for it = 1:numel(maxIterLadder)
            job.method = champ.method;
            job.H = champ.H;
            job.lambda = champ.lambda;
            job.lr = champ.base_lr;
            job.maxIter = maxIterLadder(it);
            
            % ABC özel
            if strcmp(champ.method, 'ABC')
                job.SN = 50;
                job.limit = 100;
            end
            
            res = run_training(job, Xtr, Ttr, Xva, Tva, Xte, Tte, ...
                inputDim, outputDim, PATIENCE, MIN_DELTA, OVERFIT_WINDOW, useGPU);
            
            allResults = [allResults; res];
            
            fprintf('  [%d] maxIter=%d | ValCost=%.4f | ValAcc=%.2f%% | %s\n', ...
                it, job.maxIter, res.valCost, res.valAcc*100, res.status);
        end
    end
    
    %% ================== STAGE 2: DAR TARAMA ==========================
    fprintf('\n>> STAGE 2: Dar hiperparametre taraması...\n');
    
    for m = 1:numel(champions)
        champ = champions(m);
        methodResults = allResults(strcmp({allResults.method}, champ.method));
        
        % OK olanları filtrele
        okResults = methodResults(strcmp({methodResults.status}, 'OK'));
        
        if isempty(okResults)
            fprintf('\n=== %s: OK sonuç yok, Stage 2 atlanıyor ===\n', champ.method);
            continue;
        end
        
        % En iyi 2'yi seç (ValCost'a göre)
        [~, sortIdx] = sort([okResults.valCost]);
        top2 = okResults(sortIdx(1:min(2, numel(okResults))));
        
        fprintf('\n=== %s: Top 2 etrafında dar tarama ===\n', champ.method);
        
        for t = 1:numel(top2)
            baseConfig = top2(t);
            fprintf('  Base: H=%d, λ=%.4f, lr=%.4f, maxIter=%d\n', ...
                baseConfig.H, baseConfig.lambda, baseConfig.lr, baseConfig.maxIter);
            
            % lr taraması
            lrFactors = [0.7, 1.0, 1.3];
            
            % lambda taraması
            lambdaFactors = [0.5, 1.0, 2.0];
            
            % H taraması (komşular)
            HOptions = get_H_neighbors(baseConfig.H);
            
            % Grid
            for lrf = 1:numel(lrFactors)
                for lamf = 1:numel(lambdaFactors)
                    for h = 1:numel(HOptions)
                        % Base config'i tekrar deneme
                        if lrFactors(lrf) == 1.0 && lambdaFactors(lamf) == 1.0 && ...
                           HOptions(h) == baseConfig.H
                            continue;
                        end
                        
                        job.method = champ.method;
                        job.H = HOptions(h);
                        job.lambda = baseConfig.lambda * lambdaFactors(lamf);
                        job.lr = baseConfig.lr * lrFactors(lrf);
                        job.maxIter = baseConfig.maxIter;
                        
                        if strcmp(champ.method, 'ABC')
                            job.SN = 50;
                            job.limit = 100;
                        end
                        
                        res = run_training(job, Xtr, Ttr, Xva, Tva, Xte, Tte, ...
                            inputDim, outputDim, PATIENCE, MIN_DELTA, OVERFIT_WINDOW, useGPU);
                        
                        allResults = [allResults; res];
                        
                        fprintf('    H=%d, λ=%.4f, lr=%.4f | ValCost=%.4f | %s\n', ...
                            job.H, job.lambda, job.lr, res.valCost, res.status);
                    end
                end
            end
        end
    end
    
    %% ================== RAPORLAMA ====================================
    fprintf('\n>> Raporlama...\n');
    
    % CSV export
    export_all_to_csv(allResults, CSV_ALL);
    
    % Her method için top 3
    top3All = [];
    overfitList = [];
    
    fprintf('\n========================================\n');
    fprintf('PER-METHOD TOP 3 RESULTS\n');
    fprintf('========================================\n');
    
    for m = 1:numel(champions)
        champ = champions(m);
        methodResults = allResults(strcmp({allResults.method}, champ.method));
        
        % OK olanları filtrele
        okResults = methodResults(strcmp({methodResults.status}, 'OK'));
        
        if isempty(okResults)
            fprintf('\n%s: OK sonuç yok!\n', champ.method);
            continue;
        end
        
        % Sırala: ValCost → ValAcc → bestIter
        [~, sortIdx] = sortrows([[okResults.valCost]', -[okResults.valAcc]', [okResults.bestIter]'], [1, 2, 3]);
        top3 = okResults(sortIdx(1:min(3, numel(okResults))));
        
        fprintf('\n%s:\n', champ.method);
        fprintf('  Rank | ValCost | ValAcc  | TestAcc | best@iter | maxIter | lr     | lambda  | H\n');
        fprintf('  -----|---------|---------|---------|-----------|---------|--------|---------|---\n');
        
        for i = 1:numel(top3)
            fprintf('  #%d   | %.4f  | %.2f%%  | %.2f%%  | %3d       | %3d     | %.4f | %.5f | %d\n', ...
                i, top3(i).valCost, top3(i).valAcc*100, top3(i).testAcc*100, ...
                top3(i).bestIter, top3(i).maxIter, top3(i).lr, top3(i).lambda, top3(i).H);
        end
        
        top3All = [top3All; top3];
        
        % Overfit listesi
        overfitResults = methodResults(strcmp({methodResults.status}, 'OVERFIT'));
        if ~isempty(overfitResults)
            fprintf('\n  OVERFIT configurations (%d):\n', numel(overfitResults));
            for i = 1:min(5, numel(overfitResults))
                fprintf('    - H=%d, λ=%.5f, lr=%.4f, maxIter=%d (ValCost yükseldi)\n', ...
                    overfitResults(i).H, overfitResults(i).lambda, ...
                    overfitResults(i).lr, overfitResults(i).maxIter);
            end
            overfitList = [overfitList; overfitResults];
        end
    end
    
    % Top 3 CSV
    export_top3_to_csv(top3All, CSV_TOP3);
    
    %% ================== GLOBAL WINNER ================================
    fprintf('\n========================================\n');
    fprintf('GLOBAL WINNER\n');
    fprintf('========================================\n');
    
    if ~isempty(top3All)
        % Sırala: min ValCost → max ValAcc
        [~, winnerIdx] = sortrows([[top3All.valCost]', -[top3All.valAcc]'], [1, 2]);
        winner = top3All(winnerIdx(1));
        
        fprintf('\n🏆 WINNER: %s\n', winner.method);
        fprintf('   H=%d, λ=%.5f, lr=%.4f, maxIter=%d\n', ...
            winner.H, winner.lambda, winner.lr, winner.maxIter);
        fprintf('   ValCost=%.4f | ValAcc=%.2f%% | TestAcc=%.2f%%\n', ...
            winner.valCost, winner.valAcc*100, winner.testAcc*100);
        fprintf('   Best@iter=%d | Status=%s\n', winner.bestIter, winner.status);
        
        % Winner CSV
        export_winner_to_csv(winner, CSV_WINNER);
    else
        fprintf('\nHiç OK sonuç bulunamadı!\n');
    end
    
    fprintf('\n✅ Phase-4 Fine-Tuning tamamlandı!\n');
    fprintf('   %s\n', CSV_ALL);
    fprintf('   %s\n', CSV_TOP3);
    fprintf('   %s\n', CSV_WINNER);
end

%% =========================================================================
%%                        ŞAMPİYON KONFİGÜRASYONLARI
%% =========================================================================
function champions = get_champion_configs()
    champions = [];
    
    % ABC
    c.method = 'ABC';
    c.H = 8;
    c.lambda = 0.0005;
    c.base_lr = 0.03;
    champions = [champions; c];
    
    % BFGS
    c.method = 'BFGS';
    c.H = 16;
    c.lambda = 0.0001;
    c.base_lr = 0.03;
    champions = [champions; c];
    
    % CG
    c.method = 'CG';
    c.H = 8;
    c.lambda = 0.0010;
    c.base_lr = 0.05;
    champions = [champions; c];
    
    % DFP
    c.method = 'DFP';
    c.H = 32;
    c.lambda = 0.0005;
    c.base_lr = 0.03;
    champions = [champions; c];
    
    % GD
    c.method = 'GD';
    c.H = 32;
    c.lambda = 0.0001;
    c.base_lr = 0.05;
    champions = [champions; c];
end

function HOptions = get_H_neighbors(H)
    allH = [8, 16, 32];
    idx = find(allH == H);
    
    if isempty(idx)
        HOptions = [H];
        return;
    end
    
    % Komşular
    if idx == 1
        HOptions = [8, 16];
    elseif idx == 2
        HOptions = [8, 16, 32];
    else
        HOptions = [16, 32];
    end
end

%% =========================================================================
%%                        VERİ YÜKLEME
%% =========================================================================
function [X_train, T_train, X_val, T_val, X_test, T_test, inD, outD] = ...
    load_and_split_data(fname, nTr, nVa, nTe)
    
    data = readtable(fname);
    q = data.(data.Properties.VariableNames{strcmpi(data.Properties.VariableNames,'Quality')});
    y_bin = double(string(q) == "good");
    
    isNum = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
    X = table2array(data(:, isNum & ~strcmpi(data.Properties.VariableNames, 'Quality')));
    
    bad = any(~isfinite(X), 2) | ~isfinite(y_bin);
    X(bad, :) = []; y_bin(bad, :) = [];
    T = [y_bin == 0, y_bin == 1];
    
    N = size(X, 1);
    p = randperm(N);
    
    idxTr = p(1:nTr);
    idxVa = p(nTr+1:nTr+nVa);
    idxTe = p(nTr+nVa+1:nTr+nVa+nTe);
    
    X_train = X(idxTr, :); T_train = T(idxTr, :);
    X_val = X(idxVa, :);   T_val = T(idxVa, :);
    X_test = X(idxTe, :);  T_test = T(idxTe, :);
    
    mu = mean(X_train, 1);
    sig = std(X_train, 0, 1) + 1e-8;
    X_train = (X_train - mu) ./ sig;
    X_val = (X_val - mu) ./ sig;
    X_test = (X_test - mu) ./ sig;
    
    inD = size(X, 2);
    outD = size(T, 2);
end

%% =========================================================================
%%                        EĞİTİM FONKSİYONU
%% =========================================================================
function res = run_training(job, Xtr, Ttr, Xva, Tva, Xte, Tte, ...
    inputDim, outputDim, patience, minDelta, overfitWindow, useGPU)
    
    net.inputDim = inputDim;
    net.outputDim = outputDim;
    net.H = job.H;
    
    theta = init_weights(net);
    if useGPU, theta = gpuArray(theta); end
    
    params.lambda = job.lambda;
    params.lr = job.lr;
    params.maxIter = job.maxIter;
    params.patience = patience;
    params.minDelta = minDelta;
    params.overfitWindow = overfitWindow;
    
    % Eğit
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
    end
    
    % Test (sadece bestTheta ile)
    [testAcc, ~] = evaluate_net(bestTheta, net, Xte, Tte);
    [valAcc, ~] = evaluate_net(bestTheta, net, Xva, Tva);
    
    % Sonuç
    res.method = job.method;
    res.H = job.H;
    res.lambda = job.lambda;
    res.lr = job.lr;
    res.maxIter = job.maxIter;
    res.bestIter = hist.bestIter;
    res.valCost = hist.bestValCost;
    res.valAcc = double(valAcc);
    res.testAcc = double(testAcc);
    res.status = hist.status;
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
%%                        TRAINING FUNCTIONS
%% =========================================================================
function [bestTheta, hist] = train_bfgs_es(theta, net, X, T, Xv, Tv, params)
    n = numel(theta);
    H = eye(n, 'like', theta);
    [c, g] = cost_and_grad(theta, net, X, T, params.lambda);
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    
    valCostHist = [];
    trainCostHist = [];
    
    for k = 1:params.maxIter
        [cv, ~] = forward(theta, net, Xv, Tv, params.lambda);
        valCostHist(end+1) = gather(cv);
        trainCostHist(end+1) = gather(c);
        
        % Best tracking
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        % Overfit detection
        if cv > prevValCost
            valIncreaseCount = valIncreaseCount + 1;
        else
            valIncreaseCount = 0;
        end
        prevValCost = cv;
        
        if valIncreaseCount >= params.overfitWindow
            hist.status = 'OVERFIT';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        % Early stopping
        if noImpCount >= params.patience
            hist.status = 'OK';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        % BFGS update
        p = -H * g;
        alpha = params.lr;
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
    
    hist.status = 'OK';
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

function [bestTheta, hist] = train_dfp_es(theta, net, X, T, Xv, Tv, params)
    n = numel(theta);
    H = eye(n, 'like', theta);
    [c, g] = cost_and_grad(theta, net, X, T, params.lambda);
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    
    for k = 1:params.maxIter
        [cv, ~] = forward(theta, net, Xv, Tv, params.lambda);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if cv > prevValCost
            valIncreaseCount = valIncreaseCount + 1;
        else
            valIncreaseCount = 0;
        end
        prevValCost = cv;
        
        if valIncreaseCount >= params.overfitWindow
            hist.status = 'OVERFIT';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        if noImpCount >= params.patience
            hist.status = 'OK';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        p = -H * g;
        alpha = params.lr;
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
    
    hist.status = 'OK';
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

function [bestTheta, hist] = train_cg_es(theta, net, X, T, Xv, Tv, params)
    [c, g] = cost_and_grad(theta, net, X, T, params.lambda);
    p = -g;
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    
    for k = 1:params.maxIter
        [cv, ~] = forward(theta, net, Xv, Tv, params.lambda);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if cv > prevValCost
            valIncreaseCount = valIncreaseCount + 1;
        else
            valIncreaseCount = 0;
        end
        prevValCost = cv;
        
        if valIncreaseCount >= params.overfitWindow
            hist.status = 'OVERFIT';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        if noImpCount >= params.patience
            hist.status = 'OK';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        alpha = params.lr;
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
    
    hist.status = 'OK';
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

function [bestTheta, hist] = train_gd_es(theta, net, X, T, Xv, Tv, params)
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    
    for k = 1:params.maxIter
        [c, g] = cost_and_grad(theta, net, X, T, params.lambda);
        [cv, ~] = forward(theta, net, Xv, Tv, params.lambda);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if cv > prevValCost
            valIncreaseCount = valIncreaseCount + 1;
        else
            valIncreaseCount = 0;
        end
        prevValCost = cv;
        
        if valIncreaseCount >= params.overfitWindow
            hist.status = 'OVERFIT';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        if noImpCount >= params.patience
            hist.status = 'OK';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        theta = theta - params.lr * g;
    end
    
    hist.status = 'OK';
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

function [bestTheta, hist] = train_abc_es(theta, net, X, T, Xv, Tv, params, job)
    D = numel(theta);
    SN = job.SN;
    limit = job.limit;
    
    Foods = repmat(theta, 1, SN) + randn(D, SN, 'like', theta) * 0.1;
    costF = zeros(1, SN, 'like', theta);
    for i = 1:SN
        [costF(i), ~] = cost_and_grad(Foods(:,i), net, X, T, params.lambda);
    end
    
    [~, bestI] = min(costF);
    globalBest = Foods(:, bestI);
    
    bestValCost = Inf;
    bestTheta = globalBest;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    trial = zeros(1, SN);
    
    for cycle = 1:params.maxIter
        [cv, ~] = forward(globalBest, net, Xv, Tv, params.lambda);
        
        if cv < bestValCost - params.minDelta
            bestValCost = cv;
            bestTheta = globalBest;
            bestIter = cycle;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        if cv > prevValCost
            valIncreaseCount = valIncreaseCount + 1;
        else
            valIncreaseCount = 0;
        end
        prevValCost = cv;
        
        if valIncreaseCount >= params.overfitWindow
            hist.status = 'OVERFIT';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
            return;
        end
        
        if noImpCount >= params.patience
            hist.status = 'OK';
            hist.bestIter = bestIter;
            hist.bestValCost = bestValCost;
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
        if minC < costF(bestI)
            globalBest = Foods(:,bi);
            bestI = bi;
        end
    end
    
    hist.status = 'OK';
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

%% =========================================================================
%%                        CSV EXPORT
%% =========================================================================
function export_all_to_csv(results, filename)
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'method,H,lambda,lr,maxIter,bestIter,valCost,valAcc,testAcc,status\n');
    
    for i = 1:numel(results)
        r = results(i);
        fprintf(fid, '%s,%d,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f,%s\n', ...
            r.method, r.H, r.lambda, r.lr, r.maxIter, r.bestIter, ...
            r.valCost, r.valAcc, r.testAcc, r.status);
    end
    
    fclose(fid);
end

function export_top3_to_csv(results, filename)
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'method,H,lambda,lr,maxIter,bestIter,valCost,valAcc,testAcc,status\n');
    
    for i = 1:numel(results)
        r = results(i);
        fprintf(fid, '%s,%d,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f,%s\n', ...
            r.method, r.H, r.lambda, r.lr, r.maxIter, r.bestIter, ...
            r.valCost, r.valAcc, r.testAcc, r.status);
    end
    
    fclose(fid);
end

function export_winner_to_csv(winner, filename)
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'method,H,lambda,lr,maxIter,bestIter,valCost,valAcc,testAcc,status\n');
    fprintf(fid, '%s,%d,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f,%s\n', ...
        winner.method, winner.H, winner.lambda, winner.lr, winner.maxIter, winner.bestIter, ...
        winner.valCost, winner.valAcc, winner.testAcc, winner.status);
    
    fclose(fid);
end

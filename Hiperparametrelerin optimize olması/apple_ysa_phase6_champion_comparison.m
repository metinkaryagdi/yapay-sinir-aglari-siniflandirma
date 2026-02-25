function apple_ysa_phase6_champion_comparison()
%% =========================================================================
%  PHASE-6: CHAMPION COMPARISON (All Algorithms)
%  
%  Amaç: Her algoritma için overfit-free şampiyon seç ve kapsamlı karşılaştır
%  
%  Aşamalar:
%  1) Şampiyon seçimi (ABC, BFGS, CG, DFP, GD)
%  2) İkili karşılaştırmalar (10 çift × 3 grafik = 30 grafik)
%  3) Ortak karşılaştırma (tüm algoritmalar)
%  4) Özet tablo
% ==========================================================================

    clear; clc; close all;
    
    %% ================== AYARLAR ======================================
    DATA_FILE = 'apple_quality.csv';
    
    % Use absolute path for figures directory
    currentDir = pwd;
    FIGURES_DIR = fullfile(currentDir, 'figures');
    
    N_TRAIN = 2800;
    N_VAL = 600;
    N_TEST = 600;
    
    PATIENCE = 25;
    MIN_DELTA = 1e-4;
    OVERFIT_WINDOW = 15;
    OVERFIT_THRESHOLD = 0.05;
    
    SEEDS = [42, 43, 44];
    
    % Create figures directory
    if ~exist(FIGURES_DIR, 'dir')
        mkdir(FIGURES_DIR);
        fprintf('>> Created directory: %s\n', FIGURES_DIR);
    end
    
    %% ================== ŞAMPİYON KONFİGÜRASYONLARI ===================
    % Her algoritma için overfitRate=0% hedefli konfigürasyonlar
    % BFGS: Phase-5 stabilize (lr=0.027)
    % DFP: Phase-4/5'ten overfitRate=0% veren konfigürasyon
    champions = get_champion_configs();
    
    %% ================== STAGE 1: ŞAMPİYON EĞİTİMİ ====================
    fprintf('========================================\n');
    fprintf('STAGE 1: CHAMPION TRAINING (Multi-Seed)\n');
    fprintf('========================================\n');
    
    allResults = [];
    
    for m = 1:numel(champions)
        champ = champions(m);
        fprintf('\n=== %s ===\n', champ.method);
        fprintf('   Config: H=%d, λ=%.5f, lr=%.4f, maxIter=%d\n', ...
            champ.H, champ.lambda, champ.lr, champ.maxIter);
        
        for s = 1:numel(SEEDS)
            seed = SEEDS(s);
            rng(seed);
            
            [X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = ...
                load_and_split_data(DATA_FILE, N_TRAIN, N_VAL, N_TEST);
            
            job.method = champ.method;
            job.H = champ.H;
            job.lambda = champ.lambda;
            job.lr = champ.lr;
            job.maxIter = champ.maxIter;
            
            % ABC özel parametreler
            if strcmp(champ.method, 'ABC')
                job.SN = 50;
                job.limit = 100;
            end
            
            res = train_algorithm(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
                inputDim, outputDim, PATIENCE, MIN_DELTA, OVERFIT_WINDOW, OVERFIT_THRESHOLD);
            
            res.seed = seed;
            allResults = [allResults; res];
            
            fprintf('   [seed=%d] ValCost=%.4f | ValAcc=%.2f%% | TestAcc=%.2f%% | %s\n', ...
                seed, res.bestValCost, res.bestValAcc*100, res.testAcc*100, res.status);
        end
    end
    
    %% ================== AGGREGATE BY METHOD ==========================
    methods = unique({allResults.method});
    champResults = [];
    
    fprintf('\n========================================\n');
    fprintf('CHAMPION SUMMARY (with overfitRate filtering)\n');
    fprintf('========================================\n');
    fprintf('Algorithm | mean ValCost ± std | mean ValAcc | mean TestAcc | OverfitRate | Status\n');
    fprintf('----------|---------------------|-------------|--------------|-------------|--------\n');
    
    for m = 1:numel(methods)
        methodName = methods{m};
        methodResults = allResults(strcmp({allResults.method}, methodName));
        
        % Aggregate
        agg.method = methodName;
        agg.H = methodResults(1).H;
        agg.lambda = methodResults(1).lambda;
        agg.lr = methodResults(1).lr;
        agg.maxIter = methodResults(1).maxIter;
        
        agg.meanValCost = mean([methodResults.bestValCost]);
        agg.stdValCost = std([methodResults.bestValCost]);
        agg.meanValAcc = mean([methodResults.bestValAcc]);
        agg.meanTestAcc = mean([methodResults.testAcc]);
        agg.meanBestIter = mean([methodResults.bestIter]);
        
        overfitCount = sum(strcmp({methodResults.status}, 'OVERFIT'));
        agg.overfitRate = overfitCount / numel(methodResults) * 100;
        
        % Champion status
        if agg.overfitRate == 0
            agg.championStatus = 'STABLE';
        else
            agg.championStatus = 'No fully-stable champion';
        end
        
        % Store histories for plotting
        agg.histories = methodResults;
        
        champResults = [champResults; agg];
        
        fprintf('%-9s | %.4f ± %.4f     | %.2f%%      | %.2f%%       | %.1f%%       | %s\n', ...
            methodName, agg.meanValCost, agg.stdValCost, ...
            agg.meanValAcc*100, agg.meanTestAcc*100, agg.overfitRate, agg.championStatus);
    end
    
    %% ================== STAGE 2: İKİLİ KARŞILAŞTIRMALAR =============
    fprintf('\n========================================\n');
    fprintf('STAGE 2: PAIRWISE COMPARISONS\n');
    fprintf('========================================\n');
    
    % Tüm ikili kombinasyonlar
    pairs = {};
    for i = 1:numel(methods)
        for j = i+1:numel(methods)
            pairs{end+1} = {methods{i}, methods{j}};
        end
    end
    
    fprintf('Generating %d pairwise comparisons...\n', numel(pairs));
    
    for p = 1:numel(pairs)
        pair = pairs{p};
        method1 = pair{1};
        method2 = pair{2};
        
        fprintf('  [%d/%d] %s vs %s\n', p, numel(pairs), method1, method2);
        
        champ1 = champResults(strcmp({champResults.method}, method1));
        champ2 = champResults(strcmp({champResults.method}, method2));
        
        plot_pairwise_comparison(champ1, champ2, FIGURES_DIR);
    end
    
    %% ================== STAGE 3: ORTAK KARŞILAŞTIRMA ================
    fprintf('\n========================================\n');
    fprintf('STAGE 3: COLLECTIVE COMPARISON\n');
    fprintf('========================================\n');
    
    plot_collective_comparison(champResults, FIGURES_DIR);
    
    %% ================== ÖZET TABLO ===================================
    fprintf('\n========================================\n');
    fprintf('FINAL SUMMARY TABLE\n');
    fprintf('========================================\n');
    fprintf('Algorithm | mean ValCost ± std | mean ValAcc | mean TestAcc | OverfitRate | Status\n');
    fprintf('----------|---------------------|-------------|--------------|-------------|--------\n');
    
    for m = 1:numel(champResults)
        c = champResults(m);
        fprintf('%-9s | %.4f ± %.4f     | %.2f%%      | %.2f%%       | %.1f%%       | %s\n', ...
            c.method, c.meanValCost, c.stdValCost, ...
            c.meanValAcc*100, c.meanTestAcc*100, c.overfitRate, c.championStatus);
    end
    
    fprintf('\n✅ Phase-6 Champion Comparison tamamlandı!\n');
    fprintf('   Grafikler: %s/\n', FIGURES_DIR);
end

%% =========================================================================
%%                        ŞAMPİYON KONFİGÜRASYONLARI
%% =========================================================================
function champions = get_champion_configs()
    champions = [];
    
    % ABC - Phase-3 champion (overfitRate=0%)
    c.method = 'ABC';
    c.H = 8;
    c.lambda = 0.0005;
    c.lr = 0.03;
    c.maxIter = 430;
    champions = [champions; c];
    
    % BFGS - Phase-5 stabilized (lr=0.027 for lower overfitRate)
    c.method = 'BFGS';
    c.H = 16;
    c.lambda = 0.0001;
    c.lr = 0.027;  % Stabilized from Phase-5
    c.maxIter = 430;
    champions = [champions; c];
    
    % CG - Phase-3 champion (overfitRate=0%)
    c.method = 'CG';
    c.H = 8;
    c.lambda = 0.001;
    c.lr = 0.05;
    c.maxIter = 430;
    champions = [champions; c];
    
    % DFP - Adjusted for overfitRate=0% (reduced H from 32 to 16)
    c.method = 'DFP';
    c.H = 16;  % Reduced from 32 to avoid overfitting
    c.lambda = 0.0005;
    c.lr = 0.03;
    c.maxIter = 430;
    champions = [champions; c];
    
    % GD - Phase-3 champion (overfitRate=0%)
    c.method = 'GD';
    c.H = 32;
    c.lambda = 0.0001;
    c.lr = 0.05;
    c.maxIter = 430;
    champions = [champions; c];
end

%% =========================================================================
%%                        DATA LOADING
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
%%                        TRAINING DISPATCHER
%% =========================================================================
function res = train_algorithm(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
    inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold)
    
    switch job.method
        case 'BFGS'
            res = train_bfgs(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
                inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold);
        case 'DFP'
            res = train_dfp(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
                inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold);
        case 'CG'
            res = train_cg(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
                inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold);
        case 'GD'
            res = train_gd(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
                inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold);
        case 'ABC'
            res = train_abc(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
                inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold);
    end
    
    res.method = job.method;
end

%% =========================================================================
%%                        TRAINING FUNCTIONS
%% =========================================================================
function res = train_bfgs(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
    inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold)
    
    net.inputDim = inputDim;
    net.outputDim = outputDim;
    net.H = job.H;
    
    theta = init_weights(net);
    
    n = numel(theta);
    H = eye(n);
    [c, g] = cost_and_grad(theta, net, X_train, T_train, job.lambda);
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    
    % History tracking
    trainCostHist = zeros(1, job.maxIter);
    valCostHist = zeros(1, job.maxIter);
    valAccHist = zeros(1, job.maxIter);
    
    res.status = 'OK';
    res.stoppedIter = job.maxIter;
    
    for k = 1:job.maxIter
        [cv, ~] = forward(theta, net, X_val, T_val, job.lambda);
        [valAcc, ~] = evaluate_net(theta, net, X_val, T_val);
        
        trainCostHist(k) = c;
        valCostHist(k) = cv;
        valAccHist(k) = valAcc;
        
        if cv < bestValCost - minDelta
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
        
        if valIncreaseCount >= overfitWindow || cv > bestValCost * (1 + overfitThreshold)
            res.status = 'OVERFIT';
            res.stoppedIter = k;
            break;
        end
        
        if noImpCount >= patience
            res.stoppedIter = k;
            break;
        end
        
        % BFGS update
        p = -H * g;
        alpha = job.lr;
        for ls = 1:10
            t_new = theta + alpha * p;
            [cn, ~] = cost_and_grad(t_new, net, X_train, T_train, job.lambda);
            if cn < c, break; end
            alpha = alpha * 0.5;
        end
        
        [c_new, g_new] = cost_and_grad(t_new, net, X_train, T_train, job.lambda);
        s = t_new - theta;
        y = g_new - g;
        ys = dot(y, s);
        
        if ys > 1e-10
            rho = 1 / ys;
            V = eye(n) - rho * (y * s.');
            H = V.' * H * V + rho * (s * s.');
        end
        
        theta = t_new;
        c = c_new;
        g = g_new;
    end
    
    [testAcc, ~] = evaluate_net(bestTheta, net, X_test, T_test);
    [valAcc, ~] = evaluate_net(bestTheta, net, X_val, T_val);
    
    res.H = job.H;
    res.lambda = job.lambda;
    res.lr = job.lr;
    res.maxIter = job.maxIter;
    res.bestIter = bestIter;
    res.bestValCost = bestValCost;
    res.bestValAcc = valAcc;
    res.testAcc = testAcc;
    res.trainCostHist = trainCostHist(1:res.stoppedIter);
    res.valCostHist = valCostHist(1:res.stoppedIter);
    res.valAccHist = valAccHist(1:res.stoppedIter);
end

function res = train_dfp(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
    inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold)
    
    net.inputDim = inputDim;
    net.outputDim = outputDim;
    net.H = job.H;
    
    theta = init_weights(net);
    
    n = numel(theta);
    H = eye(n);
    [c, g] = cost_and_grad(theta, net, X_train, T_train, job.lambda);
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    
    trainCostHist = zeros(1, job.maxIter);
    valCostHist = zeros(1, job.maxIter);
    valAccHist = zeros(1, job.maxIter);
    
    res.status = 'OK';
    res.stoppedIter = job.maxIter;
    
    for k = 1:job.maxIter
        [cv, ~] = forward(theta, net, X_val, T_val, job.lambda);
        [valAcc, ~] = evaluate_net(theta, net, X_val, T_val);
        
        trainCostHist(k) = c;
        valCostHist(k) = cv;
        valAccHist(k) = valAcc;
        
        if cv < bestValCost - minDelta
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
        
        if valIncreaseCount >= overfitWindow || cv > bestValCost * (1 + overfitThreshold)
            res.status = 'OVERFIT';
            res.stoppedIter = k;
            break;
        end
        
        if noImpCount >= patience
            res.stoppedIter = k;
            break;
        end
        
        p = -H * g;
        alpha = job.lr;
        for ls = 1:10
            t_new = theta + alpha * p;
            [cn, ~] = cost_and_grad(t_new, net, X_train, T_train, job.lambda);
            if cn < c, break; end
            alpha = alpha * 0.5;
        end
        
        [c_new, g_new] = cost_and_grad(t_new, net, X_train, T_train, job.lambda);
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
    
    [testAcc, ~] = evaluate_net(bestTheta, net, X_test, T_test);
    [valAcc, ~] = evaluate_net(bestTheta, net, X_val, T_val);
    
    res.H = job.H;
    res.lambda = job.lambda;
    res.lr = job.lr;
    res.maxIter = job.maxIter;
    res.bestIter = bestIter;
    res.bestValCost = bestValCost;
    res.bestValAcc = valAcc;
    res.testAcc = testAcc;
    res.trainCostHist = trainCostHist(1:res.stoppedIter);
    res.valCostHist = valCostHist(1:res.stoppedIter);
    res.valAccHist = valAccHist(1:res.stoppedIter);
end

function res = train_cg(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
    inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold)
    
    net.inputDim = inputDim;
    net.outputDim = outputDim;
    net.H = job.H;
    
    theta = init_weights(net);
    [c, g] = cost_and_grad(theta, net, X_train, T_train, job.lambda);
    p = -g;
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    
    trainCostHist = zeros(1, job.maxIter);
    valCostHist = zeros(1, job.maxIter);
    valAccHist = zeros(1, job.maxIter);
    
    res.status = 'OK';
    res.stoppedIter = job.maxIter;
    
    for k = 1:job.maxIter
        [cv, ~] = forward(theta, net, X_val, T_val, job.lambda);
        [valAcc, ~] = evaluate_net(theta, net, X_val, T_val);
        
        trainCostHist(k) = c;
        valCostHist(k) = cv;
        valAccHist(k) = valAcc;
        
        if cv < bestValCost - minDelta
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
        
        if valIncreaseCount >= overfitWindow || cv > bestValCost * (1 + overfitThreshold)
            res.status = 'OVERFIT';
            res.stoppedIter = k;
            break;
        end
        
        if noImpCount >= patience
            res.stoppedIter = k;
            break;
        end
        
        alpha = job.lr;
        for ls = 1:10
            t_new = theta + alpha * p;
            [cn, ~] = cost_and_grad(t_new, net, X_train, T_train, job.lambda);
            if cn < c, break; end
            alpha = alpha * 0.5;
        end
        
        [c_new, g_new] = cost_and_grad(t_new, net, X_train, T_train, job.lambda);
        beta = max(0, dot(g_new, g_new - g) / dot(g, g));
        p = -g_new + beta * p;
        
        theta = t_new;
        c = c_new;
        g = g_new;
    end
    
    [testAcc, ~] = evaluate_net(bestTheta, net, X_test, T_test);
    [valAcc, ~] = evaluate_net(bestTheta, net, X_val, T_val);
    
    res.H = job.H;
    res.lambda = job.lambda;
    res.lr = job.lr;
    res.maxIter = job.maxIter;
    res.bestIter = bestIter;
    res.bestValCost = bestValCost;
    res.bestValAcc = valAcc;
    res.testAcc = testAcc;
    res.trainCostHist = trainCostHist(1:res.stoppedIter);
    res.valCostHist = valCostHist(1:res.stoppedIter);
    res.valAccHist = valAccHist(1:res.stoppedIter);
end

function res = train_gd(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
    inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold)
    
    net.inputDim = inputDim;
    net.outputDim = outputDim;
    net.H = job.H;
    
    theta = init_weights(net);
    
    bestValCost = Inf;
    bestTheta = theta;
    bestIter = 0;
    noImpCount = 0;
    valIncreaseCount = 0;
    prevValCost = Inf;
    
    trainCostHist = zeros(1, job.maxIter);
    valCostHist = zeros(1, job.maxIter);
    valAccHist = zeros(1, job.maxIter);
    
    res.status = 'OK';
    res.stoppedIter = job.maxIter;
    
    for k = 1:job.maxIter
        [c, g] = cost_and_grad(theta, net, X_train, T_train, job.lambda);
        [cv, ~] = forward(theta, net, X_val, T_val, job.lambda);
        [valAcc, ~] = evaluate_net(theta, net, X_val, T_val);
        
        trainCostHist(k) = c;
        valCostHist(k) = cv;
        valAccHist(k) = valAcc;
        
        if cv < bestValCost - minDelta
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
        
        if valIncreaseCount >= overfitWindow || cv > bestValCost * (1 + overfitThreshold)
            res.status = 'OVERFIT';
            res.stoppedIter = k;
            break;
        end
        
        if noImpCount >= patience
            res.stoppedIter = k;
            break;
        end
        
        theta = theta - job.lr * g;
    end
    
    [testAcc, ~] = evaluate_net(bestTheta, net, X_test, T_test);
    [valAcc, ~] = evaluate_net(bestTheta, net, X_val, T_val);
    
    res.H = job.H;
    res.lambda = job.lambda;
    res.lr = job.lr;
    res.maxIter = job.maxIter;
    res.bestIter = bestIter;
    res.bestValCost = bestValCost;
    res.bestValAcc = valAcc;
    res.testAcc = testAcc;
    res.trainCostHist = trainCostHist(1:res.stoppedIter);
    res.valCostHist = valCostHist(1:res.stoppedIter);
    res.valAccHist = valAccHist(1:res.stoppedIter);
end

function res = train_abc(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
    inputDim, outputDim, patience, minDelta, overfitWindow, overfitThreshold)
    
    net.inputDim = inputDim;
    net.outputDim = outputDim;
    net.H = job.H;
    
    theta = init_weights(net);
    
    D = numel(theta);
    SN = job.SN;
    limit = job.limit;
    
    Foods = repmat(theta, 1, SN) + randn(D, SN) * 0.1;
    costF = zeros(1, SN);
    for i = 1:SN
        [costF(i), ~] = cost_and_grad(Foods(:,i), net, X_train, T_train, job.lambda);
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
    
    trainCostHist = zeros(1, job.maxIter);
    valCostHist = zeros(1, job.maxIter);
    valAccHist = zeros(1, job.maxIter);
    
    res.status = 'OK';
    res.stoppedIter = job.maxIter;
    
    for cycle = 1:job.maxIter
        [c, ~] = cost_and_grad(globalBest, net, X_train, T_train, job.lambda);
        [cv, ~] = forward(globalBest, net, X_val, T_val, job.lambda);
        [valAcc, ~] = evaluate_net(globalBest, net, X_val, T_val);
        
        trainCostHist(cycle) = c;
        valCostHist(cycle) = cv;
        valAccHist(cycle) = valAcc;
        
        if cv < bestValCost - minDelta
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
        
        if valIncreaseCount >= overfitWindow || cv > bestValCost * (1 + overfitThreshold)
            res.status = 'OVERFIT';
            res.stoppedIter = cycle;
            break;
        end
        
        if noImpCount >= patience
            res.stoppedIter = cycle;
            break;
        end
        
        % Employed Bees
        for i = 1:SN
            k = randi(SN);
            while k == i, k = randi(SN); end
            phi = (rand(D, 1) * 2 - 1);
            sol = Foods(:,i) + phi .* (Foods(:,i) - Foods(:,k));
            [cNew, ~] = cost_and_grad(sol, net, X_train, T_train, job.lambda);
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
            Foods(:,ind) = randn(D, 1) * 0.1;
            [costF(ind), ~] = cost_and_grad(Foods(:,ind), net, X_train, T_train, job.lambda);
            trial(ind) = 0;
        end
        
        [minC, bi] = min(costF);
        if minC < costF(bestI)
            globalBest = Foods(:,bi);
            bestI = bi;
        end
    end
    
    [testAcc, ~] = evaluate_net(bestTheta, net, X_test, T_test);
    [valAcc, ~] = evaluate_net(bestTheta, net, X_val, T_val);
    
    res.H = job.H;
    res.lambda = job.lambda;
    res.lr = job.lr;
    res.maxIter = job.maxIter;
    res.bestIter = bestIter;
    res.bestValCost = bestValCost;
    res.bestValAcc = valAcc;
    res.testAcc = testAcc;
    res.trainCostHist = trainCostHist(1:res.stoppedIter);
    res.valCostHist = valCostHist(1:res.stoppedIter);
    res.valAccHist = valAccHist(1:res.stoppedIter);
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
%%                        PLOTTING FUNCTIONS
%% =========================================================================
function plot_pairwise_comparison(champ1, champ2, figDir)
    % Extract histories
    hist1 = champ1.histories;
    hist2 = champ2.histories;
    
    % Compute maxLen across BOTH algorithms
    maxLen1 = max([hist1.stoppedIter]);
    maxLen2 = max([hist2.stoppedIter]);
    maxLen = max(maxLen1, maxLen2);
    
    % Pad histories to same length
    valCost1_all = nan(numel(hist1), maxLen);
    valAcc1_all = nan(numel(hist1), maxLen);
    trainCost1_all = nan(numel(hist1), maxLen);
    
    for i = 1:numel(hist1)
        len = numel(hist1(i).valCostHist);
        valCost1_all(i, 1:len) = hist1(i).valCostHist;
        valAcc1_all(i, 1:len) = hist1(i).valAccHist;
        trainCost1_all(i, 1:len) = hist1(i).trainCostHist;
    end
    
    valCost2_all = nan(numel(hist2), maxLen);
    valAcc2_all = nan(numel(hist2), maxLen);
    trainCost2_all = nan(numel(hist2), maxLen);
    
    for i = 1:numel(hist2)
        len = numel(hist2(i).valCostHist);
        valCost2_all(i, 1:len) = hist2(i).valCostHist;
        valAcc2_all(i, 1:len) = hist2(i).valAccHist;
        trainCost2_all(i, 1:len) = hist2(i).trainCostHist;
    end
    
    % Mean and std
    valCost1_mean = nanmean(valCost1_all, 1);
    valCost1_std = nanstd(valCost1_all, 0, 1);
    valAcc1_mean = nanmean(valAcc1_all, 1);
    trainCost1_mean = nanmean(trainCost1_all, 1);
    
    valCost2_mean = nanmean(valCost2_all, 1);
    valCost2_std = nanstd(valCost2_all, 0, 1);
    valAcc2_mean = nanmean(valAcc2_all, 1);
    trainCost2_mean = nanmean(trainCost2_all, 1);
    
    iters = 1:maxLen;
    
    % Plot 1: ValCost vs Iteration
    figure('Position', [100 100 800 600], 'Color', 'w');
    hold on; grid on;
    
    plot(iters, valCost1_mean, 'b-', 'LineWidth', 2, 'DisplayName', champ1.method);
    fill([iters, fliplr(iters)], ...
        [valCost1_mean - valCost1_std, fliplr(valCost1_mean + valCost1_std)], ...
        'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    plot(iters, valCost2_mean, 'r-', 'LineWidth', 2, 'DisplayName', champ2.method);
    fill([iters, fliplr(iters)], ...
        [valCost2_mean - valCost2_std, fliplr(valCost2_mean + valCost2_std)], ...
        'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Validation Cost', 'FontSize', 12);
    title(sprintf('%s vs %s: Validation Cost', champ1.method, champ2.method), 'FontSize', 14);
    legend('Location', 'best', 'FontSize', 11);
    
    saveas(gcf, fullfile(figDir, sprintf('%s_vs_%s_ValCost.png', champ1.method, champ2.method)));
    close(gcf);
    
    % Plot 2: ValAcc vs Iteration
    figure('Position', [100 100 800 600], 'Color', 'w');
    hold on; grid on;
    
    plot(iters, valAcc1_mean * 100, 'b-', 'LineWidth', 2, 'DisplayName', champ1.method);
    plot(iters, valAcc2_mean * 100, 'r-', 'LineWidth', 2, 'DisplayName', champ2.method);
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Validation Accuracy (%)', 'FontSize', 12);
    title(sprintf('%s vs %s: Validation Accuracy', champ1.method, champ2.method), 'FontSize', 14);
    legend('Location', 'best', 'FontSize', 11);
    
    saveas(gcf, fullfile(figDir, sprintf('%s_vs_%s_ValAcc.png', champ1.method, champ2.method)));
    close(gcf);
    
    % Plot 3: Overfitting Gap
    figure('Position', [100 100 800 600], 'Color', 'w');
    hold on; grid on;
    
    gap1 = abs(trainCost1_mean - valCost1_mean);
    gap2 = abs(trainCost2_mean - valCost2_mean);
    
    plot(iters, gap1, 'b-', 'LineWidth', 2, 'DisplayName', champ1.method);
    plot(iters, gap2, 'r-', 'LineWidth', 2, 'DisplayName', champ2.method);
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('|TrainCost - ValCost|', 'FontSize', 12);
    title(sprintf('%s vs %s: Overfitting Gap', champ1.method, champ2.method), 'FontSize', 14);
    legend('Location', 'best', 'FontSize', 11);
    
    saveas(gcf, fullfile(figDir, sprintf('%s_vs_%s_OverfitGap.png', champ1.method, champ2.method)));
    close(gcf);
end

function plot_collective_comparison(champResults, figDir)
    % Plot 1: All ValCost
    figure('Position', [100 100 1000 600], 'Color', 'w');
    hold on; grid on;
    
    colors = lines(numel(champResults));
    maxLen = 0;
    
    for m = 1:numel(champResults)
        hist = champResults(m).histories;
        maxLen = max(maxLen, max([hist.stoppedIter]));
    end
    
    for m = 1:numel(champResults)
        hist = champResults(m).histories;
        
        valCost_all = nan(numel(hist), maxLen);
        for i = 1:numel(hist)
            len = numel(hist(i).valCostHist);
            valCost_all(i, 1:len) = hist(i).valCostHist;
        end
        
        valCost_mean = nanmean(valCost_all, 1);
        
        plot(1:maxLen, valCost_mean, 'LineWidth', 2, 'Color', colors(m,:), ...
            'DisplayName', sprintf('%s (%.4f)', champResults(m).method, champResults(m).meanValCost));
    end
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Validation Cost', 'FontSize', 12);
    title('All Algorithms: Validation Cost Comparison', 'FontSize', 14);
    legend('Location', 'best', 'FontSize', 10);
    
    saveas(gcf, fullfile(figDir, 'All_Algorithms_ValCost.png'));
    close(gcf);
    
    % Plot 2: All ValAcc
    figure('Position', [100 100 1000 600], 'Color', 'w');
    hold on; grid on;
    
    for m = 1:numel(champResults)
        hist = champResults(m).histories;
        
        valAcc_all = nan(numel(hist), maxLen);
        for i = 1:numel(hist)
            len = numel(hist(i).valAccHist);
            valAcc_all(i, 1:len) = hist(i).valAccHist;
        end
        
        valAcc_mean = nanmean(valAcc_all, 1);
        
        plot(1:maxLen, valAcc_mean * 100, 'LineWidth', 2, 'Color', colors(m,:), ...
            'DisplayName', sprintf('%s (%.2f%%)', champResults(m).method, champResults(m).meanValAcc*100));
    end
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Validation Accuracy (%)', 'FontSize', 12);
    title('All Algorithms: Validation Accuracy Comparison', 'FontSize', 14);
    legend('Location', 'best', 'FontSize', 10);
    
    saveas(gcf, fullfile(figDir, 'All_Algorithms_ValAcc.png'));
    close(gcf);
    
    % Plot 3: Barplot with error bars
    figure('Position', [100 100 800 600], 'Color', 'w');
    
    methods = {champResults.method};
    meanVals = [champResults.meanValCost];
    stdVals = [champResults.stdValCost];
    
    b = bar(meanVals, 'FaceColor', 'flat');
    b.CData = colors(1:numel(champResults), :);
    
    hold on;
    errorbar(1:numel(champResults), meanVals, stdVals, 'k.', 'LineWidth', 1.5);
    
    set(gca, 'XTickLabel', methods);
    ylabel('Mean Validation Cost', 'FontSize', 12);
    title('Mean Validation Cost (±std) by Algorithm', 'FontSize', 14);
    grid on;
    
    saveas(gcf, fullfile(figDir, 'All_Algorithms_ValCost_Barplot.png'));
    close(gcf);
end

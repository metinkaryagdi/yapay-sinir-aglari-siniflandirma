function apple_ysa_phase5_micro()
%% =========================================================================
%  PHASE-5: MICRO FINE-TUNING (Multi-Seed Validation)
%  
%  Winner: BFGS, H=16, λ=0.00010, lr=0.0300, maxIter=500, best@iter=416
%  
%  Plan:
%  A) Winner repeat (3 seeds)
%  B) Micro grid (lr×λ, 9 configs, 3 seeds each)
%  C) Optional H scan (top 2 configs)
%  
%  Output: Mean±std, top 5, final recommendation
% ==========================================================================

    clear; clc; close all;
    
    %% ================== AYARLAR ======================================
    DATA_FILE = 'apple_quality.csv';
    CSV_ALL = 'results_phase5_all.csv';
    CSV_TOP5 = 'results_phase5_top5.csv';
    CSV_REC = 'results_phase5_recommendation.csv';
    
    N_TRAIN = 2800;
    N_VAL = 600;
    N_TEST = 600;
    
    PATIENCE = 25;
    MIN_DELTA = 1e-4;
    OVERFIT_WINDOW = 15;
    OVERFIT_THRESHOLD = 0.05;  % 5% degradation
    
    SEEDS = [42, 43, 44];
    
    % Winner config
    WINNER_H = 16;
    WINNER_LAMBDA = 0.00010;
    WINNER_LR = 0.0300;
    WINNER_MAXITER = 430;  % 416 + margin
    
    %% ================== A) WINNER REPEAT =============================
    fprintf('========================================\n');
    fprintf('A) WINNER REPEAT (3 seeds)\n');
    fprintf('========================================\n');
    
    winnerResults = [];
    
    for s = 1:numel(SEEDS)
        seed = SEEDS(s);
        rng(seed);
        
        [X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = ...
            load_and_split_data(DATA_FILE, N_TRAIN, N_VAL, N_TEST);
        
        job.H = WINNER_H;
        job.lambda = WINNER_LAMBDA;
        job.lr = WINNER_LR;
        job.maxIter = WINNER_MAXITER;
        
        res = run_bfgs(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
            inputDim, outputDim, PATIENCE, MIN_DELTA, OVERFIT_WINDOW, OVERFIT_THRESHOLD);
        
        res.seed = seed;
        res.config = 'WINNER';
        winnerResults = [winnerResults; res];
        
        fprintf('[BFGS] seed=%d | H=%d λ=%.5f lr=%.4f maxIter=%d | bestValCost=%.4f bestValAcc=%.2f%% TestAcc=%.2f%% | best@iter=%d stopped@iter=%d | %s\n', ...
            seed, job.H, job.lambda, job.lr, job.maxIter, ...
            res.bestValCost, res.bestValAcc*100, res.testAcc*100, ...
            res.bestIter, res.stoppedIter, res.status);
    end
    
    %% ================== B) MICRO GRID ================================
    fprintf('\n========================================\n');
    fprintf('B) MICRO GRID (lr×λ, 9 configs, 3 seeds)\n');
    fprintf('========================================\n');
    
    lrGrid = [0.0270, 0.0300, 0.0330];
    lambdaGrid = [0.00008, 0.00010, 0.00012];
    
    gridResults = [];
    
    for lr_idx = 1:numel(lrGrid)
        for lam_idx = 1:numel(lambdaGrid)
            lr = lrGrid(lr_idx);
            lam = lambdaGrid(lam_idx);
            
            fprintf('\n--- Config: lr=%.4f, λ=%.5f ---\n', lr, lam);
            
            for s = 1:numel(SEEDS)
                seed = SEEDS(s);
                rng(seed);
                
                [X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = ...
                    load_and_split_data(DATA_FILE, N_TRAIN, N_VAL, N_TEST);
                
                job.H = WINNER_H;
                job.lambda = lam;
                job.lr = lr;
                job.maxIter = WINNER_MAXITER;
                
                res = run_bfgs(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
                    inputDim, outputDim, PATIENCE, MIN_DELTA, OVERFIT_WINDOW, OVERFIT_THRESHOLD);
                
                res.seed = seed;
                res.config = sprintf('lr%.4f_lam%.5f', lr, lam);
                gridResults = [gridResults; res];
                
                fprintf('[BFGS] seed=%d | H=%d λ=%.5f lr=%.4f maxIter=%d | bestValCost=%.4f bestValAcc=%.2f%% TestAcc=%.2f%% | best@iter=%d stopped@iter=%d | %s\n', ...
                    seed, job.H, job.lambda, job.lr, job.maxIter, ...
                    res.bestValCost, res.bestValAcc*100, res.testAcc*100, ...
                    res.bestIter, res.stoppedIter, res.status);
            end
        end
    end
    
    %% ================== AGGREGATE RESULTS ============================
    allResults = [winnerResults; gridResults];
    
    % Group by config
    configs = unique({allResults.config});
    aggResults = [];
    
    for c = 1:numel(configs)
        cfg = configs{c};
        cfgResults = allResults(strcmp({allResults.config}, cfg));
        
        agg.config = cfg;
        agg.H = cfgResults(1).H;
        agg.lambda = cfgResults(1).lambda;
        agg.lr = cfgResults(1).lr;
        agg.maxIter = cfgResults(1).maxIter;
        
        % Mean & std
        agg.meanValCost = mean([cfgResults.bestValCost]);
        agg.stdValCost = std([cfgResults.bestValCost]);
        agg.meanValAcc = mean([cfgResults.bestValAcc]);
        agg.meanTestAcc = mean([cfgResults.testAcc]);
        agg.meanBestIter = mean([cfgResults.bestIter]);
        
        % Overfit rate
        overfitCount = sum(strcmp({cfgResults.status}, 'OVERFIT'));
        agg.overfitRate = overfitCount / numel(cfgResults) * 100;
        
        aggResults = [aggResults; agg];
    end
    
    %% ================== C) OPTIONAL H SCAN ===========================
    % Select top 2 configs by meanValCost
    [~, sortIdx] = sort([aggResults.meanValCost]);
    top2Configs = aggResults(sortIdx(1:min(2, numel(aggResults))));
    
    fprintf('\n========================================\n');
    fprintf('C) OPTIONAL H SCAN (top 2 configs)\n');
    fprintf('========================================\n');
    
    HGrid = [12, 16, 20];
    hScanResults = [];
    
    for t = 1:numel(top2Configs)
        topCfg = top2Configs(t);
        fprintf('\n--- Base: lr=%.4f, λ=%.5f ---\n', topCfg.lr, topCfg.lambda);
        
        for h_idx = 1:numel(HGrid)
            H = HGrid(h_idx);
            
            if H == topCfg.H
                continue;  % Already tested
            end
            
            fprintf('  Testing H=%d\n', H);
            
            for s = 1:numel(SEEDS)
                seed = SEEDS(s);
                rng(seed);
                
                [X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = ...
                    load_and_split_data(DATA_FILE, N_TRAIN, N_VAL, N_TEST);
                
                job.H = H;
                job.lambda = topCfg.lambda;
                job.lr = topCfg.lr;
                job.maxIter = WINNER_MAXITER;
                
                res = run_bfgs(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
                    inputDim, outputDim, PATIENCE, MIN_DELTA, OVERFIT_WINDOW, OVERFIT_THRESHOLD);
                
                res.seed = seed;
                res.config = sprintf('lr%.4f_lam%.5f_H%d', topCfg.lr, topCfg.lambda, H);
                hScanResults = [hScanResults; res];
                
                fprintf('    [BFGS] seed=%d | H=%d λ=%.5f lr=%.4f | bestValCost=%.4f | %s\n', ...
                    seed, job.H, job.lambda, job.lr, res.bestValCost, res.status);
            end
        end
    end
    
    % Re-aggregate with H scan
    if ~isempty(hScanResults)
        allResults = [allResults; hScanResults];
        
        configs = unique({allResults.config});
        aggResults = [];
        
        for c = 1:numel(configs)
            cfg = configs{c};
            cfgResults = allResults(strcmp({allResults.config}, cfg));
            
            agg.config = cfg;
            agg.H = cfgResults(1).H;
            agg.lambda = cfgResults(1).lambda;
            agg.lr = cfgResults(1).lr;
            agg.maxIter = cfgResults(1).maxIter;
            
            agg.meanValCost = mean([cfgResults.bestValCost]);
            agg.stdValCost = std([cfgResults.bestValCost]);
            agg.meanValAcc = mean([cfgResults.bestValAcc]);
            agg.meanTestAcc = mean([cfgResults.testAcc]);
            agg.meanBestIter = mean([cfgResults.bestIter]);
            
            overfitCount = sum(strcmp({cfgResults.status}, 'OVERFIT'));
            agg.overfitRate = overfitCount / numel(cfgResults) * 100;
            
            aggResults = [aggResults; agg];
        end
    end
    
    %% ================== PHASE-5 SUMMARY ==============================
    fprintf('\n========================================\n');
    fprintf('PHASE-5 SUMMARY\n');
    fprintf('========================================\n');
    
    % Winner repeat
    fprintf('\n1) WINNER REPEAT (3 seeds):\n');
    fprintf('   Seed | ValCost | ValAcc  | TestAcc | best@iter | stopped@iter | Status\n');
    fprintf('   -----|---------|---------|---------|-----------|--------------|--------\n');
    for i = 1:numel(winnerResults)
        r = winnerResults(i);
        fprintf('   %2d   | %.4f  | %.2f%%  | %.2f%%  | %3d       | %3d          | %s\n', ...
            r.seed, r.bestValCost, r.bestValAcc*100, r.testAcc*100, ...
            r.bestIter, r.stoppedIter, r.status);
    end
    
    % Top 5
    [~, sortIdx] = sort([aggResults.meanValCost]);
    top5 = aggResults(sortIdx(1:min(5, numel(aggResults))));
    
    fprintf('\n2) TOP 5 (by meanValCost):\n');
    fprintf('   Rank | meanValCost±std | meanValAcc | meanTestAcc | lr     | lambda  | H  | meanBest@iter | overfitRate\n');
    fprintf('   -----|-----------------|------------|-------------|--------|---------|----|--------------|-----------\n');
    
    for i = 1:numel(top5)
        t = top5(i);
        fprintf('   #%d   | %.4f±%.4f   | %.2f%%     | %.2f%%      | %.4f | %.5f | %2d | %.1f         | %.1f%%\n', ...
            i, t.meanValCost, t.stdValCost, t.meanValAcc*100, t.meanTestAcc*100, ...
            t.lr, t.lambda, t.H, t.meanBestIter, t.overfitRate);
    end
    
    % Final recommendation
    fprintf('\n3) FINAL RECOMMENDATION:\n');
    
    % Check if top 2 are very close
    if numel(top5) >= 2
        diff = abs(top5(1).meanValCost - top5(2).meanValCost);
        if diff < 0.005
            % Choose lower std
            if top5(2).stdValCost < top5(1).stdValCost
                recommended = top5(2);
                fprintf('   Top 2 very close (diff=%.4f), selecting lower std.\n', diff);
            else
                recommended = top5(1);
            end
        else
            recommended = top5(1);
        end
    else
        recommended = top5(1);
    end
    
    fprintf('\n   🏆 RECOMMENDED CONFIG:\n');
    fprintf('      H=%d, λ=%.5f, lr=%.4f, maxIter=%d\n', ...
        recommended.H, recommended.lambda, recommended.lr, recommended.maxIter);
    fprintf('      meanValCost=%.4f±%.4f | meanValAcc=%.2f%% | meanTestAcc=%.2f%%\n', ...
        recommended.meanValCost, recommended.stdValCost, ...
        recommended.meanValAcc*100, recommended.meanTestAcc*100);
    fprintf('      meanBest@iter=%.1f | overfitRate=%.1f%%\n', ...
        recommended.meanBestIter, recommended.overfitRate);
    
    %% ================== CSV EXPORT ===================================
    fprintf('\n========================================\n');
    fprintf('CSV EXPORT\n');
    fprintf('========================================\n');
    
    export_all_to_csv(allResults, CSV_ALL);
    export_top5_to_csv(top5, CSV_TOP5);
    export_recommendation_to_csv(recommended, CSV_REC);
    
    fprintf('   %s\n', CSV_ALL);
    fprintf('   %s\n', CSV_TOP5);
    fprintf('   %s\n', CSV_REC);
    
    fprintf('\n✅ Phase-5 Micro Fine-Tuning tamamlandı!\n');
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
%%                        BFGS TRAINING
%% =========================================================================
function res = run_bfgs(job, X_train, T_train, X_val, T_val, X_test, T_test, ...
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
    
    % Initialize res struct
    res.status = 'OK';
    res.stoppedIter = job.maxIter;
    
    for k = 1:job.maxIter
        [cv, ~] = forward(theta, net, X_val, T_val, job.lambda);
        
        % Best tracking
        if cv < bestValCost - minDelta
            bestValCost = cv;
            bestTheta = theta;
            bestIter = k;
            noImpCount = 0;
        else
            noImpCount = noImpCount + 1;
        end
        
        % Overfit detection (consecutive increases)
        if cv > prevValCost
            valIncreaseCount = valIncreaseCount + 1;
        else
            valIncreaseCount = 0;
        end
        prevValCost = cv;
        
        % Overfit flag
        if valIncreaseCount >= overfitWindow
            res.status = 'OVERFIT';
            res.stoppedIter = k;
            break;
        end
        
        % Degradation check
        if cv > bestValCost * (1 + overfitThreshold)
            res.status = 'OVERFIT';
            res.stoppedIter = k;
            break;
        end
        
        % Early stopping
        if noImpCount >= patience
            res.status = 'OK';
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
    
    % If loop completed without break
    if ~isfield(res, 'status')
        res.status = 'OK';
        res.stoppedIter = job.maxIter;
    end
    
    % Final evaluation
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
%%                        CSV EXPORT
%% =========================================================================
function export_all_to_csv(results, filename)
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'config,seed,H,lambda,lr,maxIter,bestIter,stoppedIter,bestValCost,bestValAcc,testAcc,status\n');
    
    for i = 1:numel(results)
        r = results(i);
        fprintf(fid, '%s,%d,%d,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%.6f,%s\n', ...
            r.config, r.seed, r.H, r.lambda, r.lr, r.maxIter, ...
            r.bestIter, r.stoppedIter, r.bestValCost, r.bestValAcc, r.testAcc, r.status);
    end
    
    fclose(fid);
end

function export_top5_to_csv(top5, filename)
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'rank,config,H,lambda,lr,maxIter,meanValCost,stdValCost,meanValAcc,meanTestAcc,meanBestIter,overfitRate\n');
    
    for i = 1:numel(top5)
        t = top5(i);
        fprintf(fid, '%d,%s,%d,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f\n', ...
            i, t.config, t.H, t.lambda, t.lr, t.maxIter, ...
            t.meanValCost, t.stdValCost, t.meanValAcc, t.meanTestAcc, ...
            t.meanBestIter, t.overfitRate);
    end
    
    fclose(fid);
end

function export_recommendation_to_csv(rec, filename)
    fid = fopen(filename, 'w');
    
    fprintf(fid, 'config,H,lambda,lr,maxIter,meanValCost,stdValCost,meanValAcc,meanTestAcc,meanBestIter,overfitRate\n');
    fprintf(fid, '%s,%d,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f\n', ...
        rec.config, rec.H, rec.lambda, rec.lr, rec.maxIter, ...
        rec.meanValCost, rec.stdValCost, rec.meanValAcc, rec.meanTestAcc, ...
        rec.meanBestIter, rec.overfitRate);
    
    fclose(fid);
end

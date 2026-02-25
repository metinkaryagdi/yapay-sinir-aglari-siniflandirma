
function apple_ysa_30run_runner_FIXED2_FULL()
% ====================================================
%  APPLE YSA - 30 RUN ISTATISTIK + BOXPLOT (5 METHOD)
%  FULL VERSION (FIX2 + EXTRA ANALYSIS PACK)
%
%  FIX2:
%   - CSV missing (NaN/Inf) temizleme -> cost NaN patlaması biter
%   - Trainer'larda bestValCost başlangıç init (iter=0) -> BestVa NaN biter
%   - NaN/Inf guard: break yerine skip/continue
%
%  EXTRA:
%   - TrainAcc saklama
%   - Genelleme Gap (Train-Val) boxplot
%   - 95% bootstrap CI (TestAcc mean)
%   - Paired Wilcoxon + paired t-test + Cohen d (best metoda karşı)
%   - Multi-metric rank table (Test↑ Val↑ Cost↓ Time↓ Gap↓)
%
%  Dosya: apple_quality.csv (Quality hedef sütunu)
% ====================================================

clear; clc; close all;

fprintf('====================================================\n');
fprintf('  APPLE YSA - 30 RUN ISTATISTIK + BOXPLOT (5 METHOD)\n');
fprintf('  FULL (FIX2 + EXTRA ANALYSIS)\n');
fprintf('====================================================\n\n');

%% ================= CONFIG =================
config = struct();
config.activation = 'tanh';
config.patience   = 50;
config.totalRuns  = 30;

config.BFGS.hiddenSizes = [16];
config.DFP.hiddenSizes  = [32];
config.CG.hiddenSizes   = [16];
config.GD.hiddenSizes   = [16];
config.ABC.hiddenSizes  = [32];

config.BFGS.lambda = 1e-4;
config.DFP.lambda  = 1e-3;
config.CG.lambda   = 5e-4;
config.GD.lambda   = 2e-5;
config.ABC.lambda  = 5e-4;

config.BFGS.lr      = 0.010;  config.BFGS.maxIter  = 250;
config.DFP.lr       = 0.003;  config.DFP.maxIter   = 500;
config.CG.lr        = 0.001;  config.CG.maxIter    = 120;
config.GD.lr        = 0.200;  config.GD.maxIter    = 1000;

config.ABC.SN       = 50;
config.ABC.limit    = 100;
config.ABC.maxCycle = 500;

config.filename = 'apple_quality.csv';
baseSeed = 42;

methods = {'BFGS','DFP','CG','GD','ABC'};

fprintf(">> Config:\n");
fprintf("   Activation: %s\n", config.activation);
fprintf("   Patience  : %d\n", config.patience);
fprintf("   Total Runs: %d\n\n", config.totalRuns);

%% ================= GPU =================
useGPU = false;
try
    if gpuDeviceCount > 0
        gpuDevice;
        useGPU = true;
        fprintf('>> GPU aktif.\n\n');
    else
        fprintf('>> GPU yok, CPU.\n\n');
    end
catch
    fprintf('>> GPU kontrol hatasi, CPU.\n\n');
end

%% ================= RESULT BUCKETS =================
R = struct();
for m = 1:numel(methods)
    name = methods{m};
    R.(name).TrainAcc = nan(config.totalRuns,1);
    R.(name).TestAcc  = nan(config.totalRuns,1);
    R.(name).ValAcc   = nan(config.totalRuns,1);
    R.(name).BestVa   = nan(config.totalRuns,1);
    R.(name).BestIt   = nan(config.totalRuns,1);
    R.(name).Time     = nan(config.totalRuns,1);
    R.(name).Failed   = false(config.totalRuns,1);
end

%% ================= MAIN LOOP =================
for run = 1:config.totalRuns
    seed = baseSeed + run;
    rng(seed);

    fprintf('------------------------------\n');
    fprintf('RUN %02d / %02d (seed=%d)\n', run, config.totalRuns, seed);
    fprintf('------------------------------\n');

    [X_train, T_train, X_val, T_val, X_test, T_test, inD, outD] = load_data_fixed(config.filename);

    if useGPU
        Xtr = gpuArray(single(X_train)); Ttr = gpuArray(single(T_train));
        Xva = gpuArray(single(X_val));   Tva = gpuArray(single(T_val));
        Xte = gpuArray(single(X_test));  Tte = gpuArray(single(T_test));
    else
        Xtr = X_train; Ttr = T_train;
        Xva = X_val;   Tva = T_val;
        Xte = X_test;  Tte = T_test;
    end

    for mi = 1:numel(methods)
        methodName = methods{mi};

        net = struct();
        net.layers = [inD, config.(methodName).hiddenSizes, outD];
        net.activation = config.activation;
        net.L = numel(net.layers);

        theta_init = init_params(net);

        t0 = tic;
        try
            switch methodName
                case 'BFGS'
                    [bestTheta, hist] = train_bfgs_es(theta_init, net, Xtr, Ttr, Xva, Tva, Xte, Tte, ...
                        config.BFGS.lambda, config.BFGS.maxIter, config.BFGS.lr, config.patience);
                case 'DFP'
                    [bestTheta, hist] = train_dfp_es(theta_init, net, Xtr, Ttr, Xva, Tva, Xte, Tte, ...
                        config.DFP.lambda, config.DFP.maxIter, config.DFP.lr, config.patience);
                case 'CG'
                    [bestTheta, hist] = train_cg_es(theta_init, net, Xtr, Ttr, Xva, Tva, Xte, Tte, ...
                        config.CG.lambda, config.CG.maxIter, config.CG.lr, config.patience);
                case 'GD'
                    [bestTheta, hist] = train_gd_es(theta_init, net, Xtr, Ttr, Xva, Tva, Xte, Tte, ...
                        config.GD.lambda, config.GD.maxIter, config.GD.lr, config.patience);
                case 'ABC'
                    [bestTheta, hist] = train_abc_es(theta_init, net, Xtr, Ttr, Xva, Tva, Xte, Tte, ...
                        config.ABC.lambda, config.ABC.SN, config.ABC.limit, config.ABC.maxCycle, config.patience);
            end
        catch ME
            elapsed = toc(t0);
            R.(methodName).Failed(run) = true;
            R.(methodName).Time(run)   = elapsed;
            fprintf('  %s  | FAILED: %s\n', methodName, ME.message);
            continue;
        end
        elapsed = toc(t0);

        [trainAcc, ~] = evaluate_net(bestTheta, net, Xtr, Ttr);
        [valAcc,   ~] = evaluate_net(bestTheta, net, Xva, Tva);
        [testAcc,  ~] = evaluate_net(bestTheta, net, Xte, Tte);

        bestVa = hist.bestValCost;
        if ~isfinite(bestVa)
            bestVa = min(hist.va, [], 'omitnan');
        end
        if ~isfinite(bestVa)
            bestVa = compute_cost(bestTheta, net, Xva, Tva, config.(methodName).lambda);
            bestVa = gather(bestVa);
        end

        R.(methodName).TrainAcc(run) = double(trainAcc);
        R.(methodName).TestAcc(run)  = double(testAcc);
        R.(methodName).ValAcc(run)   = double(valAcc);
        R.(methodName).BestVa(run)   = double(bestVa);
        R.(methodName).BestIt(run)   = double(hist.bestIter);
        R.(methodName).Time(run)     = double(elapsed);

        fprintf('  %-4s | TestAcc=%6.2f%% | ValAcc=%6.2f%% | BestVa=%.4f | BestIt=%4d | %4.2fs\n', ...
            methodName, 100*R.(methodName).TestAcc(run), 100*R.(methodName).ValAcc(run), ...
            R.(methodName).BestVa(run), R.(methodName).BestIt(run), R.(methodName).Time(run));
    end

    fprintf('\n');
end

%% ================= SUMMARY =================
fprintf('\n====================================================\n');
fprintf('                 30 RUN SUMMARY (TEST)\n');
fprintf('====================================================\n');
fprintf('METH  | TrainAcc mean±std  | TestAcc mean±std   | ValAcc mean±std    | BestVa mean | Time mean | Gap mean | Fail%%\n');
fprintf('-----------------------------------------------------------------------------------------------------------------\n');

for mi = 1:numel(methods)
    name = methods{mi};
    tr = R.(name).TrainAcc; ta = R.(name).TestAcc; va = R.(name).ValAcc; bv = R.(name).BestVa; tt = R.(name).Time; ff = R.(name).Failed;
    gap = tr - va;

    fprintf('%-5s | %6.2f ± %6.2f     | %6.2f ± %6.2f     | %6.2f ± %6.2f     | %10.4f | %8.2f | %7.3f | %5.1f\n', ...
        name, ...
        100*mean(tr,'omitnan'), 100*std(tr,'omitnan'), ...
        100*mean(ta,'omitnan'), 100*std(ta,'omitnan'), ...
        100*mean(va,'omitnan'), 100*std(va,'omitnan'), ...
        mean(bv,'omitnan'), mean(tt,'omitnan'), mean(gap,'omitnan'), 100*mean(ff));
end
fprintf('====================================================\n\n');

%% ================= BOXPLOTS =================
labels = methods;
allTestAcc = []; allValAcc = []; allBestVa = []; allTime = [];
for mi = 1:numel(methods)
    name = methods{mi};
    allTestAcc = [allTestAcc, 100*R.(name).TestAcc]; %#ok<AGROW>
    allValAcc  = [allValAcc,  100*R.(name).ValAcc];  %#ok<AGROW>
    allBestVa  = [allBestVa,  R.(name).BestVa];      %#ok<AGROW>
    allTime    = [allTime,    R.(name).Time];        %#ok<AGROW>
end

figure('Name','Apple - 30 Runs Boxplots','Position',[100 100 1400 700]);

subplot(2,2,1); boxplot(allTestAcc,'Labels',labels);
title('Test Accuracy (%) - Boxplot (Higher is Better)'); ylabel('Test Acc (%)'); grid on; xtickangle(25);

subplot(2,2,2); boxplot(allValAcc,'Labels',labels);
title('Validation Accuracy (%) - Boxplot'); ylabel('Val Acc (%)'); grid on; xtickangle(25);

subplot(2,2,3); boxplot(allBestVa,'Labels',labels);
title('Best Validation Cost - Boxplot (Lower is Better)'); ylabel('Best Val Cost'); grid on; xtickangle(25);

subplot(2,2,4); boxplot(allTime,'Labels',labels);
title('Training Time (s) - Boxplot (Lower is Better)'); ylabel('Seconds'); grid on; xtickangle(25);

%% ================= SAVE =================
RESULTS = R; %#ok<NASGU>
save('APPLE_30RUN_RESULTS.mat','RESULTS','config');
fprintf(">> Kaydedildi: APPLE_30RUN_RESULTS.mat\n");
fprintf(">> Tamam.\n");

%% ================= EXTRA ANALYSIS PACK =================
% 1) Generalization gap (Train - Val) boxplot
allGap = [];
for mi = 1:numel(methods)
    name = methods{mi};
    gap = 100*(R.(name).TrainAcc - R.(name).ValAcc); % %
    allGap = [allGap, gap]; %#ok<AGROW>
end

figure('Name','Generalization Gap','Position',[200 200 900 350]);
boxplot(allGap,'Labels',methods);
title('Generalization Gap (TrainAcc - ValAcc) %  (Lower is Better)');
ylabel('Gap (%)'); grid on; xtickangle(25);

% 2) 95% Confidence Interval (bootstrap) for TestAcc mean
B = 5000; % bootstrap samples
CI = struct();

for mi = 1:numel(methods)
    name = methods{mi};
    x = 100*R.(name).TestAcc; % %
    x = x(isfinite(x));
    if numel(x) < 5
        CI.(name) = [NaN NaN];
        continue;
    end

    bootMeans = zeros(B,1);
    n = numel(x);
    for b = 1:B
        idx = randi(n, n, 1);
        bootMeans(b) = mean(x(idx));
    end
    CI.(name) = prctile(bootMeans, [2.5 97.5]);
end

fprintf('\n====================================================\n');
fprintf('  95%% BOOTSTRAP CI (TestAcc mean)\n');
fprintf('====================================================\n');
for mi = 1:numel(methods)
    name = methods{mi};
    mu = 100*mean(R.(name).TestAcc,'omitnan');
    fprintf('%-4s | mean=%.2f%% | CI=[%.2f, %.2f]\n', name, mu, CI.(name)(1), CI.(name)(2));
end

% 3) Pairwise statistical tests vs best method (by mean TestAcc)
means = zeros(numel(methods),1);
for mi = 1:numel(methods)
    means(mi) = mean(R.(methods{mi}).TestAcc,'omitnan');
end
[~, bestIdx] = max(means);
bestName = methods{bestIdx};

fprintf('\n====================================================\n');
fprintf('  PAIRED TESTS (TestAcc) vs BEST = %s\n', bestName);
fprintf('  Wilcoxon signed-rank + paired t-test + Cohen d\n');
fprintf('====================================================\n');

xbest = 100*R.(bestName).TestAcc;

for mi = 1:numel(methods)
    name = methods{mi};
    if strcmp(name, bestName), continue; end

    x = 100*R.(name).TestAcc;

    ok = isfinite(xbest) & isfinite(x);
    a = xbest(ok); b = x(ok);
    if numel(a) < 5
        fprintf('%-4s vs %-4s | insufficient paired samples\n', name, bestName);
        continue;
    end

    try
        p_w = signrank(a, b);
    catch
        p_w = NaN;
    end

    [~, p_t] = ttest(a, b);

    d = cohend_paired(a, b);

    fprintf('%-4s vs %-4s | Wilcoxon p=%.4g | ttest p=%.4g | d=%.3f | meanDiff=%.2f\n', ...
        name, bestName, p_w, p_t, d, mean(a-b));
end

% 4) Multi-metric rank table (TestAcc↑, ValAcc↑, BestVa↓, Time↓, Gap↓)
M = numel(methods);
metrics = struct();
metrics.TestAcc = zeros(M,1);
metrics.ValAcc  = zeros(M,1);
metrics.BestVa   = zeros(M,1);
metrics.Time     = zeros(M,1);
metrics.Gap      = zeros(M,1);

for mi = 1:M
    name = methods{mi};
    metrics.TestAcc(mi) = mean(R.(name).TestAcc,'omitnan');
    metrics.ValAcc(mi)  = mean(R.(name).ValAcc,'omitnan');
    metrics.BestVa(mi)  = mean(R.(name).BestVa,'omitnan');
    metrics.Time(mi)    = mean(R.(name).Time,'omitnan');
    metrics.Gap(mi)     = mean(R.(name).TrainAcc - R.(name).ValAcc,'omitnan');
end

rank_Test = rank_desc(metrics.TestAcc);
rank_Val  = rank_desc(metrics.ValAcc);
rank_BVa  = rank_asc(metrics.BestVa);
rank_Time = rank_asc(metrics.Time);
rank_Gap  = rank_asc(metrics.Gap);

totalScore = rank_Test + rank_Val + rank_BVa + rank_Time + rank_Gap;

[~, order] = sort(totalScore, 'ascend');

fprintf('\n====================================================\n');
fprintf('  RANK TABLE (lower total is better)\n');
fprintf('  Metrics: TestAcc↑ ValAcc↑ BestVa↓ Time↓ Gap↓\n');
fprintf('====================================================\n');
fprintf('Rank | Meth | Test | Val | BestVa | Time | Gap | Total\n');
fprintf('----------------------------------------------------\n');
for k = 1:M
    i = order(k);
    fprintf('%4d | %-4s |  %2d  | %2d  |   %2d   |  %2d  | %2d  |  %3d\n', ...
        k, methods{i}, rank_Test(i), rank_Val(i), rank_BVa(i), rank_Time(i), rank_Gap(i), totalScore(i));
end
fprintf('====================================================\n');

end % <-- MAIN FUNCTION END


%% =====================================================================
%% DATA LOADER (FIXED: MISSING VALUES)
%% =====================================================================
function [X_train, T_train, X_val, T_val, X_test, T_test, inD, outD] = load_data_fixed(filename)
    data = readtable(filename);

    if ~any(strcmpi(data.Properties.VariableNames, 'Quality'))
        error('Quality sutunu bulunamadi: %s', filename);
    end

    y = data.Quality;
    if iscell(y), y = categorical(y); end

    labels = categories(y);
    if numel(labels) ~= 2
        error('Binary classification bekleniyor, label sayisi=%d', numel(labels));
    end

    ynum = double(y == labels{2});

    featureCols = ~strcmpi(data.Properties.VariableNames, 'Quality');
    X = table2array(data(:, featureCols));

    % FIX: missing/NaN/Inf temizleme -> satırı at
    badRow = any(~isfinite(X), 2);
    if any(badRow)
        X = X(~badRow, :);
        ynum = ynum(~badRow);
    end

    % One-hot target
    T = zeros(numel(ynum), 2);
    T(sub2ind(size(T), (1:numel(ynum))', ynum + 1)) = 1;

    % Stratified split (60/20/20)
    idx0 = find(ynum == 0);
    idx1 = find(ynum == 1);

    n0 = numel(idx0); n1 = numel(idx1);
    n0_tr = round(0.6*n0); n0_val = round(0.2*n0);
    n1_tr = round(0.6*n1); n1_val = round(0.2*n1);

    idx0 = idx0(randperm(n0));
    idx1 = idx1(randperm(n1));

    trIdx  = [idx0(1:n0_tr); idx1(1:n1_tr)];
    valIdx = [idx0(n0_tr+1:n0_tr+n0_val); idx1(n1_tr+1:n1_tr+n1_val)];
    teIdx  = [idx0(n0_tr+n0_val+1:end); idx1(n1_tr+n1_val+1:end)];

    trIdx  = trIdx(randperm(numel(trIdx)));
    valIdx = valIdx(randperm(numel(valIdx)));
    teIdx  = teIdx(randperm(numel(teIdx)));

    X_train = X(trIdx, :);  T_train = T(trIdx, :);
    X_val   = X(valIdx, :); T_val   = T(valIdx, :);
    X_test  = X(teIdx, :);  T_test  = T(teIdx, :);

    % standardization
    mu  = mean(X_train, 1);
    sig = std(X_train, 0, 1) + 1e-8;
    X_train = (X_train - mu) ./ sig;
    X_val   = (X_val   - mu) ./ sig;
    X_test  = (X_test  - mu) ./ sig;

    inD  = size(X, 2);
    outD = size(T, 2);
end


%% =====================================================================
%% CORE NN
%% =====================================================================
function theta = init_params(net)
    L = net.L;
    theta = [];
    for l = 1:(L - 1)
        fanIn  = net.layers(l);
        fanOut = net.layers(l + 1);
        limit = sqrt(6 / (fanIn + fanOut));
        W = (rand(fanOut, fanIn) * 2 - 1) * limit;
        b = zeros(fanOut, 1);
        theta = [theta; W(:); b(:)]; %#ok<AGROW>
    end
end

function [acc, cost] = evaluate_net(theta, net, X, T)
    [~, Y] = forward_pass(theta, net, X);
    m = size(X, 1);
    cost = -(1 / m) * sum(sum(T .* log(Y + 1e-8)));
    [~, Ypred] = max(Y, [], 2);
    [~, Ttrue] = max(T, [], 2);
    acc = mean(Ypred == Ttrue);
end

function [A, Y] = forward_pass(theta, net, X)
    L = net.L; layers = net.layers; activation = net.activation;
    A = cell(L, 1);
    A{1} = X';
    idx = 1;
    for l = 1:(L - 1)
        fanIn  = layers(l);
        fanOut = layers(l + 1);
        Wsize = fanOut * fanIn;
        W = reshape(theta(idx:idx + Wsize - 1), [fanOut, fanIn]);
        idx = idx + Wsize;
        b = theta(idx:idx + fanOut - 1);
        idx = idx + fanOut;
        Z = W * A{l} + b;
        if l < (L - 1)
            switch activation
                case 'tanh',    A{l + 1} = tanh(Z);
                case 'relu',    A{l + 1} = max(0, Z);
                case 'sigmoid', A{l + 1} = 1 ./ (1 + exp(-Z));
                otherwise, error('Bilinmeyen aktivasyon: %s', activation);
            end
        else
            expZ = exp(Z - max(Z, [], 1));
            A{l + 1} = expZ ./ sum(expZ, 1);
        end
    end
    Y = A{end}';
    Y = max(min(Y, 1-1e-8), 1e-8); % clamp
end

function grad = compute_gradient(theta, net, X, T, lambda)
    L = net.L; layers = net.layers; activation = net.activation;
    m = size(X, 1);
    [A, Y] = forward_pass(theta, net, X);
    delta = cell(L, 1);
    delta{L} = (Y - T)';

    idx = numel(theta);
    for l = (L - 1):-1:2
        fanIn  = layers(l);
        fanOut = layers(l + 1);
        idx = idx - fanOut;
        Wsize = fanOut * fanIn;
        W = reshape(theta(idx - Wsize + 1:idx), [fanOut, fanIn]);
        idx = idx - Wsize;

        delta{l} = W' * delta{l + 1};
        switch activation
            case 'tanh',    delta{l} = delta{l} .* (1 - A{l}.^2);
            case 'relu',    delta{l} = delta{l} .* double(A{l} > 0);
            case 'sigmoid', delta{l} = delta{l} .* (A{l} .* (1 - A{l}));
        end
    end

    grad = zeros(size(theta), 'like', theta);
    idx = 1;
    for l = 1:(L - 1)
        fanIn  = layers(l);
        fanOut = layers(l + 1);

        dW = (1 / m) * (delta{l + 1} * A{l}');
        Wsize = fanOut * fanIn;
        W = reshape(theta(idx:idx + Wsize - 1), [fanOut, fanIn]);
        dW = dW + (lambda / m) * W;

        grad(idx:idx + Wsize - 1) = dW(:);
        idx = idx + Wsize;

        db = (1 / m) * sum(delta{l + 1}, 2);
        grad(idx:idx + fanOut - 1) = db;
        idx = idx + fanOut;
    end
end

function cost = compute_cost(theta, net, X, T, lambda)
    [~, Y] = forward_pass(theta, net, X);
    m = size(X, 1);
    cost = -(1 / m) * sum(sum(T .* log(Y + 1e-8)));

    L = net.L; layers = net.layers;
    idx = 1; regTerm = 0;
    for l = 1:(L - 1)
        fanIn  = layers(l);
        fanOut = layers(l + 1);
        Wsize = fanOut * fanIn;
        W = reshape(theta(idx:idx + Wsize - 1), [fanOut, fanIn]);
        regTerm = regTerm + sum(W(:).^2);
        idx = idx + Wsize + fanOut;
    end
    cost = cost + (lambda / (2 * m)) * regTerm;
end


%% =====================================================================
%% TRAINERS (FIX2: best init + NaN guard continue)
%% =====================================================================
function [bestTheta, hist] = train_gd_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, maxIter, lr, patience)
    bestTheta = theta;
    bestValCost = gather(compute_cost(theta, net, Xva, Tva, lambda));
    bestIter = 0;
    noImprove = 0;

    hist = init_hist(maxIter, theta);

    for iter = 1:maxIter
        grad = compute_gradient(theta, net, Xtr, Ttr, lambda);
        theta = theta - lr * grad;

        trC = gather(compute_cost(theta, net, Xtr, Ttr, lambda));
        vaC = gather(compute_cost(theta, net, Xva, Tva, lambda));
        teC = gather(compute_cost(theta, net, Xte, Tte, lambda));

        if ~isfinite(vaC) || ~isfinite(trC)
            theta = bestTheta;
            lr = lr * 0.5;
            continue;
        end

        hist.tr(iter)=trC; hist.va(iter)=vaC; hist.te(iter)=teC;

        [hist.tr_acc(iter), ~] = evaluate_net(theta, net, Xtr, Ttr);
        [hist.va_acc(iter), ~] = evaluate_net(theta, net, Xva, Tva);
        [hist.te_acc(iter), ~] = evaluate_net(theta, net, Xte, Tte);

        if vaC < bestValCost
            bestValCost = vaC;
            bestTheta = theta;
            bestIter = iter;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end

        if noImprove >= patience
            break;
        end
    end

    hist = trim_hist(hist);
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

function [bestTheta, hist] = train_cg_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, maxIter, lr, patience)
    bestTheta = theta;
    bestValCost = gather(compute_cost(theta, net, Xva, Tva, lambda));
    bestIter = 0;
    noImprove = 0;

    hist = init_hist(maxIter, theta);

    g = compute_gradient(theta, net, Xtr, Ttr, lambda);
    d = -g;

    for iter = 1:maxIter
        theta_new = theta + lr * d;

        trC = gather(compute_cost(theta_new, net, Xtr, Ttr, lambda));
        vaC = gather(compute_cost(theta_new, net, Xva, Tva, lambda));
        teC = gather(compute_cost(theta_new, net, Xte, Tte, lambda));

        if ~isfinite(vaC) || ~isfinite(trC)
            lr = lr * 0.5;
            continue;
        end

        theta = theta_new;

        hist.tr(iter)=trC; hist.va(iter)=vaC; hist.te(iter)=teC;
        [hist.tr_acc(iter), ~] = evaluate_net(theta, net, Xtr, Ttr);
        [hist.va_acc(iter), ~] = evaluate_net(theta, net, Xva, Tva);
        [hist.te_acc(iter), ~] = evaluate_net(theta, net, Xte, Tte);

        if vaC < bestValCost
            bestValCost = vaC;
            bestTheta = theta;
            bestIter = iter;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end
        if noImprove >= patience, break; end

        g_new = compute_gradient(theta, net, Xtr, Ttr, lambda);
        beta = (g_new' * g_new) / (g' * g + 1e-8);
        d = -g_new + beta * d;
        g = g_new;
    end

    hist = trim_hist(hist);
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

function [bestTheta, hist] = train_bfgs_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, maxIter, lr, patience)
    n = numel(theta);
    H = eye(n, 'like', theta);

    bestTheta = theta;
    bestValCost = gather(compute_cost(theta, net, Xva, Tva, lambda));
    bestIter = 0;
    noImprove = 0;

    hist = init_hist(maxIter, theta);

    g = compute_gradient(theta, net, Xtr, Ttr, lambda);

    for iter = 1:maxIter
        d = -H * g;
        theta_new = theta + lr * d;
        g_new = compute_gradient(theta_new, net, Xtr, Ttr, lambda);

        s = theta_new - theta;
        y = g_new - g;

        rho = 1 / (y' * s + 1e-8);
        I = eye(n, 'like', theta);
        H = (I - rho*s*y') * H * (I - rho*y*s') + rho*(s*s');

        trC = gather(compute_cost(theta_new, net, Xtr, Ttr, lambda));
        vaC = gather(compute_cost(theta_new, net, Xva, Tva, lambda));
        teC = gather(compute_cost(theta_new, net, Xte, Tte, lambda));

        if ~isfinite(vaC) || ~isfinite(trC)
            lr = lr * 0.5;
            continue;
        end

        theta = theta_new;
        g = g_new;

        hist.tr(iter)=trC; hist.va(iter)=vaC; hist.te(iter)=teC;
        [hist.tr_acc(iter), ~] = evaluate_net(theta, net, Xtr, Ttr);
        [hist.va_acc(iter), ~] = evaluate_net(theta, net, Xva, Tva);
        [hist.te_acc(iter), ~] = evaluate_net(theta, net, Xte, Tte);

        if vaC < bestValCost
            bestValCost = vaC;
            bestTheta = theta;
            bestIter = iter;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end
        if noImprove >= patience, break; end
    end

    hist = trim_hist(hist);
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

function [bestTheta, hist] = train_dfp_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, maxIter, lr, patience)
    n = numel(theta);
    H = eye(n, 'like', theta);

    bestTheta = theta;
    bestValCost = gather(compute_cost(theta, net, Xva, Tva, lambda));
    bestIter = 0;
    noImprove = 0;

    hist = init_hist(maxIter, theta);

    g = compute_gradient(theta, net, Xtr, Ttr, lambda);

    for iter = 1:maxIter
        d = -H * g;
        theta_new = theta + lr * d;
        g_new = compute_gradient(theta_new, net, Xtr, Ttr, lambda);

        s = theta_new - theta;
        y = g_new - g;

        Hy = H * y;
        H = H + (s*s')/(s'*y + 1e-8) - (Hy*Hy')/(y'*Hy + 1e-8);

        trC = gather(compute_cost(theta_new, net, Xtr, Ttr, lambda));
        vaC = gather(compute_cost(theta_new, net, Xva, Tva, lambda));
        teC = gather(compute_cost(theta_new, net, Xte, Tte, lambda));

        if ~isfinite(vaC) || ~isfinite(trC)
            lr = lr * 0.5;
            continue;
        end

        theta = theta_new;
        g = g_new;

        hist.tr(iter)=trC; hist.va(iter)=vaC; hist.te(iter)=teC;
        [hist.tr_acc(iter), ~] = evaluate_net(theta, net, Xtr, Ttr);
        [hist.va_acc(iter), ~] = evaluate_net(theta, net, Xva, Tva);
        [hist.te_acc(iter), ~] = evaluate_net(theta, net, Xte, Tte);

        if vaC < bestValCost
            bestValCost = vaC;
            bestTheta = theta;
            bestIter = iter;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end
        if noImprove >= patience, break; end
    end

    hist = trim_hist(hist);
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end

function [bestTheta, hist] = train_abc_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, SN, limit, maxCycle, patience)
    D = numel(theta);

    pop = repmat(theta', SN, 1) + randn(SN, D, 'like', theta) * 0.1;
    trial = zeros(SN, 1, 'like', theta);

    % fitness = VALIDATION COST
    fitness = zeros(SN, 1, 'like', theta);
    for i = 1:SN
        fitness(i) = compute_cost(pop(i, :)', net, Xva, Tva, lambda);
    end

    [bestValCost, bestIdx] = min(fitness);
    bestTheta = pop(bestIdx, :)';
    bestValCost = gather(bestValCost);
    bestIter = 0;
    noImprove = 0;

    hist = init_hist(maxCycle, theta);

    for cycle = 1:maxCycle
        % employed
        for i = 1:SN
            k = randi(SN); while k==i, k=randi(SN); end
            phi = (rand(1, D, 'like', theta) * 2 - 1);
            v = pop(i, :) + phi .* (pop(i, :) - pop(k, :));
            fv = compute_cost(v', net, Xva, Tva, lambda);

            if fv < fitness(i)
                pop(i, :) = v;
                fitness(i) = fv;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
        end

        % scout
        for i = 1:SN
            if trial(i) >= limit
                pop(i, :) = theta' + randn(1, D, 'like', theta) * 0.1;
                fitness(i) = compute_cost(pop(i, :)', net, Xva, Tva, lambda);
                trial(i) = 0;
            end
        end

        [~, bestIdx] = min(fitness);
        cand = pop(bestIdx, :)';

        trC = gather(compute_cost(cand, net, Xtr, Ttr, lambda));
        vaC = gather(compute_cost(cand, net, Xva, Tva, lambda));
        teC = gather(compute_cost(cand, net, Xte, Tte, lambda));

        if ~isfinite(vaC) || ~isfinite(trC)
            continue;
        end

        hist.tr(cycle)=trC; hist.va(cycle)=vaC; hist.te(cycle)=teC;
        [hist.tr_acc(cycle), ~] = evaluate_net(cand, net, Xtr, Ttr);
        [hist.va_acc(cycle), ~] = evaluate_net(cand, net, Xva, Tva);
        [hist.te_acc(cycle), ~] = evaluate_net(cand, net, Xte, Tte);

        if vaC < bestValCost
            bestValCost = vaC;
            bestTheta = cand;
            bestIter = cycle;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end
        if noImprove >= patience, break; end
    end

    hist = trim_hist(hist);
    hist.bestIter = bestIter;
    hist.bestValCost = bestValCost;
end


%% =====================================================================
%% HIST UTILS
%% =====================================================================
function hist = init_hist(N, likeTheta)
    hist.tr = nan(N,1,'like',likeTheta);
    hist.va = nan(N,1,'like',likeTheta);
    hist.te = nan(N,1,'like',likeTheta);
    hist.tr_acc = nan(N,1,'like',likeTheta);
    hist.va_acc = nan(N,1,'like',likeTheta);
    hist.te_acc = nan(N,1,'like',likeTheta);
end

function hist = trim_hist(hist)
    k = find(~isnan(gather(hist.va)), 1, 'last');
    if isempty(k), k = 1; end
    hist.tr = hist.tr(1:k);
    hist.va = hist.va(1:k);
    hist.te = hist.te(1:k);
    hist.tr_acc = hist.tr_acc(1:k);
    hist.va_acc = hist.va_acc(1:k);
    hist.te_acc = hist.te_acc(1:k);
end


%% =====================================================================
%% EXTRA HELPERS (effect size + ranks)
%% =====================================================================
function d = cohend_paired(a, b)
    diff = a - b;
    d = mean(diff) / (std(diff) + 1e-8);
end

function r = rank_desc(x)
    [~, idx] = sort(x, 'descend');
    r = zeros(size(x));
    for k = 1:numel(idx), r(idx(k)) = k; end
end

function r = rank_asc(x)
    [~, idx] = sort(x, 'ascend');
    r = zeros(size(x));
    for k = 1:numel(idx), r(idx(k)) = k; end
end

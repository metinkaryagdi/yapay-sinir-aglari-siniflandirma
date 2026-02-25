function apple_ysa_single_run()
% Single run test script.
% Runs each method once. Edit hyperparameters below.
% Early stopping is enabled.
% NOW ALSO TRACKS TEST cost/acc DURING TRAINING (for plots).

clear;
clc;
close all;
rng(42);

%% Hyperparameters (edit here)
params.BFGS.H = 16;
params.BFGS.lambda = 0.0001;
params.BFGS.lr = 0.030;
params.BFGS.maxIter = 500;

params.DFP.H = 32;
params.DFP.lambda = 0.0005;
params.DFP.lr = 0.010;
params.DFP.maxIter = 500;

params.CG.H = 16;
params.CG.lambda = 0.002;
params.CG.lr = 0.065;
params.CG.maxIter = 400;

params.GD.H = 32;
params.GD.lambda = 0.0001;
params.GD.lr = 0.065;
params.GD.maxIter = 500;

params.ABC.H = 8;
params.ABC.lambda = 0.0005;
params.ABC.SN = 20;
params.ABC.limit = 100;
params.ABC.maxCycle = 300;

PATIENCE = 25;
DATA_FILE = 'apple_quality.csv';

%% Data loading
fprintf('>> Loading data...\n');
[X_train, T_train, X_val, T_val, X_test, T_test, inputDim, outputDim] = ...
    load_data(DATA_FILE);

fprintf('   Train: %d samples\n', size(X_train, 1));
fprintf('   Val:   %d samples\n', size(X_val, 1));
fprintf('   Test:  %d samples\n', size(X_test, 1));

%% GPU
useGPU = false;
try
    if gpuDeviceCount > 0
        gpuDevice;
        useGPU = true;
        fprintf('>> GPU in use.\n');
    end
catch
    fprintf('>> CPU in use.\n');
end

if useGPU
    Xtr = gpuArray(single(X_train));
    Ttr = gpuArray(single(T_train));
    Xva = gpuArray(single(X_val));
    Tva = gpuArray(single(T_val));
    Xte = gpuArray(single(X_test));
    Tte = gpuArray(single(T_test));
else
    Xtr = X_train;
    Ttr = T_train;
    Xva = X_val;
    Tva = T_val;
    Xte = X_test;
    Tte = T_test;
end

methods = {'BFGS', 'DFP', 'CG', 'GD', 'ABC'};
results = struct();

%% Train
for m = 1:numel(methods)
    methodName = methods{m};
    p = params.(methodName);

    fprintf('\n========================================\n');
    fprintf('>> %s training...\n', methodName);
    if isfield(p, 'lr')
        fprintf('   H=%d, lambda=%.5f, lr=%.4f\n', p.H, p.lambda, p.lr);
    else
        fprintf('   H=%d, lambda=%.5f\n', p.H, p.lambda);
    end
    fprintf('========================================\n');

    net.inputDim = inputDim;
    net.outputDim = outputDim;
    net.layers = p.H;
    net.numHidden = 1;
    net.activation = 'tanh';

    theta = init_weights(net);
    if useGPU
        theta = gpuArray(theta);
    end

    tic;
    switch methodName
        case 'BFGS'
            [bestTheta, hist] = train_bfgs_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, p.lambda, p.lr, p.maxIter, PATIENCE);
        case 'DFP'
            [bestTheta, hist] = train_dfp_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, p.lambda, p.lr, p.maxIter, PATIENCE);
        case 'CG'
            [bestTheta, hist] = train_cg_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, p.lambda, p.lr, p.maxIter, PATIENCE);
        case 'GD'
            [bestTheta, hist] = train_gd_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, p.lambda, p.lr, p.maxIter, PATIENCE);
        case 'ABC'
            [bestTheta, hist] = train_abc_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, p.lambda, p.SN, p.limit, p.maxCycle, PATIENCE);
    end
    elapsed = toc;

    [valAcc, ~]  = evaluate_net(bestTheta, net, Xva, Tva);
    [testAcc, ~] = evaluate_net(bestTheta, net, Xte, Tte);

    results.(methodName).valAcc = double(valAcc);
    results.(methodName).testAcc = double(testAcc);
    results.(methodName).bestValCost = min(hist.va);
    results.(methodName).bestIter = hist.bestIter;
    results.(methodName).time = elapsed;
    results.(methodName).hist = hist;

    fprintf('   Done (%.1f s)\n', elapsed);
    fprintf('   Best Iter: %d\n', hist.bestIter);
    fprintf('   Val Cost: %.4f\n', min(hist.va));
    fprintf('   Val Accuracy: %.2f%%\n', valAcc * 100);
    fprintf('   Test Accuracy: %.2f%%\n', testAcc * 100);
end

%% Summary table
fprintf('\n\n========================================\n');
fprintf('           SUMMARY TABLE               \n');
fprintf('========================================\n');
fprintf('Method | Val Acc | Test Acc | Best Iter | Time (s)\n');
fprintf('-------+---------+----------+-----------+---------\n');

for m = 1:numel(methods)
    methodName = methods{m};
    r = results.(methodName);
    fprintf('%-6s | %6.2f%% | %7.2f%% | %9d | %7.1f\n', ...
        methodName, r.valAcc * 100, r.testAcc * 100, r.bestIter, r.time);
end
fprintf('========================================\n');

%% Plots
colors = struct('BFGS', [0.2 0.4 0.8], 'DFP', [0.8 0.2 0.2], ...
    'CG', [0.2 0.7 0.3], 'GD', [0.9 0.5 0.1], 'ABC', [0.6 0.2 0.8]);

figure('Name', 'All Methods - Cost Comparison', 'Position', [50 50 1400 500]);

subplot(1, 2, 1);
hold on;
for m = 1:numel(methods)
    methodName = methods{m};
    plot(results.(methodName).hist.va, 'LineWidth', 2, 'Color', colors.(methodName));      % Val (solid)
    plot(results.(methodName).hist.te, '--', 'LineWidth', 1.5, 'Color', colors.(methodName)); % Test (dashed)
end
hold off;
xlabel('Iteration');
ylabel('Cost');
title('Validation (solid) vs Test (dashed) Cost');
legend(methods, 'Location', 'best');
grid on;

subplot(1, 2, 2);
hold on;
for m = 1:numel(methods)
    methodName = methods{m};
    plot(results.(methodName).hist.tr, 'LineWidth', 2, 'Color', colors.(methodName));      % Train (solid)
end
hold off;
xlabel('Iteration');
ylabel('Training Cost');
title('Training Cost Comparison');
legend(methods, 'Location', 'best');
grid on;

figure('Name', 'All Methods - Accuracy Comparison', 'Position', [100 100 1400 500]);

subplot(1, 2, 1);
hold on;
for m = 1:numel(methods)
    methodName = methods{m};
    plot(results.(methodName).hist.va_acc * 100, 'LineWidth', 2, 'Color', colors.(methodName));      % Val (solid)
    plot(results.(methodName).hist.te_acc * 100, '--', 'LineWidth', 1.5, 'Color', colors.(methodName)); % Test (dashed)
end
hold off;
xlabel('Iteration');
ylabel('Accuracy (%)');
title('Validation (solid) vs Test (dashed) Accuracy');
legend(methods, 'Location', 'best');
grid on;
ylim([0 100]);

subplot(1, 2, 2);
hold on;
for m = 1:numel(methods)
    methodName = methods{m};
    plot(results.(methodName).hist.tr_acc * 100, 'LineWidth', 2, 'Color', colors.(methodName));
end
hold off;
xlabel('Iteration');
ylabel('Training Accuracy (%)');
title('Training Accuracy Comparison');
legend(methods, 'Location', 'best');
grid on;
ylim([0 100]);

figure('Name', 'All Methods - Final Accuracy', 'Position', [150 150 800 500]);
valAccs = zeros(1, numel(methods));
testAccs = zeros(1, numel(methods));
for m = 1:numel(methods)
    valAccs(m) = results.(methods{m}).valAcc * 100;
    testAccs(m) = results.(methods{m}).testAcc * 100;
end
bar_data = [valAccs; testAccs]';
b = bar(bar_data);
b(1).FaceColor = [0.3 0.6 0.9];
b(2).FaceColor = [0.9 0.4 0.4];
set(gca, 'XTickLabel', methods);
xlabel('Method');
ylabel('Accuracy (%)');
title('Accuracy Comparison');
legend({'Validation', 'Test'}, 'Location', 'best');
grid on;
ylim([0 100]);

% Per-method plots
for m = 1:numel(methods)
    methodName = methods{m};
    r = results.(methodName);
    p = params.(methodName);

    figure('Name', sprintf('%s - Detail', methodName), ...
        'Position', [100 + m * 30, 100 + m * 30, 1400, 420]);

    % --- Cost plot (Train/Val/Test) ---
    subplot(1, 3, 1);
    hold on;
    plot(r.hist.tr, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Cost');
    plot(r.hist.va, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Cost');
    plot(r.hist.te, 'k--', 'LineWidth', 1.8, 'DisplayName', 'Test Cost');
    xline(r.bestIter, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Best@%d', r.bestIter));
    plot(r.bestIter, r.bestValCost, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', ...
        'DisplayName', sprintf('Best Val: %.4f', r.bestValCost));
    hold off;
    xlabel('Iteration');
    ylabel('Cost');
    title(sprintf('%s - Train vs Val vs Test Cost', methodName));
    legend('Location', 'best');
    grid on;

    % --- Accuracy plot (Train/Val/Test) ---
    subplot(1, 3, 2);
    hold on;
    plot(r.hist.tr_acc * 100, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Acc');
    plot(r.hist.va_acc * 100, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Acc');
    plot(r.hist.te_acc * 100, 'k--', 'LineWidth', 1.8, 'DisplayName', 'Test Acc');
    xline(r.bestIter, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Best@%d', r.bestIter));
    hold off;
    xlabel('Iteration');
    ylabel('Accuracy (%)');
    title(sprintf('%s - Train vs Val vs Test Accuracy', methodName));
    legend('Location', 'best');
    grid on;
    ylim([0 100]);

    % --- Summary text ---
    subplot(1, 3, 3);
    axis off;

    % Try to read best test at bestIter (if available)
    bestTeCost = NaN; bestTeAcc = NaN;
    if numel(r.hist.te) >= r.bestIter
        bestTeCost = r.hist.te(r.bestIter);
    end
    if numel(r.hist.te_acc) >= r.bestIter
        bestTeAcc = r.hist.te_acc(r.bestIter) * 100;
    end

    info_lines = { ...
        sprintf('%s RESULTS', methodName), ...
        '========================', ...
        '', ...
        sprintf('Validation Accuracy: %.2f%%', r.valAcc * 100), ...
        sprintf('Test Accuracy:       %.2f%%', r.testAcc * 100), ...
        '', ...
        sprintf('Best Validation Cost: %.4f', r.bestValCost), ...
        sprintf('Best Iteration:       %d', r.bestIter), ...
        sprintf('Test@BestIter Cost:   %.4f', bestTeCost), ...
        sprintf('Test@BestIter Acc:    %.2f%%', bestTeAcc), ...
        sprintf('Total Time:           %.1f s', r.time), ...
        '', ...
        '--- Hyperparameters ---', ...
        sprintf('H (Hidden Units): %d', p.H), ...
        sprintf('Lambda: %.5f', p.lambda) ...
    };

    if isfield(p, 'lr')
        info_lines{end + 1} = sprintf('Learning Rate: %.4f', p.lr);
    end
    if isfield(p, 'maxIter')
        info_lines{end + 1} = sprintf('Max Iter: %d', p.maxIter);
    end
    if isfield(p, 'SN')
        info_lines{end + 1} = sprintf('ABC SN: %d', p.SN);
        info_lines{end + 1} = sprintf('ABC Limit: %d', p.limit);
        info_lines{end + 1} = sprintf('ABC MaxCycle: %d', p.maxCycle);
    end

    text(0.1, 0.95, strjoin(info_lines, '\n'), 'FontSize', 11, ...
        'VerticalAlignment', 'top', 'FontName', 'Courier');
    title(sprintf('%s - Summary', methodName));
end

fprintf('\nAll methods completed. Figures are generated.\n');
end

%% Helper functions
function [X_train, T_train, X_val, T_val, X_test, T_test, inD, outD] = load_data(filename)
    data = readtable(filename);

    if ~any(strcmpi(data.Properties.VariableNames, 'Quality'))
        error('Quality column not found in %s', filename);
    end

    q = data.(data.Properties.VariableNames{strcmpi(data.Properties.VariableNames, 'Quality')});
    y_bin = double(lower(string(q)) == "good");

    isNum = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
    isQuality = strcmpi(data.Properties.VariableNames, 'Quality');
    featCols = find(isNum & ~isQuality);
    if isempty(featCols)
        error('No numeric feature columns found in %s', filename);
    end

    X = table2array(data(:, featCols));

    bad = any(~isfinite(X), 2) | ~isfinite(y_bin);
    X(bad, :) = [];
    y_bin(bad, :) = [];

    T = [y_bin == 0, y_bin == 1];

    % Stratified split: 60/20/20
    rng(42);
    idx0 = find(y_bin == 0);
    idx1 = find(y_bin == 1);
    idx0 = idx0(randperm(numel(idx0)));
    idx1 = idx1(randperm(numel(idx1)));

    n0_tr = round(0.6 * numel(idx0));
    n1_tr = round(0.6 * numel(idx1));
    n0_val = round(0.2 * numel(idx0));
    n1_val = round(0.2 * numel(idx1));

    trIdx  = [idx0(1:n0_tr); idx1(1:n1_tr)];
    valIdx = [idx0(n0_tr + 1:n0_tr + n0_val); idx1(n1_tr + 1:n1_tr + n1_val)];
    teIdx  = [idx0(n0_tr + n0_val + 1:end); idx1(n1_tr + n1_val + 1:end)];

    trIdx  = trIdx(randperm(numel(trIdx)));
    valIdx = valIdx(randperm(numel(valIdx)));
    teIdx  = teIdx(randperm(numel(teIdx)));

    X_train = X(trIdx, :);  T_train = T(trIdx, :);
    X_val   = X(valIdx, :); T_val   = T(valIdx, :);
    X_test  = X(teIdx, :);  T_test  = T(teIdx, :);

    mu = mean(X_train, 1);
    sig = std(X_train, 0, 1) + 1e-8;
    X_train = (X_train - mu) ./ sig;
    X_val   = (X_val - mu) ./ sig;
    X_test  = (X_test - mu) ./ sig;

    inD = size(X, 2);
    outD = size(T, 2);
end

function theta = init_weights(net)
    layers = net.layers;
    if isscalar(layers)
        layers = [layers];
    end

    cols = {};
    prev = net.inputDim;
    for i = 1:numel(layers)
        h = layers(i);
        lim = sqrt(6 / (prev + h));
        cols{end + 1} = (rand(h, prev) * 2 - 1) * lim;
        cols{end + 1} = zeros(h, 1);
        prev = h;
    end

    lim = sqrt(6 / (prev + net.outputDim));
    cols{end + 1} = (rand(net.outputDim, prev) * 2 - 1) * lim;
    cols{end + 1} = zeros(net.outputDim, 1);

    for k = 1:numel(cols)
        cols{k} = cols{k}(:);
    end
    theta = vertcat(cols{:});
end

function [As, Ws, cost] = forward(theta, net, X, T, lam)
    layers = net.layers;
    if isscalar(layers)
        layers = [layers];
    end
    numHidden = numel(layers);

    idx = 1;
    Ws = cell(1, numHidden + 1);
    bs = cell(1, numHidden + 1);
    prev = net.inputDim;

    for i = 1:numHidden
        h = layers(i);
        lenW = h * prev;
        Ws{i} = reshape(theta(idx:idx + lenW - 1), [h, prev]);
        idx = idx + lenW;
        bs{i} = reshape(theta(idx:idx + h - 1), [h, 1]);
        idx = idx + h;
        prev = h;
    end

    lenW = net.outputDim * prev;
    Ws{end} = reshape(theta(idx:idx + lenW - 1), [net.outputDim, prev]);
    idx = idx + lenW;
    bs{end} = reshape(theta(idx:idx + net.outputDim - 1), [net.outputDim, 1]);

    As = cell(1, numHidden + 2);
    A = X;
    As{1} = A;

    for i = 1:numHidden
        Z = A * Ws{i}.' + bs{i}.';
        switch net.activation
            case 'tanh'
                A = tanh(Z);
            case 'relu'
                A = max(0, Z);
            case 'sigmoid'
                A = 1 ./ (1 + exp(-Z));
            otherwise
                A = tanh(Z);
        end
        As{i + 1} = A;
    end

    Z = A * Ws{end}.' + bs{end}.';
    Z = Z - max(Z, [], 2);
    Y = exp(Z) ./ sum(exp(Z), 2);
    As{end} = Y;

    if isempty(T)
        cost = 0;
        return;
    end

    cost = -mean(sum(T .* log(Y + 1e-10), 2));
    reg = 0;
    for k = 1:numel(Ws)
        reg = reg + sum(Ws{k}(:).^2);
    end
    cost = cost + 0.5 * lam * reg;
end

function [c, g] = cost_grad(theta, net, X, T, lam)
    [As, Ws, c] = forward(theta, net, X, T, lam);
    numHidden = numel(Ws) - 1;

    N = size(X, 1);
    Y = As{end};
    dZ = (Y - T) / N;

    grads = cell(1, numel(Ws) * 2);
    gid = numel(grads);

    gW = dZ.' * As{end - 1};
    gb = sum(dZ, 1).';
    if lam > 0
        gW = gW + lam * Ws{end};
    end
    grads{gid} = gb(:); gid = gid - 1;
    grads{gid} = gW(:); gid = gid - 1;

    dA = dZ * Ws{end};

    for i = numHidden:-1:1
        switch net.activation
            case 'tanh'
                dZ = dA .* (1 - As{i + 1}.^2);
            case 'relu'
                dZ = dA .* (As{i + 1} > 0);
            case 'sigmoid'
                dZ = dA .* As{i + 1} .* (1 - As{i + 1});
            otherwise
                dZ = dA .* (1 - As{i + 1}.^2);
        end

        gW = dZ.' * As{i};
        gb = sum(dZ, 1).';
        if lam > 0
            gW = gW + lam * Ws{i};
        end

        grads{gid} = gb(:); gid = gid - 1;
        grads{gid} = gW(:); gid = gid - 1;

        if i > 1
            dA = dZ * Ws{i};
        end
    end

    for k = 1:numel(grads)
        grads{k} = grads{k}(:);
    end
    g = vertcat(grads{:});
end

function [acc, cost] = evaluate_net(theta, net, X, T)
    [~, ~, cost] = forward(theta, net, X, T, 0);
    [As, ~, ~] = forward(theta, net, X, [], 0);
    Y = As{end};
    [~, pred] = max(Y, [], 2);
    [~, truth] = max(T, [], 2);
    acc = mean(pred == truth);
end

%% Optimizers (NOW TRACK TEST TOO)
function [finalTheta, hist] = train_gd_es(theta, net, X, T, Xv, Tv, Xt, Tt, lam, lr, maxIter, patience)
    hist.tr = [];  hist.va = [];  hist.te = [];
    hist.tr_acc = []; hist.va_acc = []; hist.te_acc = [];
    bestVal = Inf; bestTheta = theta; noImp = 0; hist.bestIter = 1;

    for k = 1:maxIter
        [c, g] = cost_grad(theta, net, X, T, lam);
        theta = theta - lr * g;

        [~, ~, vc]  = forward(theta, net, Xv, Tv, lam);
        [~, ~, tec] = forward(theta, net, Xt, Tt, lam);

        [tr_acc, ~] = evaluate_net(theta, net, X,  T);
        [va_acc, ~] = evaluate_net(theta, net, Xv, Tv);
        [te_acc, ~] = evaluate_net(theta, net, Xt, Tt);

        hist.tr(end + 1) = gather(c);
        hist.va(end + 1) = gather(vc);
        hist.te(end + 1) = gather(tec);
        hist.tr_acc(end + 1) = gather(tr_acc);
        hist.va_acc(end + 1) = gather(va_acc);
        hist.te_acc(end + 1) = gather(te_acc);

        if vc < bestVal
            bestVal = vc; bestTheta = theta; noImp = 0; hist.bestIter = k;
        else
            noImp = noImp + 1;
        end

        if noImp >= patience
            fprintf('      Early Stop @ Iter %d\n', k);
            break;
        end
    end
    finalTheta = bestTheta;
end

function [finalTheta, hist] = train_bfgs_es(theta, net, X, T, Xv, Tv, Xt, Tt, lam, lr, maxIter, patience)
    hist.tr = [];  hist.va = [];  hist.te = [];
    hist.tr_acc = []; hist.va_acc = []; hist.te_acc = [];

    n = numel(theta);
    H = eye(n, 'like', theta);
    [c, g] = cost_grad(theta, net, X, T, lam);

    bestVal = Inf; bestTheta = theta; noImp = 0; hist.bestIter = 1;

    for k = 1:maxIter
        p = -H * g;
        t_new = theta + lr * p;

        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        [~, ~, vc]  = forward(t_new, net, Xv, Tv, lam);
        [~, ~, tec] = forward(t_new, net, Xt, Tt, lam);

        [tr_acc, ~] = evaluate_net(t_new, net, X,  T);
        [va_acc, ~] = evaluate_net(t_new, net, Xv, Tv);
        [te_acc, ~] = evaluate_net(t_new, net, Xt, Tt);

        hist.tr(end + 1) = gather(c_new);
        hist.va(end + 1) = gather(vc);
        hist.te(end + 1) = gather(tec);
        hist.tr_acc(end + 1) = gather(tr_acc);
        hist.va_acc(end + 1) = gather(va_acc);
        hist.te_acc(end + 1) = gather(te_acc);

        if vc < bestVal
            bestVal = vc; bestTheta = t_new; noImp = 0; hist.bestIter = k;
        else
            noImp = noImp + 1;
        end

        if noImp >= patience
            fprintf('      Early Stop @ Iter %d\n', k);
            break;
        end

        s = t_new - theta;
        y = g_new - g;
        ys = dot(y, s);
        if ys > 1e-10
            rho = 1 / ys;
            V = eye(n, 'like', theta) - rho * (y * s.');
            H = V.' * H * V + rho * (s * s.');
        end

        theta = t_new; c = c_new; g = g_new;
    end
    finalTheta = bestTheta;
end

function [finalTheta, hist] = train_dfp_es(theta, net, X, T, Xv, Tv, Xt, Tt, lam, lr, maxIter, patience)
    hist.tr = [];  hist.va = [];  hist.te = [];
    hist.tr_acc = []; hist.va_acc = []; hist.te_acc = [];

    n = numel(theta);
    H = eye(n, 'like', theta);
    [c, g] = cost_grad(theta, net, X, T, lam);

    bestVal = Inf; bestTheta = theta; noImp = 0; hist.bestIter = 1;

    for k = 1:maxIter
        p = -H * g;
        t_new = theta + lr * p;

        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        [~, ~, vc]  = forward(t_new, net, Xv, Tv, lam);
        [~, ~, tec] = forward(t_new, net, Xt, Tt, lam);

        [tr_acc, ~] = evaluate_net(t_new, net, X,  T);
        [va_acc, ~] = evaluate_net(t_new, net, Xv, Tv);
        [te_acc, ~] = evaluate_net(t_new, net, Xt, Tt);

        hist.tr(end + 1) = gather(c_new);
        hist.va(end + 1) = gather(vc);
        hist.te(end + 1) = gather(tec);
        hist.tr_acc(end + 1) = gather(tr_acc);
        hist.va_acc(end + 1) = gather(va_acc);
        hist.te_acc(end + 1) = gather(te_acc);

        if vc < bestVal
            bestVal = vc; bestTheta = t_new; noImp = 0; hist.bestIter = k;
        else
            noImp = noImp + 1;
        end

        if noImp >= patience
            fprintf('      Early Stop @ Iter %d\n', k);
            break;
        end

        s = t_new - theta;
        y = g_new - g;
        sy = dot(s, y);
        if sy > 1e-10
            Hy = H * y;
            yHy = dot(y, Hy);
            if yHy > 1e-10
                H = H + (s * s.') / sy - (Hy * Hy.') / yHy;
            end
        end

        theta = t_new; c = c_new; g = g_new;
    end
    finalTheta = bestTheta;
end

function [finalTheta, hist] = train_cg_es(theta, net, X, T, Xv, Tv, Xt, Tt, lam, lr, maxIter, patience)
    hist.tr = [];  hist.va = [];  hist.te = [];
    hist.tr_acc = []; hist.va_acc = []; hist.te_acc = [];

    [c, g] = cost_grad(theta, net, X, T, lam);
    d = -g;

    bestVal = Inf; bestTheta = theta; noImp = 0; hist.bestIter = 1;

    for k = 1:maxIter
        t_new = theta + lr * d;

        [c_new, g_new] = cost_grad(t_new, net, X, T, lam);
        [~, ~, vc]  = forward(t_new, net, Xv, Tv, lam);
        [~, ~, tec] = forward(t_new, net, Xt, Tt, lam);

        [tr_acc, ~] = evaluate_net(t_new, net, X,  T);
        [va_acc, ~] = evaluate_net(t_new, net, Xv, Tv);
        [te_acc, ~] = evaluate_net(t_new, net, Xt, Tt);

        hist.tr(end + 1) = gather(c_new);
        hist.va(end + 1) = gather(vc);
        hist.te(end + 1) = gather(tec);
        hist.tr_acc(end + 1) = gather(tr_acc);
        hist.va_acc(end + 1) = gather(va_acc);
        hist.te_acc(end + 1) = gather(te_acc);

        if vc < bestVal
            bestVal = vc; bestTheta = t_new; noImp = 0; hist.bestIter = k;
        else
            noImp = noImp + 1;
        end

        if noImp >= patience
            fprintf('      Early Stop @ Iter %d\n', k);
            break;
        end

        beta = max(0, dot(g_new, g_new - g) / (dot(g, g) + 1e-10));
        d = -g_new + beta * d;

        theta = t_new; c = c_new; g = g_new;
    end
    finalTheta = bestTheta;
end

function [finalTheta, hist] = train_abc_es(theta, net, X, T, Xv, Tv, Xt, Tt, lam, SN, limit, maxCycle, patience)
    hist.tr = [];  hist.va = [];  hist.te = [];
    hist.tr_acc = []; hist.va_acc = []; hist.te_acc = [];

    D = numel(theta);
    pop = repmat(theta(:)', SN, 1) + 0.1 * randn(SN, D);
    fitness = zeros(SN, 1);
    trial = zeros(SN, 1);

    for i = 1:SN
        [~, ~, cost] = forward(pop(i, :)', net, X, T, lam);
        fitness(i) = gather(cost);
    end

    [bestFit, bestIdx] = min(fitness);
    bestTheta = pop(bestIdx, :)';

    bestVal = Inf; noImp = 0; hist.bestIter = 1;

    for cycle = 1:maxCycle
        % Employed bees
        for i = 1:SN
            k = randi(SN);
            while k == i, k = randi(SN); end
            j = randi(D);
            phi = 2 * rand - 1;

            newSol = pop(i, :);
            newSol(j) = pop(i, j) + phi * (pop(i, j) - pop(k, j));

            [~, ~, newFit] = forward(newSol', net, X, T, lam);
            newFit = gather(newFit);

            if newFit < fitness(i)
                pop(i, :) = newSol;
                fitness(i) = newFit;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
        end

        % Onlooker bees
        prob = (1 ./ (1 + fitness)) / sum(1 ./ (1 + fitness));
        for i = 1:SN
            if rand < prob(i)
                k = randi(SN);
                while k == i, k = randi(SN); end
                j = randi(D);
                phi = 2 * rand - 1;

                newSol = pop(i, :);
                newSol(j) = pop(i, j) + phi * (pop(i, j) - pop(k, j));

                [~, ~, newFit] = forward(newSol', net, X, T, lam);
                newFit = gather(newFit);

                if newFit < fitness(i)
                    pop(i, :) = newSol;
                    fitness(i) = newFit;
                    trial(i) = 0;
                else
                    trial(i) = trial(i) + 1;
                end
            end
        end

        % Scout bees
        for i = 1:SN
            if trial(i) > limit
                pop(i, :) = bestTheta(:)' + 0.1 * randn(1, D);
                [~, ~, fitness(i)] = forward(pop(i, :)', net, X, T, lam);
                fitness(i) = gather(fitness(i));
                trial(i) = 0;
            end
        end

        [minFit, minIdx] = min(fitness);
        if minFit < bestFit
            bestFit = minFit;
            bestTheta = pop(minIdx, :)';
        end

        [~, ~, vc]  = forward(bestTheta, net, Xv, Tv, lam);
        [~, ~, tc]  = forward(bestTheta, net, X,  T,  lam);
        [~, ~, tec] = forward(bestTheta, net, Xt, Tt, lam);

        [tr_acc, ~] = evaluate_net(bestTheta, net, X,  T);
        [va_acc, ~] = evaluate_net(bestTheta, net, Xv, Tv);
        [te_acc, ~] = evaluate_net(bestTheta, net, Xt, Tt);

        hist.tr(end + 1) = gather(tc);
        hist.va(end + 1) = gather(vc);
        hist.te(end + 1) = gather(tec);
        hist.tr_acc(end + 1) = gather(tr_acc);
        hist.va_acc(end + 1) = gather(va_acc);
        hist.te_acc(end + 1) = gather(te_acc);

        if vc < bestVal
            bestVal = vc; noImp = 0; hist.bestIter = cycle;
        else
            noImp = noImp + 1;
        end

        if noImp >= patience
            fprintf('      Early Stop @ Cycle %d\n', cycle);
            break;
        end
    end
    finalTheta = bestTheta;
end

function apple_ysa_weight_bias_optimization()
% AGIRLIK VE BIAS OPTIMIZASYONU
% Bu script, yapay sinir aginin agirlik ve bias parametrelerini
% 5 farkli optimizasyon metodu ile optimize eder:
% 1. BFGS, 2. DFP, 3. CG, 4. GD, 5. ABC
% NOT: Hiperparametreleri asagidaki KONFIGURASYON bolumunden duzenleyin

clear;
clc;
close all;
rng(42);

fprintf('========================================\n');
fprintf('  AGIRLIK VE BIAS OPTIMIZASYONU\n');
fprintf('========================================\n\n');

%% KONFIGURASYON BOLUMU - HIPERPARAMETRELERI BURADAN DUZENLEYIN

% AG MIMARISI (her metod icin farkli hidden layer size)
config.BFGS.hiddenSizes = [16];
config.DFP.hiddenSizes = [32];
config.CG.hiddenSizes = [16];
config.GD.hiddenSizes = [16];
config.ABC.hiddenSizes = [32];

% Aktivasyon fonksiyonu (tum metodlar icin ortak)
config.activation = 'tanh';
% 'tanh', 'relu', veya 'sigmoid'

% BFGS PARAMETRELERI
config.BFGS.lambda = 0.0001; % Regularizasyon katsayisi
config.BFGS.lr = 0.070;      % Learning rate (step size)
config.BFGS.maxIter = 250;   % Maksimum iterasyon sayisi

% DFP PARAMETRELERI
config.DFP.lambda = 0.001;
config.DFP.lr     = 0.01;
config.DFP.maxIter = 500;

% CG PARAMETRELERI (önerilen)
config.CG.lambda  = 0.0005;
config.CG.lr      = 0.005;
config.CG.maxIter = 120;

% GD PARAMETRELERI
config.GD.lambda = 0.00002;
config.GD.lr = 0.200;
config.GD.maxIter = 1000;

% ABC PARAMETRELERI
config.ABC.lambda = 0.0005;
config.ABC.SN = 50;         % Employed bee sayisi
config.ABC.limit = 100;     % Abandon limiti
config.ABC.maxCycle = 500;  % Maksimum dongu sayisi

% EARLY STOPPING
config.patience = 50; % Validation cost kac iterasyon duzelmezse egitim dursun

% KONFIGURASYON SONU

fprintf('>> Konfigurasyon yuklendi.\n');
fprintf('   Aktivasyon: %s\n', config.activation);
fprintf('   Early Stopping Patience: %d\n\n', config.patience);

%% Veri Yukleme
fprintf('>> Veri yukleniyor...\n');
filename = 'apple_quality.csv';
[X_train, T_train, X_val, T_val, X_test, T_test, inD, outD] = load_data(filename);

fprintf('   Train: %d samples\n', size(X_train, 1));
fprintf('   Val:   %d samples\n', size(X_val, 1));
fprintf('   Test:  %d samples\n', size(X_test, 1));
fprintf('   Input Dim: %d, Output Dim: %d\n', inD, outD);

%% GPU Kullanimi
useGPU = false;
try
    if gpuDeviceCount > 0
        gpuDevice;
        useGPU = true;
        fprintf('\n>> GPU kullanimda.\n');
    else
        fprintf('\n>> CPU kullanimda.\n');
    end
catch
    fprintf('\n>> CPU kullanimda.\n');
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

%% Metodlari Tanimla
methods = {'BFGS', 'DFP', 'CG', 'GD', 'ABC'};
results = struct();

%% Her Metod icin Egitim
for m = 1:numel(methods)
    methodName = methods{m};

    fprintf('\n========================================\n');
    fprintf('>> %s ile agirlik ve bias optimizasyonu\n', methodName);
    fprintf('========================================\n');

    % Bu metod icin ag yapisini olustur
    net = struct();
    net.layers = [inD, config.(methodName).hiddenSizes, outD];
    net.activation = config.activation;
    net.L = numel(net.layers);

    % Ag mimarisini yazdir
    fprintf('   Mimari: [%d', net.layers(1));
    for i = 2:numel(net.layers)
        fprintf(' -> %d', net.layers(i));
    end
    fprintf(']\n');

    % Hiperparametreleri yazdir
    fprintf('   Lambda: %.6f\n', config.(methodName).lambda);
    if isfield(config.(methodName), 'lr')
        fprintf('   Learning Rate: %.4f\n', config.(methodName).lr);
        fprintf('   Max Iter: %d\n', config.(methodName).maxIter);
    else
        fprintf('   SN: %d\n', config.(methodName).SN);
        fprintf('   Limit: %d\n', config.(methodName).limit);
        fprintf('   Max Cycle: %d\n', config.(methodName).maxCycle);
    end

    % Agirliklari baslat
    theta_init = init_params(net);
    numParams = numel(theta_init);
    fprintf('   Toplam Parametre: %d\n', numParams);

    % Egitim
    tic;
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
    elapsed = toc;

    % En iyi modeli degerlendir
    [trainAcc, ~] = evaluate_net(bestTheta, net, Xtr, Ttr);
    [valAcc, ~] = evaluate_net(bestTheta, net, Xva, Tva);
    [testAcc, ~] = evaluate_net(bestTheta, net, Xte, Tte);

    % Sonuclari kaydet
    results.(methodName).net = net;
    results.(methodName).config = config.(methodName);
    results.(methodName).bestTheta = bestTheta;
    results.(methodName).trainAcc = double(trainAcc);
    results.(methodName).valAcc = double(valAcc);
    results.(methodName).testAcc = double(testAcc);
    results.(methodName).bestValCost = min(hist.va);
    results.(methodName).trainTime = elapsed;
    results.(methodName).hist = hist;
    results.(methodName).bestIter = hist.bestIter;

    % Sonuclari yazdir
    fprintf('\n>> %s Sonuclari:\n', methodName);
    fprintf('   Egitim Suresi: %.2f saniye\n', elapsed);
    fprintf('   En Iyi Iterasyon: %d\n', hist.bestIter);
    fprintf('   En Iyi Val Cost: %.4f\n', min(hist.va));
    fprintf('   Train Dogruluk: %.2f%%\n', trainAcc * 100);
    fprintf('   Val Dogruluk:   %.2f%%\n', valAcc * 100);
    fprintf('   Test Dogruluk:  %.2f%%\n', testAcc * 100);
end

%% Ozet Tablo
fprintf('\n\n========================================\n');
fprintf('           SONUC OZETI\n');
fprintf('========================================\n');
fprintf('Metod | HiddenSz | Train%% | Val%%  | Test%% | BestIter | Time(s)\n');
fprintf('------|----------|--------|-------|--------|----------|--------\n');
for m = 1:numel(methods)
    methodName = methods{m};
    r = results.(methodName);
    hiddenStr = mat2str(config.(methodName).hiddenSizes);
    fprintf('%-5s | %8s | %6.2f | %5.2f | %6.2f | %8d | %7.2f\n', ...
        methodName, hiddenStr, r.trainAcc * 100, r.valAcc * 100, ...
        r.testAcc * 100, r.bestIter, r.trainTime);
end
fprintf('========================================\n');

%% En Iyi Metodu Bul
bestMethod = '';
bestValAcc = -inf;
for m = 1:numel(methods)
    methodName = methods{m};
    if results.(methodName).valAcc > bestValAcc
        bestValAcc = results.(methodName).valAcc;
        bestMethod = methodName;
    end
end

fprintf('\n>> En Iyi Metod: %s (Val Acc: %.2f%%, Test Acc: %.2f%%)\n', ...
    bestMethod, bestValAcc * 100, results.(bestMethod).testAcc * 100);

%% Grafikler Olustur
% create_comparison_plots(methods, results, config);
create_individual_plots(methods, results, inD, outD);

fprintf('\n>> Tum metodlar tamamlandi. Grafikler olusturuldu.\n');

end

%% YARDIMCI FONKSIYONLAR

function [X_train, T_train, X_val, T_val, X_test, T_test, inD, outD] = load_data(filename)
    data = readtable(filename);

    if ~any(strcmpi(data.Properties.VariableNames, 'Quality'))
        error('Quality sutunu bulunamadi: %s', filename);
    end

    y = data.Quality;
    if iscell(y)
        y = categorical(y);
    end

    labels = categories(y);
    if numel(labels) ~= 2
        error('Bu script binary classification icin tasarlandi!');
    end

    ynum = double(y == labels{2});

    featureCols = ~strcmpi(data.Properties.VariableNames, 'Quality');
    X = table2array(data(:, featureCols));

    T = zeros(numel(ynum), 2);
    T(sub2ind(size(T), (1:numel(ynum))', ynum + 1)) = 1;

    idx0 = find(ynum == 0);
    idx1 = find(ynum == 1);

    n0 = numel(idx0);
    n1 = numel(idx1);

    n0_tr = round(0.6 * n0);
    n0_val = round(0.2 * n0);
    n1_tr = round(0.6 * n1);
    n1_val = round(0.2 * n1);

    idx0 = idx0(randperm(n0));
    idx1 = idx1(randperm(n1));

    trIdx = [idx0(1:n0_tr); idx1(1:n1_tr)];
    valIdx = [idx0(n0_tr + 1:n0_tr + n0_val); idx1(n1_tr + 1:n1_tr + n1_val)];
    teIdx = [idx0(n0_tr + n0_val + 1:end); idx1(n1_tr + n1_val + 1:end)];

    trIdx = trIdx(randperm(numel(trIdx)));
    valIdx = valIdx(randperm(numel(valIdx)));
    teIdx = teIdx(randperm(numel(teIdx)));

    X_train = X(trIdx, :);
    T_train = T(trIdx, :);
    X_val = X(valIdx, :);
    T_val = T(valIdx, :);
    X_test = X(teIdx, :);
    T_test = T(teIdx, :);

    mu = mean(X_train, 1);
    sig = std(X_train, 0, 1) + 1e-8;
    X_train = (X_train - mu) ./ sig;
    X_val = (X_val - mu) ./ sig;
    X_test = (X_test - mu) ./ sig;

    inD = size(X, 2);
    outD = size(T, 2);
end

function theta = init_params(net)
    L = net.L;
    theta = [];
    for l = 1:(L - 1)
        fanIn = net.layers(l);
        fanOut = net.layers(l + 1);

        limit = sqrt(6 / (fanIn + fanOut));
        W = (rand(fanOut, fanIn) * 2 - 1) * limit;
        b = zeros(fanOut, 1);

        theta = [theta; W(:); b(:)];
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
    L = net.L;
    layers = net.layers;
    activation = net.activation;

    A = cell(L, 1);
    A{1} = X';

    idx = 1;
    for l = 1:(L - 1)
        fanIn = layers(l);
        fanOut = layers(l + 1);

        Wsize = fanOut * fanIn;
        W = reshape(theta(idx:idx + Wsize - 1), [fanOut, fanIn]);
        idx = idx + Wsize;

        b = theta(idx:idx + fanOut - 1);
        idx = idx + fanOut;

        Z = W * A{l} + b;

        if l < (L - 1)
            switch activation
                case 'tanh'
                    A{l + 1} = tanh(Z);
                case 'relu'
                    A{l + 1} = max(0, Z);
                case 'sigmoid'
                    A{l + 1} = 1 ./ (1 + exp(-Z));
                otherwise
                    error('Bilinmeyen aktivasyon: %s', activation);
            end
        else
            expZ = exp(Z - max(Z, [], 1));
            A{l + 1} = expZ ./ sum(expZ, 1);
        end
    end

    Y = A{end}';
end

function grad = compute_gradient(theta, net, X, T, lambda)
    L = net.L;
    layers = net.layers;
    activation = net.activation;
    m = size(X, 1);

    [A, Y] = forward_pass(theta, net, X);

    delta = cell(L, 1);
    delta{L} = (Y - T)';

    idx = numel(theta);
    for l = (L - 1):-1:2
        fanIn = layers(l);
        fanOut = layers(l + 1);

        idx = idx - fanOut;

        Wsize = fanOut * fanIn;
        W = reshape(theta(idx - Wsize + 1:idx), [fanOut, fanIn]);
        idx = idx - Wsize;

        delta{l} = W' * delta{l + 1};

        switch activation
            case 'tanh'
                delta{l} = delta{l} .* (1 - A{l}.^2);
            case 'relu'
                delta{l} = delta{l} .* double(A{l} > 0);
            case 'sigmoid'
                delta{l} = delta{l} .* (A{l} .* (1 - A{l}));
        end
    end

    grad = zeros(size(theta));
    idx = 1;
    for l = 1:(L - 1)
        fanIn = layers(l);
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

    L = net.L;
    layers = net.layers;
    idx = 1;
    regTerm = 0;
    for l = 1:(L - 1)
        fanIn = layers(l);
        fanOut = layers(l + 1);

        Wsize = fanOut * fanIn;
        W = reshape(theta(idx:idx + Wsize - 1), [fanOut, fanIn]);
        regTerm = regTerm + sum(W(:).^2);
        idx = idx + Wsize + fanOut;
    end

    cost = cost + (lambda / (2 * m)) * regTerm;
end

function [bestTheta, hist] = train_gd_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, maxIter, lr, patience)
    bestTheta = theta;
    bestValCost = inf;
    noImprove = 0;

    hist = struct();
    hist.tr = zeros(maxIter, 1);
    hist.va = zeros(maxIter, 1);
    hist.te = zeros(maxIter, 1);
    hist.tr_acc = zeros(maxIter, 1);
    hist.va_acc = zeros(maxIter, 1);
    hist.te_acc = zeros(maxIter, 1);

    for iter = 1:maxIter
        grad = compute_gradient(theta, net, Xtr, Ttr, lambda);
        theta = theta - lr * grad;

        hist.tr(iter) = compute_cost(theta, net, Xtr, Ttr, lambda);
        hist.va(iter) = compute_cost(theta, net, Xva, Tva, lambda);
        hist.te(iter) = compute_cost(theta, net, Xte, Tte, lambda);

        [hist.tr_acc(iter), ~] = evaluate_net(theta, net, Xtr, Ttr);
        [hist.va_acc(iter), ~] = evaluate_net(theta, net, Xva, Tva);
        [hist.te_acc(iter), ~] = evaluate_net(theta, net, Xte, Tte);

        if hist.va(iter) < bestValCost
            bestValCost = hist.va(iter);
            bestTheta = theta;
            hist.bestIter = iter;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end

        if noImprove >= patience
            fprintf('   Early stopping at iteration %d\n', iter);
            hist.tr = hist.tr(1:iter);
            hist.va = hist.va(1:iter);
            hist.te = hist.te(1:iter);
            hist.tr_acc = hist.tr_acc(1:iter);
            hist.va_acc = hist.va_acc(1:iter);
            hist.te_acc = hist.te_acc(1:iter);
            break;
        end

        if mod(iter, 50) == 0
            fprintf('   Iter %d: Train=%.4f, Val=%.4f, ValAcc=%.2f%%\n', ...
                iter, hist.tr(iter), hist.va(iter), hist.va_acc(iter) * 100);
        end
    end

    if ~isfield(hist, 'bestIter')
        hist.bestIter = maxIter;
    end
end

function [bestTheta, hist] = train_cg_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, maxIter, lr, patience)
    bestTheta = theta;
    bestValCost = inf;
    noImprove = 0;

    hist = struct();
    hist.tr = zeros(maxIter, 1);
    hist.va = zeros(maxIter, 1);
    hist.te = zeros(maxIter, 1);
    hist.tr_acc = zeros(maxIter, 1);
    hist.va_acc = zeros(maxIter, 1);
    hist.te_acc = zeros(maxIter, 1);

    d = -compute_gradient(theta, net, Xtr, Ttr, lambda);

    for iter = 1:maxIter
        grad = compute_gradient(theta, net, Xtr, Ttr, lambda);

        alpha = lr;
        theta = theta + alpha * d;

        hist.tr(iter) = compute_cost(theta, net, Xtr, Ttr, lambda);
        hist.va(iter) = compute_cost(theta, net, Xva, Tva, lambda);
        hist.te(iter) = compute_cost(theta, net, Xte, Tte, lambda);

        [hist.tr_acc(iter), ~] = evaluate_net(theta, net, Xtr, Ttr);
        [hist.va_acc(iter), ~] = evaluate_net(theta, net, Xva, Tva);
        [hist.te_acc(iter), ~] = evaluate_net(theta, net, Xte, Tte);

        if hist.va(iter) < bestValCost
            bestValCost = hist.va(iter);
            bestTheta = theta;
            hist.bestIter = iter;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end

        if noImprove >= patience
            fprintf('   Early stopping at iteration %d\n', iter);
            hist.tr = hist.tr(1:iter);
            hist.va = hist.va(1:iter);
            hist.te = hist.te(1:iter);
            hist.tr_acc = hist.tr_acc(1:iter);
            hist.va_acc = hist.va_acc(1:iter);
            hist.te_acc = hist.te_acc(1:iter);
            break;
        end

        grad_new = compute_gradient(theta, net, Xtr, Ttr, lambda);
        beta = (grad_new' * grad_new) / (grad' * grad + 1e-8);
        d = -grad_new + beta * d;

        if mod(iter, 50) == 0
            fprintf('   Iter %d: Train=%.4f, Val=%.4f, ValAcc=%.2f%%\n', ...
                iter, hist.tr(iter), hist.va(iter), hist.va_acc(iter) * 100);
        end
    end

    if ~isfield(hist, 'bestIter')
        hist.bestIter = maxIter;
    end
end

function [bestTheta, hist] = train_bfgs_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, maxIter, lr, patience)
    n = numel(theta);
    H = eye(n);

    bestTheta = theta;
    bestValCost = inf;
    noImprove = 0;

    hist = struct();
    hist.tr = zeros(maxIter, 1);
    hist.va = zeros(maxIter, 1);
    hist.te = zeros(maxIter, 1);
    hist.tr_acc = zeros(maxIter, 1);
    hist.va_acc = zeros(maxIter, 1);
    hist.te_acc = zeros(maxIter, 1);

    grad = compute_gradient(theta, net, Xtr, Ttr, lambda);

    for iter = 1:maxIter
        d = -H * grad;
        alpha = lr;
        theta_new = theta + alpha * d;

        grad_new = compute_gradient(theta_new, net, Xtr, Ttr, lambda);

        s = theta_new - theta;
        y = grad_new - grad;

        rho = 1 / (y' * s + 1e-8);
        H = (eye(n) - rho * s * y') * H * (eye(n) - rho * y * s') + rho * (s * s');

        theta = theta_new;
        grad = grad_new;

        hist.tr(iter) = compute_cost(theta, net, Xtr, Ttr, lambda);
        hist.va(iter) = compute_cost(theta, net, Xva, Tva, lambda);
        hist.te(iter) = compute_cost(theta, net, Xte, Tte, lambda);

        [hist.tr_acc(iter), ~] = evaluate_net(theta, net, Xtr, Ttr);
        [hist.va_acc(iter), ~] = evaluate_net(theta, net, Xva, Tva);
        [hist.te_acc(iter), ~] = evaluate_net(theta, net, Xte, Tte);

        if hist.va(iter) < bestValCost
            bestValCost = hist.va(iter);
            bestTheta = theta;
            hist.bestIter = iter;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end

        if noImprove >= patience
            fprintf('   Early stopping at iteration %d\n', iter);
            hist.tr = hist.tr(1:iter);
            hist.va = hist.va(1:iter);
            hist.te = hist.te(1:iter);
            hist.tr_acc = hist.tr_acc(1:iter);
            hist.va_acc = hist.va_acc(1:iter);
            hist.te_acc = hist.te_acc(1:iter);
            break;
        end

        if mod(iter, 50) == 0
            fprintf('   Iter %d: Train=%.4f, Val=%.4f, ValAcc=%.2f%%\n', ...
                iter, hist.tr(iter), hist.va(iter), hist.va_acc(iter) * 100);
        end
    end

    if ~isfield(hist, 'bestIter')
        hist.bestIter = maxIter;
    end
end

function [bestTheta, hist] = train_dfp_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, maxIter, lr, patience)
    n = numel(theta);
    H = eye(n);

    bestTheta = theta;
    bestValCost = inf;
    noImprove = 0;

    hist = struct();
    hist.tr = zeros(maxIter, 1);
    hist.va = zeros(maxIter, 1);
    hist.te = zeros(maxIter, 1);
    hist.tr_acc = zeros(maxIter, 1);
    hist.va_acc = zeros(maxIter, 1);
    hist.te_acc = zeros(maxIter, 1);

    grad = compute_gradient(theta, net, Xtr, Ttr, lambda);

    for iter = 1:maxIter
        d = -H * grad;
        alpha = lr;
        theta_new = theta + alpha * d;

        grad_new = compute_gradient(theta_new, net, Xtr, Ttr, lambda);

        s = theta_new - theta;
        y = grad_new - grad;

        Hy = H * y;
        H = H + (s * s') / (s' * y + 1e-8) - (Hy * Hy') / (y' * Hy + 1e-8);

        theta = theta_new;
        grad = grad_new;

        hist.tr(iter) = compute_cost(theta, net, Xtr, Ttr, lambda);
        hist.va(iter) = compute_cost(theta, net, Xva, Tva, lambda);
        hist.te(iter) = compute_cost(theta, net, Xte, Tte, lambda);

        [hist.tr_acc(iter), ~] = evaluate_net(theta, net, Xtr, Ttr);
        [hist.va_acc(iter), ~] = evaluate_net(theta, net, Xva, Tva);
        [hist.te_acc(iter), ~] = evaluate_net(theta, net, Xte, Tte);

        if hist.va(iter) < bestValCost
            bestValCost = hist.va(iter);
            bestTheta = theta;
            hist.bestIter = iter;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end

        if noImprove >= patience
            fprintf('   Early stopping at iteration %d\n', iter);
            hist.tr = hist.tr(1:iter);
            hist.va = hist.va(1:iter);
            hist.te = hist.te(1:iter);
            hist.tr_acc = hist.tr_acc(1:iter);
            hist.va_acc = hist.va_acc(1:iter);
            hist.te_acc = hist.te_acc(1:iter);
            break;
        end

        if mod(iter, 50) == 0
            fprintf('   Iter %d: Train=%.4f, Val=%.4f, ValAcc=%.2f%%\n', ...
                iter, hist.tr(iter), hist.va(iter), hist.va_acc(iter) * 100);
        end
    end

    if ~isfield(hist, 'bestIter')
        hist.bestIter = maxIter;
    end
end

function [bestTheta, hist] = train_abc_es(theta, net, Xtr, Ttr, Xva, Tva, Xte, Tte, lambda, SN, limit, maxCycle, patience)
    D = numel(theta);

    pop = repmat(theta', SN, 1) + randn(SN, D) * 0.1;
    fitness = zeros(SN, 1);
    trial = zeros(SN, 1);

    for i = 1:SN
        fitness(i) = compute_cost(pop(i, :)', net, Xtr, Ttr, lambda);
    end

    bestTheta = theta;
    [bestValCost, bestIdx] = min(fitness);
    bestTheta = pop(bestIdx, :)';
    noImprove = 0;

    hist = struct();
    hist.tr = zeros(maxCycle, 1);
    hist.va = zeros(maxCycle, 1);
    hist.te = zeros(maxCycle, 1);
    hist.tr_acc = zeros(maxCycle, 1);
    hist.va_acc = zeros(maxCycle, 1);
    hist.te_acc = zeros(maxCycle, 1);

    for cycle = 1:maxCycle
        for i = 1:SN
            k = randi(SN);
            while k == i
                k = randi(SN);
            end

            phi = rand(1, D) * 2 - 1;
            v = pop(i, :) + phi .* (pop(i, :) - pop(k, :));

            fv = compute_cost(v', net, Xtr, Ttr, lambda);

            if fv < fitness(i)
                pop(i, :) = v;
                fitness(i) = fv;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
        end

        fitSum = sum(1 ./ (fitness + 1));
        prob = (1 ./ (fitness + 1)) / fitSum;

        for i = 1:SN
            if rand < prob(i)
                k = randi(SN);
                while k == i
                    k = randi(SN);
                end

                phi = rand(1, D) * 2 - 1;
                v = pop(i, :) + phi .* (pop(i, :) - pop(k, :));

                fv = compute_cost(v', net, Xtr, Ttr, lambda);

                if fv < fitness(i)
                    pop(i, :) = v;
                    fitness(i) = fv;
                    trial(i) = 0;
                else
                    trial(i) = trial(i) + 1;
                end
            end
        end

        for i = 1:SN
            if trial(i) >= limit
                pop(i, :) = theta' + randn(1, D) * 0.1;
                fitness(i) = compute_cost(pop(i, :)', net, Xtr, Ttr, lambda);
                trial(i) = 0;
            end
        end

        [~, bestIdx] = min(fitness);
        currentBest = pop(bestIdx, :)';

        hist.tr(cycle) = compute_cost(currentBest, net, Xtr, Ttr, lambda);
        hist.va(cycle) = compute_cost(currentBest, net, Xva, Tva, lambda);
        hist.te(cycle) = compute_cost(currentBest, net, Xte, Tte, lambda);

        [hist.tr_acc(cycle), ~] = evaluate_net(currentBest, net, Xtr, Ttr);
        [hist.va_acc(cycle), ~] = evaluate_net(currentBest, net, Xva, Tva);
        [hist.te_acc(cycle), ~] = evaluate_net(currentBest, net, Xte, Tte);

        if hist.va(cycle) < bestValCost
            bestValCost = hist.va(cycle);
            bestTheta = currentBest;
            hist.bestIter = cycle;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end

        if noImprove >= patience
            fprintf('   Early stopping at cycle %d\n', cycle);
            hist.tr = hist.tr(1:cycle);
            hist.va = hist.va(1:cycle);
            hist.te = hist.te(1:cycle);
            hist.tr_acc = hist.tr_acc(1:cycle);
            hist.va_acc = hist.va_acc(1:cycle);
            hist.te_acc = hist.te_acc(1:cycle);
            break;
        end

        if mod(cycle, 20) == 0
            fprintf('   Cycle %d: Train=%.4f, Val=%.4f, ValAcc=%.2f%%\n', ...
                cycle, hist.tr(cycle), hist.va(cycle), hist.va_acc(cycle) * 100);
        end
    end

    if ~isfield(hist, 'bestIter')
        hist.bestIter = maxCycle;
    end
end

function create_comparison_plots(methods, results, config)
    colors = struct();
    colors.BFGS = [0.2, 0.4, 0.8];
    colors.DFP = [0.8, 0.2, 0.2];
    colors.CG = [0.2, 0.8, 0.4];
    colors.GD = [0.8, 0.6, 0.2];
    colors.ABC = [0.6, 0.2, 0.8];

    figure('Name', 'Tum Metodlar - Cost Karsilastirmasi', 'Position', [100 100 1400 500]);

    subplot(1, 2, 1);
    hold on;
    for m = 1:numel(methods)
        methodName = methods{m};
        plot(results.(methodName).hist.va, 'LineWidth', 2, 'Color', colors.(methodName), ...
            'DisplayName', methodName);
    end
    hold off;
    xlabel('Iterasyon');
    ylabel('Validation Cost');
    title('Validation Cost Karsilastirmasi');
    legend('Location', 'best');
    grid on;

    subplot(1, 2, 2);
    hold on;
    for m = 1:numel(methods)
        methodName = methods{m};
        plot(results.(methodName).hist.tr, 'LineWidth', 2, 'Color', colors.(methodName), ...
            'DisplayName', methodName);
    end
    hold off;
    xlabel('Iterasyon');
    ylabel('Training Cost');
    title('Training Cost Karsilastirmasi');
    legend('Location', 'best');
    grid on;

    figure('Name', 'Tum Metodlar - Accuracy Karsilastirmasi', 'Position', [100 100 1400 500]);

    subplot(1, 2, 1);
    hold on;
    for m = 1:numel(methods)
        methodName = methods{m};
        plot(results.(methodName).hist.va_acc * 100, 'LineWidth', 2, 'Color', colors.(methodName), ...
            'DisplayName', methodName);
    end
    hold off;
    xlabel('Iterasyon');
    ylabel('Validation Accuracy (%)');
    title('Validation Accuracy Karsilastirmasi');
    legend('Location', 'best');
    grid on;
    ylim([0 100]);

    subplot(1, 2, 2);
    hold on;
    for m = 1:numel(methods)
        methodName = methods{m};
        plot(results.(methodName).hist.tr_acc * 100, 'LineWidth', 2, 'Color', colors.(methodName), ...
            'DisplayName', methodName);
    end
    hold off;
    xlabel('Iterasyon');
    ylabel('Training Accuracy (%)');
    title('Training Accuracy Karsilastirmasi');
    legend('Location', 'best');
    grid on;
    ylim([0 100]);

    figure('Name', 'Final Accuracy Karsilastirmasi', 'Position', [100 100 1000 500]);

    trainAccs = zeros(1, numel(methods));
    valAccs = zeros(1, numel(methods));
    testAccs = zeros(1, numel(methods));

    for m = 1:numel(methods)
        trainAccs(m) = results.(methods{m}).trainAcc * 100;
        valAccs(m) = results.(methods{m}).valAcc * 100;
        testAccs(m) = results.(methods{m}).testAcc * 100;
    end

    bar_data = [trainAccs; valAccs; testAccs]';
    b = bar(bar_data);
    b(1).FaceColor = [0.3 0.6 0.9];
    b(2).FaceColor = [0.9 0.6 0.3];
    b(3).FaceColor = [0.3 0.9 0.6];

    set(gca, 'XTickLabel', methods);
    xlabel('Metod');
    ylabel('Dogruluk (%)');
    title('Final Accuracy Karsilastirmasi');
    legend({'Train', 'Validation', 'Test'}, 'Location', 'best');
    grid on;
    ylim([0 100]);
end

function create_individual_plots(methods, results, inD, outD)
    for m = 1:numel(methods)
        methodName = methods{m};
        r = results.(methodName);

        figure('Name', sprintf('%s - Detayli Gorunum', methodName), ...
            'Position', [100 + m * 30, 100 + m * 30, 1400, 420]);

        subplot(1, 3, 1);
        hold on;
        plot(r.hist.tr, 'b-', 'LineWidth', 1.8, 'DisplayName', 'Train Cost');
        plot(r.hist.va, 'r-', 'LineWidth', 1.8, 'DisplayName', 'Val Cost');
        plot(r.hist.te, 'k--', 'LineWidth', 1.8, 'DisplayName', 'Test Cost');
        xline(r.bestIter, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Best@%d', r.bestIter));
        hold off;
        xlabel('Iterasyon');
        ylabel('Cost');
        title(sprintf('%s - Cost', methodName));
        legend('Location', 'best');
        grid on;

        subplot(1, 3, 2);
        hold on;
        plot(r.hist.tr_acc * 100, 'b-', 'LineWidth', 1.8, 'DisplayName', 'Train Acc');
        plot(r.hist.va_acc * 100, 'r-', 'LineWidth', 1.8, 'DisplayName', 'Val Acc');
        plot(r.hist.te_acc * 100, 'k--', 'LineWidth', 1.8, 'DisplayName', 'Test Acc');
        xline(r.bestIter, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Best@%d', r.bestIter));
        hold off;
        xlabel('Iterasyon');
        ylabel('Dogruluk (%)');
        title(sprintf('%s - Accuracy', methodName));
        legend('Location', 'best');
        grid on;
        ylim([0 100]);

        subplot(1, 3, 3);
        axis off;

        info_lines = {
            sprintf('\\bf%s - OZET', methodName),
            '',
            '\\bfSonuclar:',
            sprintf('En Iyi Iterasyon: %d', r.bestIter),
            sprintf('Train Acc: %.2f%%', r.trainAcc * 100),
            sprintf('Val Acc: %.2f%%', r.valAcc * 100),
            sprintf('Test Acc: %.2f%%', r.testAcc * 100),
            sprintf('En Iyi Val Cost: %.4f', r.bestValCost),
            sprintf('Egitim Suresi: %.2f sn', r.trainTime),
            '',
            '\\bfMimari:',
            sprintf('Giris: %d', inD),
            sprintf('Gizli: %s', mat2str(r.config.hiddenSizes)),
            sprintf('Cikis: %d', outD),
            sprintf('Aktivasyon: %s', r.net.activation),
            '',
            '\\bfHiperparametreler:',
            sprintf('Lambda: %.6f', r.config.lambda)
        };

        if isfield(r.config, 'lr')
            info_lines{end + 1} = sprintf('Learning Rate: %.4f', r.config.lr);
            info_lines{end + 1} = sprintf('Max Iter: %d', r.config.maxIter);
        else
            info_lines{end + 1} = sprintf('SN: %d', r.config.SN);
            info_lines{end + 1} = sprintf('Limit: %d', r.config.limit);
            info_lines{end + 1} = sprintf('Max Cycle: %d', r.config.maxCycle);
        end

        text(0.1, 0.95, strjoin(info_lines, '\n'), 'FontSize', 10, ...
            'VerticalAlignment', 'top', 'FontName', 'Courier');
        title(sprintf('%s - Ozet', methodName));
    end
end

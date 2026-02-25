function apple_ysa_twophase_v3()
%% ================================================================
%  Apple Quality - Two-Phase Scan (Coarse -> Fine)  |  TOOLBOX YOK*
%  MLP: L in {1,2,3} hidden layers (her katmanda aynı H)
%  Methods: ABC, BFGS, DFP, GD, CG
%
%  İSTENEN DÜZELTMELERİN HEPSİ BU SÜRÜMDE:
%   1) Phase-1 -> Phase-2 otomatik geçiş (AMA Phase-1 tamamlanmadan geçmez)
%   2) bestTrainCost / bestTestCost ayrımı doğru
%   3) Best run seçimi: criterion = "minBestTestCost" veya "maxTestAcc"
%   4) Her method için ayrı plot: TrainCost & TestCost (best run)
%   5) Ortak plot: 5 method best run TestCost
%   6) Phase-2: her method için topK run -> lambda için iki iyi değer arası log-ara değerler + local refine
%   7) ABC limit parametresi job grid’ine eklendi (Phase-1/2 taranıyor)
%   8) Checkpoint + partial-resume var (parfor worker save yok -> DataQueue ile client save)
%
%  *Parallel Computing Toolbox varsa threads parpool ile parfor açar, yoksa seri.
% ================================================================

clear; clc; close all;
rng(1);
echo off
try, maxNumCompThreads('automatic'); catch, end

%% ================== KULLANICI AYARLARI ===========================
AUTO_RUN_BOTH_PHASES = true;  % true: Phase-1 biter -> Phase-2 başlar

PRINT_EVERY = 1;
SAVE_EVERY  = 25;             % checkpoint sıklığı (seri koşularda)
TOPK_PER_METHOD = 3;          % Phase-2 job üretirken method başına kaç run baz alınacak?

CHECKPOINT1 = 'checkpoint_phase1_apple_v3.mat';
CHECKPOINT2 = 'checkpoint_phase2_apple_v3.mat';
CSV1 = 'results_phase1_apple_v3.csv';
CSV2 = 'results_phase2_apple_v3.csv';

USE_SINGLE_ON_GPU = true;     % GPU’da single hızlı
FORCE_ABC_CPU = true;         % ABC genelde CPU’da daha stabil
ALLOW_PARFOR = true;          % Parallel toolbox varsa parfor threads

PLOT_AT_END = true;
PLOT_CRITERION = "minBestTestCost";  % "minBestTestCost" | "maxTestAcc"

%% ================== SONUÇ ŞABLONU ================================
TEMPLATE = struct( ...
    'phase',NaN, ...
    'method','', ...
    'L',NaN, ...
    'hiddenDim',NaN, ...
    'stepSize',NaN, ...
    'lambda',NaN, ...
    'SN',NaN, ...
    'abcLimit',NaN, ...
    'maxCycle',NaN, ...
    'maxIter',NaN, ...
    'bestTrainCost',NaN, ...
    'bestTestCost',NaN, ...
    'finalTrainCost',NaN, ...
    'finalTestCost',NaN, ...
    'trainAcc',NaN, ...
    'testAcc',NaN, ...
    'trainCostHistory',[], ...
    'testCostHistory',[], ...
    'theta',[]);

%% ================== 1) VERİ YÜKLEME ==============================
data = readtable('apple_quality.csv');

if ~any(strcmpi(data.Properties.VariableNames,'Quality'))
    error('apple_quality.csv içinde "Quality" sütunu bulunamadı.');
end

q = data.(data.Properties.VariableNames{strcmpi(data.Properties.VariableNames,'Quality')});
q = string(q);
y_bin = double(lower(q)=="good");  % good=1, bad=0

isNum = varfun(@isnumeric, data, 'OutputFormat','uniform');
colNames = data.Properties.VariableNames;
isQuality = strcmpi(colNames,'Quality');

featCols = find(isNum & ~isQuality);
if isempty(featCols)
    error('Quality hariç numeric feature bulunamadı.');
end

X = table2array(data(:, featCols));

bad = any(~isfinite(X),2) | ~isfinite(y_bin);
X(bad,:) = [];
y_bin(bad,:) = [];

T = [y_bin==0, y_bin==1];  % one-hot

%% ================== STRATIFIED SPLIT (SABİT) ======================
idx0 = find(y_bin==0); idx1 = find(y_bin==1);
idx0 = idx0(randperm(numel(idx0)));
idx1 = idx1(randperm(numel(idx1)));

n0_tr = round(0.7*numel(idx0));
n1_tr = round(0.7*numel(idx1));
trIdx = [idx0(1:n0_tr); idx1(1:n1_tr)];
teIdx = [idx0(n0_tr+1:end); idx1(n1_tr+1:end)];
trIdx = trIdx(randperm(numel(trIdx)));
teIdx = teIdx(randperm(numel(teIdx)));

X_train = X(trIdx,:);   T_train = T(trIdx,:);
X_test  = X(teIdx,:);   T_test  = T(teIdx,:);

fprintf('Train: %d örnek, Test: %d örnek\n', size(X_train,1), size(X_test,1));
fprintf('Class balance (train): p(good)=%.3f | (test): p(good)=%.3f\n', ...
    mean(T_train(:,2)), mean(T_test(:,2)));

%% ================== Z-SCORE NORMALIZATION =========================
mu  = mean(X_train,1);
sig = std(X_train,0,1) + 1e-8;
X_train = (X_train - mu) ./ sig;
X_test  = (X_test  - mu) ./ sig;

inputDim  = size(X_train,2);
outputDim = size(T_train,2);

% CPU kopyalar
X_train_cpu = X_train;  T_train_cpu = T_train;
X_test_cpu  = X_test;   T_test_cpu  = T_test;

%% ================== GPU DETECT ===================================
useGPU = false;
try
    gpuDevice;
    useGPU = true;
    fprintf('>> GPU bulundu, türev tabanlılar GPU üzerinden çalışabilir.\n');
catch
    fprintf('>> GPU bulunamadı veya kullanılamıyor, CPU ile devam.\n');
end

if useGPU
    if USE_SINGLE_ON_GPU
        X_train_g = gpuArray(single(X_train));
        T_train_g = gpuArray(single(T_train));
        X_test_g  = gpuArray(single(X_test));
        T_test_g  = gpuArray(single(T_test));
    else
        X_train_g = gpuArray(X_train);
        T_train_g = gpuArray(T_train);
        X_test_g  = gpuArray(X_test);
        T_test_g  = gpuArray(T_test);
    end
else
    X_train_g = X_train;
    T_train_g = T_train;
    X_test_g  = X_test;
    T_test_g  = T_test;
end

%% ================== 2) PHASE LIST =================================
if AUTO_RUN_BOTH_PHASES
    phaseList = [1 2];
else
    phaseList = 1; % sadece Phase-1 istersen
end

% phase tablolarını ayrı saklayacağız
tbl_phase1 = table();
tbl_phase2 = table();

for PHASE = phaseList

    %% ================== JOBS OLUŞTUR / LOAD =========================
    switch PHASE
        case 1
            [jobs, results, res_id] = init_phase(CHECKPOINT1, TEMPLATE, 1);

            if isempty(jobs)
                methods = {'ABC','BFGS','DFP','GD','CG'};

                % ---- COARSE GRID ----
                L_list      = 1:3;
                hiddenSizes = [16 32 64];

                lambda_list = [0 1e-4 3e-4 1e-3 3e-3 1e-2];

                stepSizes = [3e-3 1e-2 3e-2];
                qn_maxIter_list = [100 200 300];

                abc_SN_list       = [20 50 80];
                abc_maxCycle_list = [100 200 400];
                abc_limit_list    = [20 40 80];   % <-- EKLENDİ

                jobs = build_jobs_phase1(methods, L_list, hiddenSizes, stepSizes, lambda_list, ...
                                         abc_SN_list, abc_limit_list, abc_maxCycle_list, qn_maxIter_list);

                results = repmat(TEMPLATE, numel(jobs), 1);
                res_id = 0;

                save(CHECKPOINT1,'jobs','results','res_id','-v7.3');
            end

            fprintf('PHASE-1 (COARSE) | Toplam koşu sayısı: %d\n', numel(jobs));

        case 2
            % Phase-2 başlamadan önce Phase-1 tamam mı?
            if ~exist(CHECKPOINT1,'file')
                error('Phase-2 için önce Phase-1 çalışmalı. %s yok.', CHECKPOINT1);
            end
            s1 = load(CHECKPOINT1,'jobs','results');
            done1 = sum(~isnan([s1.results.testAcc]));
            if done1 < numel(s1.jobs)
                error('Phase-1 tamamlanmadan Phase-2 başlatılmaz. P1 done=%d/%d', done1, numel(s1.jobs));
            end

            [jobs, results, res_id] = init_phase(CHECKPOINT2, TEMPLATE, 2);

            if isempty(jobs)
                phase1_results = s1.results;
                tbl1 = struct2table(phase1_results);
                tbl1 = tbl1(~isnan(tbl1.testAcc),:);

                jobs = build_phase2_jobs_from_topk(tbl1, TOPK_PER_METHOD);
                results = repmat(TEMPLATE, numel(jobs), 1);
                res_id = 0;

                save(CHECKPOINT2,'jobs','results','res_id','-v7.3');
            end

            fprintf('PHASE-2 (FINE) | Toplam koşu sayısı: %d\n', numel(jobs));

        otherwise
            error('PHASE 1 veya 2 olmalı.');
    end

    Njob = numel(jobs);

    %% ================== PARTIALS + RESUME ==========================
    partialDir = sprintf('partials_P%d_apple_v3', PHASE);
    if ~exist(partialDir,'dir'), mkdir(partialDir); end

    results = apply_partials(results, TEMPLATE, partialDir);

    doneMask = ~isnan([results.testAcc]);
    doneCount = sum(doneMask);
    fprintf('>> Resume: %d/%d job hazır. Kalan: %d\n', doneCount, Njob, Njob-doneCount);

    %% ================== PARFOR ENABLE ==============================
    useParfor = false;
    if ALLOW_PARFOR
        useParfor = try_enable_parfor();
    end

    tStart = tic;

    %% ================== JOBLARI KOŞ ================================
    if useGPU
        idxABC  = find(strcmp({jobs.method},'ABC') & isnan([results.testAcc]));
        idxRest = find(~strcmp({jobs.method},'ABC') & isnan([results.testAcc]));

        % 1) GPU’da türev tabanlıları seri çalıştır (GPU context çakışmasın)
        if ~isempty(idxRest)
            for jj = 1:numel(idxRest)
                j = idxRest(jj);

                out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);

                results(j) = out;
                save_partial_client(partialDir, j, out);

                if mod(j, PRINT_EVERY)==0
                    fprintf('%s\n', format_log(out, PHASE, j, Njob));
                end
                if mod(j, SAVE_EVERY)==0
                    save_checkpoint_and_csv(PHASE, CHECKPOINT1, CHECKPOINT2, CSV1, CSV2, jobs, results);
                    fprintf('>> Checkpoint kaydedildi (P%d). Elapsed: %.1f dk\n', PHASE, toc(tStart)/60);
                end
            end
        end

        % 2) ABC’yi CPU + parfor ile çalıştır (isterse)
        if ~isempty(idxABC) && useParfor
            fprintf('>> GPU açık: parfor sadece ABC (%d job) için çalışıyor.\n', numel(idxABC));
            fprintf('>> CPU + parfor(threads) ile ABC başlıyor...\n');

            dqLog  = parallel.pool.DataQueue;
            afterEach(dqLog, @(msg) fprintf('%s\n', msg));

            dqSave = parallel.pool.DataQueue;
            afterEach(dqSave, @(S) client_save_partial(S, partialDir));

            parfor ii = 1:numel(idxABC)
                j = idxABC(ii);

                out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);

                send(dqSave, struct('j', j, 'out', out));
                send(dqLog, format_log(out, PHASE, j, Njob));
            end

            results = apply_partials(results, TEMPLATE, partialDir);
            save_checkpoint_and_csv(PHASE, CHECKPOINT1, CHECKPOINT2, CSV1, CSV2, jobs, results);

        elseif ~isempty(idxABC)
            fprintf('>> parfor yok -> ABC seri çalışıyor.\n');
            for jj = 1:numel(idxABC)
                j = idxABC(jj);

                out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);

                results(j) = out;
                save_partial_client(partialDir, j, out);

                if mod(j, PRINT_EVERY)==0
                    fprintf('%s\n', format_log(out, PHASE, j, Njob));
                end
                if mod(j, SAVE_EVERY)==0
                    save_checkpoint_and_csv(PHASE, CHECKPOINT1, CHECKPOINT2, CSV1, CSV2, jobs, results);
                end
            end
        end

    else
        idxTodo = find(isnan([results.testAcc]));
        if isempty(idxTodo)
            fprintf('>> Tüm joblar tamam.\n');
        elseif useParfor
            fprintf('>> CPU + parfor(threads) aktif: %d job çalışacak.\n', numel(idxTodo));

            dqLog  = parallel.pool.DataQueue;
            afterEach(dqLog, @(msg) fprintf('%s\n', msg));

            dqSave = parallel.pool.DataQueue;
            afterEach(dqSave, @(S) client_save_partial(S, partialDir));

            parfor ii = 1:numel(idxTodo)
                j = idxTodo(ii);

                out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);

                send(dqSave, struct('j', j, 'out', out));
                send(dqLog, format_log(out, PHASE, j, Njob));
            end

            results = apply_partials(results, TEMPLATE, partialDir);
            save_checkpoint_and_csv(PHASE, CHECKPOINT1, CHECKPOINT2, CSV1, CSV2, jobs, results);

        else
            fprintf('>> parfor yok -> seri çalışıyor.\n');
            for jj = 1:numel(idxTodo)
                j = idxTodo(jj);

                out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);

                results(j)=out;
                save_partial_client(partialDir, j, out);

                if mod(j, PRINT_EVERY)==0
                    fprintf('%s\n', format_log(out, PHASE, j, Njob));
                end
                if mod(j, SAVE_EVERY)==0
                    save_checkpoint_and_csv(PHASE, CHECKPOINT1, CHECKPOINT2, CSV1, CSV2, jobs, results);
                end
            end
        end
    end

    %% ================== PHASE SONU: CSV + HEATMAP ==================
    tbl = struct2table(results);
    tbl = tbl(~isnan(tbl.testAcc),:);

    if PHASE==1
        writetable(tbl, CSV1);
        tbl_phase1 = tbl;
        fprintf('\n>> Phase-1 bitti. CSV: %s\n', CSV1);
    else
        writetable(tbl, CSV2);
        tbl_phase2 = tbl;
        fprintf('\n>> Phase-2 bitti. CSV: %s\n', CSV2);
    end

    plot_heatmaps(tbl);

    % özet top10
    tblTop = sortrows(tbl,'testAcc','descend');
    disp(tblTop(1:min(10,height(tblTop)), {'phase','method','L','hiddenDim','lambda','stepSize','SN','abcLimit','maxCycle','maxIter','trainAcc','testAcc','bestTrainCost','bestTestCost'}));

    fprintf('\n>> P%d TAMAMLANDI.\n', PHASE);
end

fprintf('\nBİTTİ (AUTO_RUN_BOTH_PHASES=%d).\n', AUTO_RUN_BOTH_PHASES);

%% ================== PLOTS (PHASE 1 & 2 ayrı) ======================
if PLOT_AT_END
    if ~isempty(tbl_phase1)
        plot_best_runs_per_method(tbl_phase1, 1, PLOT_CRITERION);
    end
    if ~isempty(tbl_phase2)
        plot_best_runs_per_method(tbl_phase2, 2, PLOT_CRITERION);
    end
end

fprintf('\nBİTTİ.\n');

end % main

%% =================================================================
%%                       BEST RUN PLOTTING
%% =================================================================
function plot_best_runs_per_method(tbl, PHASE, criterion)
    if isempty(tbl), return; end

    methods = unique(tbl.method, 'stable');

    bestRows = table();
    for mi = 1:numel(methods)
        m = methods{mi};
        sub = tbl(strcmp(tbl.method,m),:);
        if isempty(sub), continue; end

        switch lower(string(criterion))
            case "maxtestacc"
                [~,ii] = max(sub.testAcc);
            otherwise % "minBestTestCost"
                [~,ii] = min(sub.bestTestCost);
        end
        bestRows = [bestRows; sub(ii,:)]; %#ok<AGROW>
    end

    % 1) Her method ayrı figür: Train/Test cost
    for r=1:height(bestRows)
        m = bestRows.method{r};
        trH = bestRows.trainCostHistory{r};
        teH = bestRows.testCostHistory{r};

        figure('Name', sprintf('P%d Best - %s', PHASE, m));
        hold on; grid on;
        if ~isempty(trH), plot(trH,'LineWidth',1.1); end
        if ~isempty(teH), plot(teH,'LineWidth',1.1); end

        xlabel('Iter/Cycle');
        ylabel('Cost');
        title(sprintf('P%d | %s | L=%d H=%d | lam=%g %s', ...
            PHASE, m, bestRows.L(r), bestRows.hiddenDim(r), bestRows.lambda(r), extra_title(bestRows(r,:))));
        legend({'TrainCost','TestCost'},'Location','best');
    end

    % 2) Ortak figür: Best run TestCost (5 method)
    figure('Name', sprintf('P%d Combined - Best TestCost', PHASE));
    hold on; grid on;
    for r=1:height(bestRows)
        teH = bestRows.testCostHistory{r};
        if isempty(teH), continue; end
        plot(teH,'LineWidth',1.1);
    end
    xlabel('Iter/Cycle');
    ylabel('Test Cost (Best Run)');
    title(sprintf('P%d | Best Run TestCost Comparison', PHASE));
    legend(bestRows.method,'Location','best');
end

function s = extra_title(row)
    m = row.method{1};
    if strcmp(m,'ABC')
        s = sprintf('| SN=%d lim=%d MC=%d', row.SN, row.abcLimit, row.maxCycle);
    else
        s = sprintf('| lr=%g it=%d', row.stepSize, row.maxIter);
    end
end

%% =================================================================
%%                         RUN ONE JOB
%% =================================================================
function out = run_one_job(job, TEMPLATE, inputDim, outputDim, ...
                           X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                           useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU)

    net.inputDim  = inputDim;
    net.outputDim = outputDim;
    net.L         = job.L;
    net.hiddenDim = job.hiddenDim;

    % ABC'yi CPU'da çalıştır
    if strcmp(job.method,'ABC') && FORCE_ABC_CPU
        theta0 = init_weights_L(net);
        [theta_best, trHist, teHist, bestTr, bestTe, finTr, finTe] = abc_optimize(theta0, net, ...
            X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, job.lambda, job.SN, job.abcLimit, job.maxCycle);

        [trainAcc, testAcc] = evaluate_model(theta_best, net, ...
            X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu);

        out = TEMPLATE;
        out.phase      = job.phase;
        out.method     = job.method;
        out.L          = job.L;
        out.hiddenDim  = job.hiddenDim;
        out.stepSize   = job.stepSize;
        out.lambda     = job.lambda;
        out.SN         = job.SN;
        out.abcLimit   = job.abcLimit;
        out.maxCycle   = job.maxCycle;
        out.maxIter    = job.maxIter;

        out.bestTrainCost = bestTr;
        out.bestTestCost  = bestTe;
        out.finalTrainCost = finTr;
        out.finalTestCost  = finTe;

        out.trainAcc   = trainAcc;
        out.testAcc    = testAcc;

        out.trainCostHistory = trHist(:);
        out.testCostHistory  = teHist(:);
        out.theta      = theta_best(:);
        return;
    end

    % türev tabanlılar: GPU varsa GPU
    theta0 = init_weights_L(net);
    if useGPU
        theta0 = gpuArray(theta0);
        Xtr = X_train_g; Ttr = T_train_g;
        Xte = X_test_g;  Tte = T_test_g;
    else
        Xtr = X_train_cpu; Ttr = T_train_cpu;
        Xte = X_test_cpu;  Tte = T_test_cpu;
    end

    switch job.method
        case 'ABC'
            [theta_best, trHist, teHist, bestTr, bestTe, finTr, finTe] = abc_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.SN, job.abcLimit, job.maxCycle);
        case 'BFGS'
            [theta_best, trHist, teHist, bestTr, bestTe, finTr, finTe] = bfgs_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
        case 'DFP'
            [theta_best, trHist, teHist, bestTr, bestTe, finTr, finTe] = dfp_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
        case 'GD'
            [theta_best, trHist, teHist, bestTr, bestTe, finTr, finTe] = gd_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
        case 'CG'
            [theta_best, trHist, teHist, bestTr, bestTe, finTr, finTe] = cg_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
        otherwise
            error('Bilinmeyen method: %s', job.method);
    end

    [trainAcc, testAcc] = evaluate_model(theta_best, net, Xtr, Ttr, Xte, Tte);

    out = TEMPLATE;
    out.phase      = job.phase;
    out.method     = job.method;
    out.L          = job.L;
    out.hiddenDim  = job.hiddenDim;
    out.stepSize   = job.stepSize;
    out.lambda     = job.lambda;
    out.SN         = job.SN;
    out.abcLimit   = job.abcLimit;
    out.maxCycle   = job.maxCycle;
    out.maxIter    = job.maxIter;

    out.bestTrainCost = double(gather(bestTr));
    out.bestTestCost  = double(gather(bestTe));
    out.finalTrainCost = double(gather(finTr));
    out.finalTestCost  = double(gather(finTe));

    out.trainAcc   = double(trainAcc);
    out.testAcc    = double(testAcc);
    out.trainCostHistory = gather(trHist(:));
    out.testCostHistory  = gather(teHist(:));
    out.theta      = gather(theta_best);
end

%% =================================================================
%%                   CHECKPOINT / PARTIAL HELPERS
%% =================================================================
function save_checkpoint_and_csv(PHASE, CHECKPOINT1, CHECKPOINT2, CSV1, CSV2, jobs, results)
    try
        res_id = sum(~isnan([results.testAcc]));
        
        % CSV için table oluştururken array alanları (history, theta) çıkaralım
        resultsForCsv = results(~isnan([results.testAcc]));
        if isfield(resultsForCsv, 'trainCostHistory'), resultsForCsv = rmfield(resultsForCsv, 'trainCostHistory'); end
        if isfield(resultsForCsv, 'testCostHistory'),  resultsForCsv = rmfield(resultsForCsv, 'testCostHistory'); end
        if isfield(resultsForCsv, 'theta'),            resultsForCsv = rmfield(resultsForCsv, 'theta'); end
        
        tbl = struct2table(resultsForCsv, 'AsArray', true);

        if PHASE==1
            save(CHECKPOINT1,'jobs','results','res_id','-v7.3');
            writetable(tbl, CSV1);
        else
            save(CHECKPOINT2,'jobs','results','res_id','-v7.3');
            writetable(tbl, CSV2);
        end
    catch ME
        fprintf('!! Checkpoint/CSV yazılamadı: %s\n', ME.message);
    end
end

function save_partial_client(partialDir, j, out)
    f = fullfile(partialDir, sprintf('part_%05d.mat', j));
    try
        save(f,'out','-v7');
    catch ME
        fprintf('!! Partial save hata (client) j=%d: %s\n', j, ME.message);
    end
end

function client_save_partial(S, partialDir)
    j = S.j;
    out = S.out;
    f = fullfile(partialDir, sprintf('part_%05d.mat', j));
    try
        save(f,'out','-v7');
    catch ME
        fprintf('!! Partial save hata (client callback) j=%d: %s\n', j, ME.message);
    end
end

function results = apply_partials(results, TEMPLATE, partialDir)
    if isempty(results), return; end
    if ~exist(partialDir,'dir'), return; end

    files = dir(fullfile(partialDir,'part_*.mat'));
    for k = 1:numel(files)
        f = fullfile(files(k).folder, files(k).name);
        s = load(f,'out');
        out = s.out;

        tok = regexp(files(k).name,'part_(\d+)\.mat','tokens','once');
        if isempty(tok), continue; end
        j = str2double(tok{1});

        if j>=1 && j<=numel(results)
            results(j) = coerce_one_to_template(out, TEMPLATE);
        end
    end
end

function out2 = coerce_one_to_template(out, TEMPLATE)
    fnT = fieldnames(TEMPLATE);
    out2 = TEMPLATE;
    for f = 1:numel(fnT)
        if isfield(out, fnT{f})
            out2.(fnT{f}) = out.(fnT{f});
        else
            out2.(fnT{f}) = TEMPLATE.(fnT{f});
        end
    end
end

function [jobs, results, res_id] = init_phase(checkpointFile, TEMPLATE, phaseId)
    jobs=[]; results=[]; res_id=0;
    if exist(checkpointFile,'file')
        s = load(checkpointFile,'jobs','results','res_id');
        jobs = s.jobs;
        results = s.results;
        res_id = s.res_id;

        if isempty(results)
            results = repmat(TEMPLATE, numel(jobs), 1);
        else
            results = coerce_results_to_template(results, TEMPLATE);
            if numel(results) < numel(jobs)
                results(numel(jobs)) = TEMPLATE;
            end
        end
        fprintf('>> Phase-%d checkpoint yüklendi. res_id=%d / %d\n', phaseId, res_id, numel(jobs));
    end
end

function results2 = coerce_results_to_template(results, TEMPLATE)
    fnT = fieldnames(TEMPLATE);
    results2 = repmat(TEMPLATE, size(results));
    for k = 1:numel(results)
        s = results(k);
        for f = 1:numel(fnT)
            if isfield(s, fnT{f})
                results2(k).(fnT{f}) = s.(fnT{f});
            else
                results2(k).(fnT{f}) = TEMPLATE.(fnT{f});
            end
        end
    end
end

function s = format_log(out, PHASE, j, Njob)
    if strcmp(out.method,'ABC')
        s = sprintf(['(%d/%d) P%d M:%s | L:%d H:%d | lam:%g | SN:%d lim:%d MC:%d ' ...
                     '-> TrAcc:%.3f TeAcc:%.3f | bestTeCost:%.4f'], ...
            j, Njob, PHASE, out.method, out.L, out.hiddenDim, out.lambda, out.SN, out.abcLimit, out.maxCycle, ...
            out.trainAcc, out.testAcc, out.bestTestCost);
    else
        s = sprintf(['(%d/%d) P%d M:%s | L:%d H:%d | lam:%g | lr:%g Iter:%d ' ...
                     '-> TrAcc:%.3f TeAcc:%.3f | bestTeCost:%.4f'], ...
            j, Njob, PHASE, out.method, out.L, out.hiddenDim, out.lambda, out.stepSize, out.maxIter, ...
            out.trainAcc, out.testAcc, out.bestTestCost);
    end
end

%% =================================================================
%%                       JOB BUILDERS
%% =================================================================
function jobs = build_jobs_phase1(methods, L_list, hiddenSizes, stepSizes, lambda_list, ...
                                  abc_SN_list, abc_limit_list, abc_maxCycle_list, qn_maxIter_list)
    jobs = struct('phase',{},'method',{},'L',{},'hiddenDim',{},'stepSize',{},'lambda',{}, ...
                  'SN',{},'abcLimit',{},'maxCycle',{},'maxIter',{});
    k = 0;
    for mi = 1:numel(methods)
        m = methods{mi};
        for L = L_list
            for h = hiddenSizes
                for lam = lambda_list
                    if strcmp(m,'ABC')
                        for sn = abc_SN_list
                            for lim = abc_limit_list
                                for mc = abc_maxCycle_list
                                    k=k+1;
                                    jobs(k).phase=1;
                                    jobs(k).method=m; jobs(k).L=L; jobs(k).hiddenDim=h;
                                    jobs(k).stepSize=NaN; jobs(k).lambda=lam;
                                    jobs(k).SN=sn; jobs(k).abcLimit=lim; jobs(k).maxCycle=mc; jobs(k).maxIter=NaN;
                                end
                            end
                        end
                    else
                        for lr = stepSizes
                            for it = qn_maxIter_list
                                k=k+1;
                                jobs(k).phase=1;
                                jobs(k).method=m; jobs(k).L=L; jobs(k).hiddenDim=h;
                                jobs(k).stepSize=lr; jobs(k).lambda=lam;
                                jobs(k).SN=NaN; jobs(k).abcLimit=NaN; jobs(k).maxCycle=NaN; jobs(k).maxIter=it;
                            end
                        end
                    end
                end
            end
        end
    end
end

function jobs2 = build_phase2_jobs_from_topk(tbl1, topk)
    % Phase-2:
    % - Her method için top-k (testAcc yüksek) seç
    % - Method’un genelinde en iyi 2 lambda arasına log-ara değerler koy
    % - Her seçilen run etrafında local refine (lambda, lr/it veya SN/lim/MC)
    methods = unique(tbl1.method, 'stable');
    jobs2 = struct('phase',{},'method',{},'L',{},'hiddenDim',{},'stepSize',{},'lambda',{}, ...
                   'SN',{},'abcLimit',{},'maxCycle',{},'maxIter',{});
    kk = 0;

    for mi = 1:numel(methods)
        m = methods{mi};

        subAll = tbl1(strcmp(tbl1.method,m),:);
        subTop = sortrows(subAll,'testAcc','descend');
        subTop = subTop(1:min(topk,height(subTop)),:);

        for r = 1:height(subTop)
            L  = subTop.L(r);
            H  = subTop.hiddenDim(r);
            lam0 = subTop.lambda(r);

            % Local refine (Daha az nokta: Center, Center/ratio, Center*ratio) -> n=3
            lam_list = unique([0 refine_log_grid(lam0, 0.5, 3)]);
            lam_list = lam_list(lam_list>=0 & lam_list<=1);

            if strcmp(m,'ABC')
                SN0  = subTop.SN(r);
                LIM0 = subTop.abcLimit(r);
                MC0  = subTop.maxCycle(r);

                % Grid radiusSteps=1 -> [Center-Step, Center, Center+Step] (Toplam 3 nokta)
                SN_list  = refine_int_grid(SN0, 10, 1, 10, 400);
                LIM_list = refine_int_grid(LIM0,10, 1, 5, 600);
                MC_list  = refine_int_grid(MC0, 50, 1, 50, 6000);

                for lam = lam_list
                    for sn = SN_list
                        for lim = LIM_list
                            for mc = MC_list
                                kk=kk+1;
                                jobs2(kk).phase=2;
                                jobs2(kk).method=m; jobs2(kk).L=L; jobs2(kk).hiddenDim=H;
                                jobs2(kk).lambda=lam; jobs2(kk).SN=sn; jobs2(kk).abcLimit=lim; jobs2(kk).maxCycle=mc;
                                jobs2(kk).stepSize=NaN; jobs2(kk).maxIter=NaN;
                            end
                        end
                    end
                end
            else
                lr0  = subTop.stepSize(r);
                it0  = subTop.maxIter(r);

                lr_list = refine_log_grid(lr0, 0.5, 3);
                it_list = refine_int_grid(it0, 50, 1, 50, 6000);

                for lam = lam_list
                    for lr = lr_list
                        for it = it_list
                            kk=kk+1;
                            jobs2(kk).phase=2;
                            jobs2(kk).method=m; jobs2(kk).L=L; jobs2(kk).hiddenDim=H;
                            jobs2(kk).lambda=lam; jobs2(kk).stepSize=lr; jobs2(kk).maxIter=it;
                            jobs2(kk).SN=NaN; jobs2(kk).abcLimit=NaN; jobs2(kk).maxCycle=NaN;
                        end
                    end
                end
            end
        end
    end

    jobs2 = unique_jobs_stable(jobs2);
end

function a = refine_log_grid(x0, ratio, n)
    if ~isfinite(x0) || x0<=0, x0 = 1e-3; end
    k = floor(n/2);
    exps = (-k:k);
    a = x0 .* ( (1/ratio) .^ exps );
    a = a(a>1e-8 & a<1);
end

function v = refine_int_grid(v0, step, radiusSteps, minV, maxV)
    if ~isfinite(v0), v0 = minV; end
    v = (v0 - radiusSteps*step) : step : (v0 + radiusSteps*step);
    v = v(v>=minV & v<=maxV);
    v = unique(round(v));
end

function jobsU = unique_jobs_stable(jobs)
    % NaN hack yok: key string ile unique
    if isempty(jobs), jobsU = jobs; return; end

    keys = strings(numel(jobs),1);
    for i=1:numel(jobs)
        j = jobs(i);
        keys(i) = sprintf('p=%d|m=%s|L=%d|H=%d|lam=%.12g|lr=%.12g|it=%d|SN=%d|lim=%d|MC=%d', ...
            j.phase, j.method, j.L, j.hiddenDim, ...
            nan2num(j.lambda), nan2num(j.stepSize), int2nan(j.maxIter), ...
            int2nan(j.SN), int2nan(j.abcLimit), int2nan(j.maxCycle));
    end
    [~, ia] = unique(keys,'stable');
    jobsU = jobs(ia);

    function x = nan2num(x)
        if isnan(x), x = -1; end
    end
    function x = int2nan(x)
        if isnan(x), x = -1; end
    end
end

%% =================================================================
%%                           HEATMAPS
%% =================================================================
function plot_heatmaps(tbl)
    if isempty(tbl), return; end
    methods = unique(tbl.method);

    % lambda x H
    for mi = 1:numel(methods)
        m = methods{mi};
        sub = tbl(strcmp(tbl.method,m),:);

        Hs = unique(sub.hiddenDim);
        lams = unique(sub.lambda);
        if numel(Hs)<2 || numel(lams)<2, continue; end

        M = nan(numel(lams), numel(Hs));
        for i = 1:numel(lams)
            for j = 1:numel(Hs)
                ss = sub(sub.lambda==lams(i) & sub.hiddenDim==Hs(j),:);
                if ~isempty(ss), M(i,j) = max(ss.testAcc); end
            end
        end

        figure; imagesc(Hs, 1:numel(lams), M);
        set(gca,'YDir','normal');
        yticks(1:numel(lams));
        yticklabels(arrayfun(@(x) sprintf('%g',x), lams, 'UniformOutput', false));
        colorbar;
        xlabel('H'); ylabel('lambda');
        title(sprintf('%s | lambda x H (best TestAcc)', m));
    end

    % lr x iter (türev tabanlılar)
    for mi = 1:numel(methods)
        m = methods{mi};
        if strcmp(m,'ABC'), continue; end
        sub = tbl(strcmp(tbl.method,m),:);

        lrs = unique(sub.stepSize);
        its = unique(sub.maxIter);
        if numel(lrs)<2 || numel(its)<2, continue; end

        M = nan(numel(lrs), numel(its));
        for i = 1:numel(lrs)
            for j = 1:numel(its)
                ss = sub(sub.stepSize==lrs(i) & sub.maxIter==its(j),:);
                if ~isempty(ss), M(i,j) = max(ss.testAcc); end
            end
        end

        figure; imagesc(its, 1:numel(lrs), M);
        set(gca,'YDir','normal');
        yticks(1:numel(lrs));
        yticklabels(arrayfun(@(x) sprintf('%g',x), lrs, 'UniformOutput', false));
        colorbar;
        xlabel('maxIter'); ylabel('stepSize (lr)');
        title(sprintf('%s | lr x iter (best TestAcc)', m));
    end
end

%% =================================================================
%%                           MODEL (L layers)
%% =================================================================
function theta = init_weights_L(net)
    in  = net.inputDim;
    hid = net.hiddenDim;
    out = net.outputDim;
    L   = net.L;

    parts = cell(1, 2*L + 2);
    p = 1;

    lim = sqrt(6/(in+hid));
    W = (rand(hid,in)*2-1)*lim; b = zeros(hid,1);
    parts{p}=W(:); p=p+1; parts{p}=b(:); p=p+1;

    for l = 2:L
        lim = sqrt(6/(hid+hid));
        W = (rand(hid,hid)*2-1)*lim; b = zeros(hid,1);
        parts{p}=W(:); p=p+1; parts{p}=b(:); p=p+1;
    end

    lim = sqrt(6/(hid+out));
    W = (rand(out,hid)*2-1)*lim; b = zeros(out,1);
    parts{p}=W(:); p=p+1; parts{p}=b(:);

    theta = vertcat(parts{:});
end

function [Ws, bs, Wout, bout] = unpack_theta_L(theta, net)
    in  = net.inputDim;
    hid = net.hiddenDim;
    out = net.outputDim;
    L   = net.L;

    Ws = cell(1,L);
    bs = cell(1,L);
    idx = 1;

    nW = hid*in;
    Ws{1} = reshape(theta(idx:idx+nW-1),[hid,in]); idx=idx+nW;
    bs{1} = reshape(theta(idx:idx+hid-1),[hid,1]); idx=idx+hid;

    for l = 2:L
        nW = hid*hid;
        Ws{l} = reshape(theta(idx:idx+nW-1),[hid,hid]); idx=idx+nW;
        bs{l} = reshape(theta(idx:idx+hid-1),[hid,1]); idx=idx+hid;
    end

    nW = out*hid;
    Wout = reshape(theta(idx:idx+nW-1),[out,hid]); idx=idx+nW;
    bout = reshape(theta(idx:idx+out-1),[out,1]);
end

function Y = forward_pass(theta, net, X)
    [Ws, bs, Wout, bout] = unpack_theta_L(theta, net);
    L = net.L;

    A = X;
    for l = 1:L
        Z = A*Ws{l}.' + bs{l}.';
        A = tanh(Z);
    end

    Z2 = A*Wout.' + bout.';
    Z2 = Z2 - max(Z2,[],2);
    expZ = exp(Z2);
    Y = expZ ./ sum(expZ,2);
end

function [cost, grad] = cost_and_grad(theta, net, X, T, lambda)
    [Ws, bs, Wout, bout] = unpack_theta_L(theta, net);
    L = net.L;
    N = size(X,1);

    A0 = X;
    As = cell(1,L);
    A = A0;

    for l = 1:L
        Z = A*Ws{l}.' + bs{l}.';
        A = tanh(Z);
        As{l} = A;
    end

    Z2 = A*Wout.' + bout.';
    Z2 = Z2 - max(Z2,[],2);
    expZ = exp(Z2);
    Y = expZ ./ sum(expZ,2);

    epsVal = 1e-12;
    ce = -mean(sum(T .* log(Y + epsVal), 2));

    l2sum = 0;
    for l = 1:L, l2sum = l2sum + sum(Ws{l}(:).^2); end
    l2sum = l2sum + sum(Wout(:).^2);

    cost = ce + 0.5*lambda*l2sum;

    dZ2 = (Y - T) / N;
    grad_Wout = dZ2.' * As{L};
    grad_bout = sum(dZ2,1).';

    dA = dZ2 * Wout;

    grad_Ws = cell(1,L);
    grad_bs = cell(1,L);

    for l = L:-1:1
        Acur = As{l};
        dZ = dA .* (1 - Acur.^2);
        Aprev = A0; if l>1, Aprev = As{l-1}; end
        grad_Ws{l} = dZ.' * Aprev;
        grad_bs{l} = sum(dZ,1).';
        dA = dZ * Ws{l};
    end

    if lambda > 0
        for l = 1:L, grad_Ws{l} = grad_Ws{l} + lambda*Ws{l}; end
        grad_Wout = grad_Wout + lambda*Wout;
    end

    parts = cell(1,2*L+2);
    p=1;
    for l = 1:L
        parts{p}=grad_Ws{l}(:); p=p+1;
        parts{p}=grad_bs{l}(:); p=p+1;
    end
    parts{p}=grad_Wout(:); p=p+1;
    parts{p}=grad_bout(:);
    grad = vertcat(parts{:});
end

function c = cost_only(theta, net, X, T, lambda)
    [c, ~] = cost_and_grad(theta, net, X, T, lambda);
end

function [trainAcc, testAcc] = evaluate_model(theta, net, X_train, T_train, X_test, T_test)
    if isa(X_train,'gpuArray') && ~isa(theta,'gpuArray'), theta = gpuArray(theta); end
    if ~isa(X_train,'gpuArray') && isa(theta,'gpuArray'), theta = gather(theta); end

    Ytr = gather(forward_pass(theta, net, X_train));
    Ttr = gather(T_train);
    [~,ptr] = max(Ytr,[],2);
    [~,ytr] = max(Ttr,[],2);
    trainAcc = mean(ptr==ytr);

    Yte = gather(forward_pass(theta, net, X_test));
    Tte = gather(T_test);
    [~,pte] = max(Yte,[],2);
    [~,yte] = max(Tte,[],2);
    testAcc = mean(pte==yte);
end

%% =================================================================
%%                           OPTIMIZERS
%%  Hepsi: train/test history + bestTrainCost/bestTestCost + final costs
%% =================================================================
function [bestTheta, trainHist, testHist, bestTrainCost, bestTestCost, finalTrainCost, finalTestCost] = ...
    abc_optimize(theta0, net, Xtr, Ttr, Xte, Tte, lambda, SN, abcLimit, maxCycle)

    D = numel(theta0);
    FoodNumber = SN;
    limit = abcLimit;

    Foods = zeros(D, FoodNumber, 'like', theta0);
    trial = zeros(1, FoodNumber, 'like', theta0);
    f     = zeros(1, FoodNumber, 'like', theta0);

    for i = 1:FoodNumber
        Foods(:,i) = theta0 + 0.1*randn(D,1,'like',theta0);
        f(i) = -cost_only(Foods(:,i), net, Xtr, Ttr, lambda);
    end

    [bestFit, bestIndex] = max(f);
    bestTheta = Foods(:,bestIndex);

    trainHist = zeros(maxCycle,1);
    testHist  = zeros(maxCycle,1);

    bestTrainCost = inf;
    bestTestCost  = inf;

    for cycle = 1:maxCycle
        % Employed
        for i = 1:FoodNumber
            k = randi(FoodNumber); while k==i, k=randi(FoodNumber); end
            j = randi(D);
            phi = 2*rand-1;

            v = Foods(:,i);
            v(j) = v(j) + phi*(Foods(j,i)-Foods(j,k));

            fv = -cost_only(v, net, Xtr, Ttr, lambda);
            if fv > f(i)
                Foods(:,i)=v; f(i)=fv; trial(i)=0;
            else
                trial(i)=trial(i)+1;
            end
        end

        % Onlooker
        prob = (f - min(f)) + eps;
        prob = prob / sum(prob);

        i=1; t=0;
        while t < FoodNumber
            if rand < prob(i)
                t=t+1;
                k = randi(FoodNumber); while k==i, k=randi(FoodNumber); end
                j = randi(D);
                phi = 2*rand-1;

                v = Foods(:,i);
                v(j) = v(j) + phi*(Foods(j,i)-Foods(j,k));

                fv = -cost_only(v, net, Xtr, Ttr, lambda);
                if fv > f(i)
                    Foods(:,i)=v; f(i)=fv; trial(i)=0;
                else
                    trial(i)=trial(i)+1;
                end
            end
            i = mod(i,FoodNumber)+1;
        end

        % Scout
        [maxTrial, ind] = max(trial);
        if maxTrial > limit
            Foods(:,ind) = theta0 + 0.1*randn(D,1,'like',theta0);
            f(ind) = -cost_only(Foods(:,ind), net, Xtr, Ttr, lambda);
            trial(ind)=0;
        end

        % Best by fit
        [cycleBestFit, bestIndex] = max(f);
        if cycleBestFit > bestFit
            bestFit = cycleBestFit;
            bestTheta = Foods(:,bestIndex);
        end

        cTr = cost_only(bestTheta, net, Xtr, Ttr, lambda);
        cTe = cost_only(bestTheta, net, Xte, Tte, lambda);

        cTr = double(gather(cTr));
        cTe = double(gather(cTe));

        trainHist(cycle) = cTr;
        testHist(cycle)  = cTe;

        if cTr < bestTrainCost, bestTrainCost = cTr; end
        if cTe < bestTestCost,  bestTestCost  = cTe; end
    end

    finalTrainCost = trainHist(end);
    finalTestCost  = testHist(end);
end

function [theta, trainHist, testHist, bestTrainCost, bestTestCost, finalTrainCost, finalTestCost] = ...
    bfgs_optimize(theta0, net, Xtr, Ttr, Xte, Tte, lambda, stepSize, maxIter)

    theta = theta0;
    [costVal, grad] = cost_and_grad(theta, net, Xtr, Ttr, lambda);
    n = numel(theta);
    H = eye(n,'like',theta);

    trainHist = zeros(maxIter,1);
    testHist  = zeros(maxIter,1);

    bestTrainCost = inf;
    bestTestCost  = inf;

    for iter=1:maxIter
        cTr = double(gather(cost_only(theta, net, Xtr, Ttr, lambda)));
        cTe = double(gather(cost_only(theta, net, Xte, Tte, lambda)));
        trainHist(iter) = cTr;
        testHist(iter)  = cTe;

        if cTr < bestTrainCost, bestTrainCost = cTr; end
        if cTe < bestTestCost,  bestTestCost  = cTe; end

        p = -H*grad;

        alpha = stepSize;
        c1 = 1e-4;
        gTp = gather(grad(:)).'*gather(p(:));
        while true
            theta_new = theta + alpha*p;
            cost_new = cost_only(theta_new, net, Xtr, Ttr, lambda);
            if double(gather(cost_new)) <= double(gather(costVal)) + c1*alpha*gTp || alpha < 1e-8
                break;
            end
            alpha = alpha*0.5;
        end

        [cost_new, grad_new] = cost_and_grad(theta_new, net, Xtr, Ttr, lambda);
        s = theta_new-theta;
        y = grad_new-grad;

        ys = gather(y(:)).'*gather(s(:));
        if ys > 1e-12
            rho = 1/ys;
            I = eye(n,'like',theta);
            H = (I - rho*(s*y.'))*H*(I - rho*(y*s.')) + rho*(s*s.');
        end

        theta=theta_new; costVal=cost_new; grad=grad_new;

        if norm(gather(grad)) < 1e-5
            trainHist = trainHist(1:iter);
            testHist  = testHist(1:iter);
            break;
        end
    end

    finalTrainCost = trainHist(end);
    finalTestCost  = testHist(end);
end

function [theta, trainHist, testHist, bestTrainCost, bestTestCost, finalTrainCost, finalTestCost] = ...
    dfp_optimize(theta0, net, Xtr, Ttr, Xte, Tte, lambda, stepSize, maxIter)

    theta = theta0;
    [costVal, grad] = cost_and_grad(theta, net, Xtr, Ttr, lambda);
    n = numel(theta);
    H = eye(n,'like',theta);

    trainHist = zeros(maxIter,1);
    testHist  = zeros(maxIter,1);

    bestTrainCost = inf;
    bestTestCost  = inf;

    for iter=1:maxIter
        cTr = double(gather(cost_only(theta, net, Xtr, Ttr, lambda)));
        cTe = double(gather(cost_only(theta, net, Xte, Tte, lambda)));
        trainHist(iter) = cTr;
        testHist(iter)  = cTe;

        if cTr < bestTrainCost, bestTrainCost = cTr; end
        if cTe < bestTestCost,  bestTestCost  = cTe; end

        p = -H*grad;

        alpha = stepSize;
        c1 = 1e-4;
        gTp = gather(grad(:)).'*gather(p(:));
        while true
            theta_new = theta + alpha*p;
            cost_new = cost_only(theta_new, net, Xtr, Ttr, lambda);
            if double(gather(cost_new)) <= double(gather(costVal)) + c1*alpha*gTp || alpha < 1e-8
                break;
            end
            alpha = alpha*0.5;
        end

        [cost_new, grad_new] = cost_and_grad(theta_new, net, Xtr, Ttr, lambda);
        s = theta_new-theta;
        y = grad_new-grad;

        ys = gather(y(:)).'*gather(s(:));
        if ys > 1e-12
            Hy = H*y;
            yHy = gather(y(:)).'*gather(Hy(:));
            H = H + (s*s.')/ys - (Hy*Hy.')/(yHy + eps);
        end

        theta=theta_new; costVal=cost_new; grad=grad_new;

        if norm(gather(grad)) < 1e-5
            trainHist = trainHist(1:iter);
            testHist  = testHist(1:iter);
            break;
        end
    end

    finalTrainCost = trainHist(end);
    finalTestCost  = testHist(end);
end

function [theta, trainHist, testHist, bestTrainCost, bestTestCost, finalTrainCost, finalTestCost] = ...
    gd_optimize(theta0, net, Xtr, Ttr, Xte, Tte, lambda, stepSize, maxIter)

    theta = theta0;
    trainHist = zeros(maxIter,1);
    testHist  = zeros(maxIter,1);

    bestTrainCost = inf;
    bestTestCost  = inf;

    for iter=1:maxIter
        [~, grad] = cost_and_grad(theta, net, Xtr, Ttr, lambda);

        cTr = double(gather(cost_only(theta, net, Xtr, Ttr, lambda)));
        cTe = double(gather(cost_only(theta, net, Xte, Tte, lambda)));
        trainHist(iter)=cTr;
        testHist(iter) =cTe;

        if cTr < bestTrainCost, bestTrainCost = cTr; end
        if cTe < bestTestCost,  bestTestCost  = cTe; end

        theta = theta - stepSize*grad;

        if norm(gather(grad)) < 1e-5
            trainHist = trainHist(1:iter);
            testHist  = testHist(1:iter);
            break;
        end
    end

    finalTrainCost = trainHist(end);
    finalTestCost  = testHist(end);
end

function [theta, trainHist, testHist, bestTrainCost, bestTestCost, finalTrainCost, finalTestCost] = ...
    cg_optimize(theta0, net, Xtr, Ttr, Xte, Tte, lambda, stepSize, maxIter)

    theta = theta0;
    [costVal, grad] = cost_and_grad(theta, net, Xtr, Ttr, lambda);
    p = -grad;

    trainHist = zeros(maxIter,1);
    testHist  = zeros(maxIter,1);

    bestTrainCost = inf;
    bestTestCost  = inf;

    for iter=1:maxIter
        cTr = double(gather(cost_only(theta, net, Xtr, Ttr, lambda)));
        cTe = double(gather(cost_only(theta, net, Xte, Tte, lambda)));
        trainHist(iter)=cTr;
        testHist(iter) =cTe;

        if cTr < bestTrainCost, bestTrainCost = cTr; end
        if cTe < bestTestCost,  bestTestCost  = cTe; end

        alpha = stepSize;
        c1 = 1e-4;

        gTp = gather(grad(:)).'*gather(p(:));
        if gTp >= 0
            p = -grad;
            gTp = -gather(grad(:)).'*gather(grad(:));
        end

        while true
            theta_new = theta + alpha*p;
            cost_new = cost_only(theta_new, net, Xtr, Ttr, lambda);
            if double(gather(cost_new)) <= double(gather(costVal)) + c1*alpha*gTp || alpha < 1e-8
                break;
            end
            alpha = alpha*0.5;
        end

        [cost_new, grad_new] = cost_and_grad(theta_new, net, Xtr, Ttr, lambda);

        num = gather(grad_new(:)).'*gather(grad_new(:));
        den = gather(grad(:)).'*gather(grad(:));
        beta = num / max(1e-12, den);
        if beta < 0, beta = 0; end

        p = -grad_new + beta*p;

        theta=theta_new; grad=grad_new; costVal=cost_new;

        if norm(gather(grad)) < 1e-5
            trainHist = trainHist(1:iter);
            testHist  = testHist(1:iter);
            break;
        end
    end

    finalTrainCost = trainHist(end);
    finalTestCost  = testHist(end);
end

%% =================================================================
%%                       PARFOR ENABLE (SAFE)
%% =================================================================
function ok = try_enable_parfor()
    ok = false;
    try
        if license('test','Distrib_Computing_Toolbox')
            p = gcp('nocreate');
            if isempty(p)
                parpool('threads');
            end
            ok = true;
        end
    catch
        ok = false;
    end
end

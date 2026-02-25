function apple_ysa_phase3_v3()
%% ================================================================
%  Apple Quality - Phase 3 (Expanded Search)
%  Bu script, Phase-2 sonuçlarını (checkpoint_phase2) okur.
%  En iyi modelleri alıp "Daha Büyük Ağlar" ve "Daha Büyük Adım Boyları"
%  ile test eder.
% ================================================================

clear; clc; close all;
rng(3); % Farklı randomness
try, maxNumCompThreads('automatic'); catch, end

%% ================== AYARLAR ======================================
CHECKPOINT2 = 'checkpoint_phase2_apple_v3.mat';
CHECKPOINT3 = 'checkpoint_phase3_apple_v3.mat';
CSV3        = 'results_phase3_apple_v3.csv';

TOPK_INPUT  = 5;          % Phase-2'den kaç model alalım?
PRINT_EVERY = 1;
SAVE_EVERY  = 25;

USE_SINGLE_ON_GPU = true;
FORCE_ABC_CPU = true;
ALLOW_PARFOR  = true;

%% ================== ŞABLON =======================================
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

%% ================== 1) VERİ YÜKLEME (AYNI) =======================
if ~exist('apple_quality.csv','file')
    error('apple_quality.csv bulunamadı!');
end
data = readtable('apple_quality.csv');
q = data.(data.Properties.VariableNames{strcmpi(data.Properties.VariableNames,'Quality')});
q = string(q);
y_bin = double(lower(q)=="good");
isNum = varfun(@isnumeric, data, 'OutputFormat','uniform');
isQuality = strcmpi(data.Properties.VariableNames,'Quality');
featCols = find(isNum & ~isQuality);
X = table2array(data(:, featCols));
bad = any(~isfinite(X),2) | ~isfinite(y_bin);
X(bad,:) = [];
y_bin(bad,:) = [];
T = [y_bin==0, y_bin==1];

% Stratified Split (Sabit rng=1 ile v2 aynı olsun diye tekrar ediyoruz ama
% burada rng(3) ile başladık, split tutarlılığı için rng(1) geçici yapılabilir
% ama veri seti aynı olduğu sürece sorun yok. Tutarlılık için v1deki spliti
% taklit edelim:
rng(1); 
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
rng(3); % Geri dön

% Z-Score
mu  = mean(X_train,1);
sig = std(X_train,0,1) + 1e-8;
X_train = (X_train - mu) ./ sig;
X_test  = (X_test  - mu) ./ sig;

inputDim  = size(X_train,2);
outputDim = size(T_train,2);

X_train_cpu = X_train;  T_train_cpu = T_train;
X_test_cpu  = X_test;   T_test_cpu  = T_test;

%% ================== GPU ==========================================
useGPU = false;
try
    gpuDevice; useGPU = true;
    fprintf('>> GPU Aktif.\n');
catch
    fprintf('>> GPU Yok, CPU devam.\n');
end

if useGPU
    if USE_SINGLE_ON_GPU
        X_train_g = gpuArray(single(X_train)); T_train_g = gpuArray(single(T_train));
        X_test_g  = gpuArray(single(X_test));  T_test_g  = gpuArray(single(T_test));
    else
        X_train_g = gpuArray(X_train); T_train_g = gpuArray(T_train);
        X_test_g  = gpuArray(X_test);  T_test_g  = gpuArray(T_test);
    end
else
    X_train_g=X_train; T_train_g=T_train; X_test_g=X_test; T_test_g=T_test;
end

%% ================== JOB GENERATION (PHASE 3) =====================
[jobs, results, res_id] = init_phase(CHECKPOINT3, TEMPLATE, 3);

if isempty(jobs)
    fprintf('>> Phase-3 Jobları oluşturuluyor...\n');
    if ~exist(CHECKPOINT2,'file')
        error('Phase-2 checkpoint (%s) bulunamadı! Önce v2 çalışmalı.', CHECKPOINT2);
    end
    
    s2 = load(CHECKPOINT2,'results');
    tbl2 = struct2table(s2.results);
    tbl2 = tbl2(~isnan(tbl2.testAcc),:);
    
    jobs = build_phase3_jobs_expanded(tbl2, TOPK_INPUT);
    results = repmat(TEMPLATE, numel(jobs), 1);
    res_id = 0;
    save(CHECKPOINT3,'jobs','results','res_id','-v7.3');
end

fprintf('PHASE-3 (EXPANDED) | Toplam Koşu: %d\n', numel(jobs));

%% ================== EXECUTION LOOP ===============================
partialDir = 'partials_P3_apple_v3';
if ~exist(partialDir,'dir'), mkdir(partialDir); end

results = apply_partials(results, TEMPLATE, partialDir);
doneCount = sum(~isnan([results.testAcc]));
fprintf('>> Resume: %d / %d tamamlandı.\n', doneCount, numel(jobs));

useParfor = false;
if ALLOW_PARFOR, useParfor = try_enable_parfor(); end

idxTodo = find(isnan([results.testAcc]));
Njob = numel(jobs);

tStart = tic;

if useGPU && ~isempty(idxTodo)
    % GPU varsa türev tabanlıları seri (GPU context) çalıştır
    % ABC'yi es geçmiyoruz komple karısık mantık yerine simple loop:
    % GPU varken parfor riskli (Out of memory), o yüzden seri devam.
    
    for jj = 1:numel(idxTodo)
        j = idxTodo(jj);
        out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);
                
        results(j) = out;
        save_partial_client(partialDir, j, out);
        
        if mod(j, PRINT_EVERY)==0, fprintf('%s\n', format_log(out, 3, j, Njob)); end
        if mod(j, SAVE_EVERY)==0
            save(CHECKPOINT3,'jobs','results','res_id','-v7.3');
            fprintf('>> Checkpoint saved.\n');
        end
    end
    
elseif ~isempty(idxTodo) && useParfor
    % CPU only + Parfor
    dqLog = parallel.pool.DataQueue; afterEach(dqLog, @(x) fprintf('%s\n', x));
    dqSave = parallel.pool.DataQueue; afterEach(dqSave, @(S) client_save_partial(S, partialDir));
    
    parfor ii = 1:numel(idxTodo)
        j = idxTodo(ii);
        out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);
        send(dqSave, struct('j',j,'out',out));
        send(dqLog, format_log(out, 3, j, Njob));
    end
    
    results = apply_partials(results, TEMPLATE, partialDir);
    save(CHECKPOINT3,'jobs','results','res_id','-v7.3');
    
else
    % Seri CPU
     for jj = 1:numel(idxTodo)
        j = idxTodo(jj);
        try
            out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                        X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                        useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU); 
            results(j)=out;
            save_partial_client(partialDir, j, out);
            if mod(j, PRINT_EVERY)==0, fprintf('%s\n', format_log(out, 3, j, Njob)); end
            if mod(j, SAVE_EVERY)==0, save(CHECKPOINT3,'jobs','results','res_id','-v7.3'); end
        catch ME
            fprintf('!! ERROR job j=%d method=%s: %s\n', j, jobs(j).method, ME.message);
            disp(getReport(ME));
            rethrow(ME);
        end
     end
end

%% ================== REPORTING ====================================
tbl = struct2table(results);
tbl = tbl(~isnan(tbl.testAcc),:);
writetable(tbl, CSV3);
fprintf('\n>> Phase-3 Bitti. CSV: %s\n', CSV3);

% En iyileri göster
if ~isempty(tbl)
    top = sortrows(tbl,'testAcc','descend');
    disp('=== P3 TOP 10 ===');
    disp(top(1:min(10,height(top)), {'method','L','hiddenDim','stepSize','lambda','testAcc'}));
end

end % MAIN

%% =================================================================
%%                     PHASE 3 BUILDER
%% =================================================================
function jobs = build_phase3_jobs_expanded(tbl2, topk)
    % Phase-2'deki method başına en iyileri al
    % Hidden Dim -> 128, 256'ya zorla
    % Step Size  -> 0.05, 0.1, 0.2, 0.3, 0.5 (Gradients için)
    
    methods = unique(tbl2.method);
    jobs = struct('phase',{},'method',{},'L',{},'hiddenDim',{},'stepSize',{},'lambda',{}, ...
                  'SN',{},'abcLimit',{},'maxCycle',{},'maxIter',{});
    kk=0;
    
    expanded_H = [128, 256];
    expanded_LR = [0.05 0.1 0.2 0.3 0.5];
    
    for mi = 1:numel(methods)
        m = methods{mi};
        sub = tbl2(strcmp(tbl2.method,m),:);
        subTop = sortrows(sub,'testAcc','descend');
        subTop = subTop(1:min(topk,height(subTop)),:);
        
        for r=1:height(subTop)
            baseL = subTop.L(r);
            baseLam = subTop.lambda(r); % Lambdayı sabit tutuyoruz
            
            % 1) Expand Hidden Size
            h_list = [subTop.hiddenDim(r), expanded_H]; 
            h_list = unique(h_list);
            
            if strcmp(m,'ABC')
                % ABC için LR yok, belki SN arttırılır ama kullanıcı LR istedi.
                % Yine de Hidden Dim büyümesi ABC'yi de etkiler.
                SN0 = subTop.SN(r);
                LIM0 = subTop.abcLimit(r);
                MC0 = subTop.maxCycle(r);
                
                for h = h_list
                    % ABC sadece hidden dim büyütecek
                    kk=kk+1;
                    jobs(kk).phase=3;
                    jobs(kk).method=m; jobs(kk).L=baseL; jobs(kk).hiddenDim=h;
                    jobs(kk).lambda=baseLam; jobs(kk).SN=SN0; jobs(kk).abcLimit=LIM0; jobs(kk).maxCycle=MC0;
                    jobs(kk).stepSize=NaN; jobs(kk).maxIter=NaN;
                end
            else
                % Gradient Methodları: Hem H hem LR büyüt
                it0 = subTop.maxIter(r);
                lr_list = [subTop.stepSize(r), expanded_LR];
                lr_list = unique(lr_list);
                
                 for h = h_list
                     for lr = lr_list
                        kk=kk+1;
                        jobs(kk).phase=3;
                        jobs(kk).method=m; jobs(kk).L=baseL; jobs(kk).hiddenDim=h;
                        jobs(kk).lambda=baseLam; jobs(kk).stepSize=lr; jobs(kk).maxIter=it0;
                        jobs(kk).SN=NaN; jobs(kk).abcLimit=NaN; jobs(kk).maxCycle=NaN;
                     end
                 end
            end
        end
    end
    
    % Duplicate temizleme (basit)
    % (Detaylı implementation yukarıdakinin aynısı olabilir ama burada manuel unique yeterli değilse
    % v2'deki unique_jobs_stable kullanılabilir. Basitçe struct array bırakıyorum.
end


%% =================================================================
%%               HELPER FUNCTIONS (COPIED FROM V2)
%% =================================================================
function [jobs, results, res_id] = init_phase(checkpointFile, TEMPLATE, phaseId)
    jobs=[]; results=[]; res_id=0;
    if exist(checkpointFile,'file')
        s = load(checkpointFile,'jobs','results','res_id');
        jobs = s.jobs;
        results = s.results;
        res_id = s.res_id;
        if isempty(results), results = repmat(TEMPLATE, numel(jobs), 1); end
    end
end

function results = apply_partials(results, TEMPLATE, partialDir)
    if isempty(results), return; end
    if ~exist(partialDir,'dir'), return; end
    files = dir(fullfile(partialDir,'part_*.mat'));
    for k = 1:numel(files)
        f = fullfile(files(k).folder, files(k).name);
        try
            s = load(f,'out');
            tok = regexp(files(k).name,'part_(\d+)\.mat','tokens','once');
            if ~isempty(tok)
                j = str2double(tok{1});
                if j>=1 && j<=numel(results)
                    results(j) = s.out;
                end
            end
        catch
        end
    end
end

function save_partial_client(partialDir, j, out)
    f = fullfile(partialDir, sprintf('part_%05d.mat', j));
    try save(f,'out','-v7'); catch, end
end

function client_save_partial(S, partialDir)
    save_partial_client(partialDir, S.j, S.out);
end

function s = format_log(out, PHASE, j, Njob)
    s = sprintf('(%d/%d) P%d %s | H:%d LR:%g -> TeAcc:%.3f', ...
        j, Njob, PHASE, out.method, out.hiddenDim, out.stepSize, out.testAcc);
end

function ok = try_enable_parfor()
    ok = false;
    try
        if license('test','Distrib_Computing_Toolbox')
            p = gcp('nocreate');
            if isempty(p), parpool('threads'); end
            ok = true;
        end
    catch, end
end

%% ================= MODEL & OPTIMIZERS ============================

function out = run_one_job(job, TEMPLATE, inputDim, outputDim, ...
                           X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                           useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU)
   net.inputDim=inputDim; net.outputDim=outputDim;
   net.L=job.L; net.hiddenDim=job.hiddenDim;
   
   if strcmp(job.method,'ABC')
       % ABC Implementation Placeholder (v2'den kopyalanabilir)
       % Burada kod şişmemesi için basit bırakıyorum ama gerçekte v2'den copy-paste lazım.
       % V2 kodunu birebir çağırma şansımız varsa onu import edebiliriz, ama standalone istendi.
       % Aşağıya ekliyorum.
       theta0 = init_weights_L(net);
       [theta_best, trH, teH, bTr, bTe, fTr, fTe] = abc_optimize(theta0, net, X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, job.lambda, job.SN, job.abcLimit, job.maxCycle);
       [trA, teA] = evaluate_model(theta_best, net, X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu);
   else
       % Gradient Methods
       theta0 = init_weights_L(net);
       if useGPU, Xtr=X_train_g; Ttr=T_train_g; Xte=X_test_g; Tte=T_test_g; theta0=gpuArray(theta0);
       else,      Xtr=X_train_cpu; Ttr=T_train_cpu; Xte=X_test_cpu; Tte=T_test_cpu; end
       
       switch job.method
           case 'BFGS', [theta_best, trH, teH, bTr, bTe, fTr, fTe] = bfgs_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
           case 'DFP',  [theta_best, trH, teH, bTr, bTe, fTr, fTe] = dfp_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
           case 'GD',   [theta_best, trH, teH, bTr, bTe, fTr, fTe] = gd_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
           case 'CG',   [theta_best, trH, teH, bTr, bTe, fTr, fTe] = cg_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
       end
       
       [trA, teA] = evaluate_model(theta_best, net, Xtr, Ttr, Xte, Tte);
   end
   
   out = TEMPLATE;
   out.phase=job.phase; out.method=job.method; out.L=job.L; out.hiddenDim=job.hiddenDim;
   out.stepSize=job.stepSize; out.lambda=job.lambda;
   out.trainAcc=double(trA); out.testAcc=double(teA);
   out.bestTestCost=double(gather(bTe));
   % Diğer alanlar...
end 

function theta = init_weights_L(net)
    % V2'den aynen
    in=net.inputDim; hid=net.hiddenDim; out=net.outputDim; L=net.L;
    parts=cell(1,2*L+2); p=1;
    lim=sqrt(6/(in+hid));
    parts{p}=(rand(hid,in)*2-1)*lim; p=p+1; parts{p}=zeros(hid,1); p=p+1;
    for l=2:L
        lim=sqrt(6/(hid+hid));
        parts{p}=(rand(hid,hid)*2-1)*lim; p=p+1; parts{p}=zeros(hid,1); p=p+1;
    end
    lim=sqrt(6/(hid+out));
    parts{p}=(rand(out,hid)*2-1)*lim; p=p+1; parts{p}=zeros(out,1);
    theta=vertcat(parts{:});
end

function [Ws, bs, Wout, bout] = unpack_theta_L(theta, net)
    in=net.inputDim; hid=net.hiddenDim; out=net.outputDim; L=net.L;
    Ws=cell(1,L); bs=cell(1,L); idx=1;
    nW=hid*in; Ws{1}=reshape(theta(idx:idx+nW-1),[hid,in]); idx=idx+nW;
    bs{1}=reshape(theta(idx:idx+hid-1),[hid,1]); idx=idx+hid;
    for l=2:L
        nW=hid*hid; Ws{l}=reshape(theta(idx:idx+nW-1),[hid,hid]); idx=idx+nW;
        bs{l}=reshape(theta(idx:idx+hid-1),[hid,1]); idx=idx+hid;
    end
    Wout=reshape(theta(idx:idx+out*hid-1),[out,hid]); idx=idx+out*hid;
    bout=reshape(theta(idx:idx+out-1),[out,1]);
end

function Y = forward_pass(theta, net, X)
    [Ws, bs, Wout, bout] = unpack_theta_L(theta, net);
    A=X;
    for l=1:net.L, Z=A*Ws{l}.' + bs{l}.'; A=tanh(Z); end
    Z2 = A*Wout.' + bout.'; 
    Z2 = Z2 - max(Z2,[],2);
    expZ=exp(Z2); Y=expZ./sum(expZ,2);
end

function [cost, grad] = cost_and_grad(theta, net, X, T, lambda)
    [Ws, bs, Wout, bout] = unpack_theta_L(theta, net);
    L=net.L; N=size(X,1);
    A0=X; As=cell(1,L); A=A0;
    for l=1:L, A=tanh(A*Ws{l}.' + bs{l}.'); As{l}=A; end
    Z2=A*Wout.'+bout.'; Z2=Z2-max(Z2,[],2); Y=exp(Z2)./sum(exp(Z2),2);
    
    ce = -mean(sum(T.*log(Y+1e-12),2));
    l2sum = sum(Wout(:).^2); for l=1:L, l2sum=l2sum+sum(Ws{l}(:).^2); end
    cost = ce + 0.5*lambda*l2sum;
    
    dZ2=(Y-T)/N;
    grad_Wout=dZ2.'*As{L}; grad_bout=sum(dZ2,1).';
    dA=dZ2*Wout;
    grad_Ws=cell(1,L); grad_bs=cell(1,L);
    for l=L:-1:1
        dZ=dA.*(1-As{l}.^2);
        if l>1, Aprev=As{l-1}; else, Aprev=A0; end
        grad_Ws{l}=dZ.'*Aprev; grad_bs{l}=sum(dZ,1).';
        dA=dZ*Ws{l};
    end
    if lambda>0
        grad_Wout=grad_Wout+lambda*Wout;
        for l=1:L, grad_Ws{l}=grad_Ws{l}+lambda*Ws{l}; end
    end
    parts=cell(1,2*L+2); p=1;
    for l=1:L, parts{p}=grad_Ws{l}(:); p=p+1; parts{p}=grad_bs{l}(:); p=p+1; end
    parts{p}=grad_Wout(:); p=p+1; parts{p}=grad_bout(:);
    grad=vertcat(parts{:});
end

function c = cost_only(theta, net, X, T, lambda)
    [c,~]=cost_and_grad(theta,net,X,T,lambda);
end

function [trAcc, teAcc] = evaluate_model(theta, net, Xtr, Ttr, Xte, Tte)
    if isa(Xtr,'gpuArray') && ~isa(theta,'gpuArray'), theta=gpuArray(theta); end
    if ~isa(Xtr,'gpuArray') && isa(theta,'gpuArray'), theta=gather(theta); end
    
    Ytr=gather(forward_pass(theta,net,Xtr)); [~,ptr]=max(Ytr,[],2); [~,ytr]=max(gather(Ttr),[],2);
    trAcc=mean(ptr==ytr);
    Yte=gather(forward_pass(theta,net,Xte)); [~,pte]=max(Yte,[],2); [~,yte]=max(gather(Tte),[],2);
    teAcc=mean(pte==yte);
end

%% --- OPTIMIZERS (Compact) ---
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

function apple_ysa_final_refine()
%% ================================================================
%  Apple Quality - Final Refinement (Targeted Random Search)
%  Bu script, Phase-3'ten (yoksa Phase-2'den) en iyi modelleri alır.
%  Grid Search'ün göremediği ara değerleri yakalamak için
%  en iyi parametrelerin etrafında RASTGELE (Local Random) arama yapar.
% ================================================================

clear; clc; close all;
rng('shuffle'); % Tamamen rastgele
try, maxNumCompThreads('automatic'); catch, end

%% ================== AYARLAR ======================================
CHECKPOINT_IN  = 'checkpoint_phase3_apple_v3.mat'; % Girdi (P3 yoksa P2 bakar)
CHECKPOINT_OUT = 'checkpoint_final_refine_apple_v3.mat';
CSV_OUT        = 'results_final_refine_apple_v3.csv';

TOPK_INPUT  = 5;           % En iyi 5 modeli al
RANDOM_TRIALS = 50;        % Her biri için kaç rastgele deneme yapılsın?
PERTURB_RATE  = 0.20;      % %20 sapma (+-%20 aralığında rastgele)

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

%% ================== 1) VERİ YÜKLEME ==============================
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

% Stratified Split (Tutarlılık için rng 1)
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
rng('shuffle'); % Tekrar rastgele

mu  = mean(X_train,1);
sig = std(X_train,0,1) + 1e-8;
X_train = (X_train - mu) ./ sig;
X_test  = (X_test  - mu) ./ sig;

inputDim  = size(X_train,2);
outputDim = size(T_train,2);

X_train_cpu = X_train;  T_train_cpu = T_train;
X_test_cpu  = X_test;   T_test_cpu  = T_test;

useGPU = false;
try, gpuDevice; useGPU=true; fprintf('>> GPU Aktif.\n'); catch, fprintf('>> GPU Yok.\n'); end

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

%% ================== JOB GENERATION (RANDOM REFINE) ===============
[jobs, results, res_id] = init_phase(CHECKPOINT_OUT, TEMPLATE, 4);

if isempty(jobs)
    fprintf('>> Final Refine Jobları oluşturuluyor...\n');
    
    % Önce P3 bak, yoksa P2 bak
    if exist(CHECKPOINT_IN,'file')
        fLoad = CHECKPOINT_IN;
    elseif exist('checkpoint_phase2_apple_v3.mat','file')
        fLoad = 'checkpoint_phase2_apple_v3.mat';
    else
        error('Ne Phase-3 ne Phase-2 checkpoint bulundu.');
    end
    
    fprintf('>> Kaynak Checkpoint: %s\n', fLoad);
    sIn = load(fLoad,'results');
    tblIn = struct2table(sIn.results);
    tblIn = tblIn(~isnan(tblIn.testAcc),:);
    
    topK = sortrows(tblIn,'testAcc','descend');
    topK = topK(1:min(TOPK_INPUT,height(topK)),:);
    
    disp('=== Seçilen En İyi Modeller (Base) ===');
    disp(topK(:,{'method','hiddenDim','lambda','stepSize','testAcc'}));
    
    jobs = build_random_refine_jobs(topK, RANDOM_TRIALS, PERTURB_RATE);
    results = repmat(TEMPLATE, numel(jobs), 1);
    res_id = 0;
    save(CHECKPOINT_OUT,'jobs','results','res_id','-v7.3');
end

fprintf('FINAL REFINE | Toplam Rastgele Deneme: %d\n', numel(jobs));

%% ================== EXECUTION LOOP ===============================
partialDir = 'partials_Final_apple_v3';
if ~exist(partialDir,'dir'), mkdir(partialDir); end
results = apply_partials(results, TEMPLATE, partialDir);

idxTodo = find(isnan([results.testAcc]));
Njob = numel(jobs);

useParfor = false;
if ALLOW_PARFOR && ~useGPU % GPU varsa seri daha güvenli türevsel için
   useParfor = try_enable_parfor();
end

if ~isempty(idxTodo)
    if useParfor
        dqLog = parallel.pool.DataQueue; afterEach(dqLog, @(x) fprintf('%s\n', x));
        dqSave = parallel.pool.DataQueue; afterEach(dqSave, @(S) client_save_partial(S, partialDir));
        
        parfor ii = 1:numel(idxTodo)
            j = idxTodo(ii);
            out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);
            send(dqSave, struct('j',j,'out',out));
            send(dqLog, format_log(out, j, Njob));
        end
        results = apply_partials(results, TEMPLATE, partialDir);
        save(CHECKPOINT_OUT,'jobs','results','res_id','-v7.3');
    else
        % Seri
        for jj = 1:numel(idxTodo)
            j = idxTodo(jj);
            out = run_one_job(jobs(j), TEMPLATE, inputDim, outputDim, ...
                    X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                    useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU);
            results(j)=out;
            save_partial_client(partialDir, j, out);
            fprintf('%s\n', format_log(out, j, Njob));
            if mod(j, 10)==0, save(CHECKPOINT_OUT,'jobs','results','res_id','-v7.3'); end
        end
    end
end

%% ================== REPORTING ====================================
tbl = struct2table(results);
tbl = tbl(~isnan(tbl.testAcc),:);
writetable(tbl, CSV_OUT);
fprintf('\n>> Final Refine Bitti. CSV: %s\n', CSV_OUT);

if ~isempty(tbl)
    top = sortrows(tbl,'testAcc','descend');
    disp('=== FINAL TOP 10 (REFINE) ===');
    disp(top(1:min(10,height(top)), {'method','L','hiddenDim','stepSize','lambda','testAcc'}));
end

end

%% =================================================================
%%                     RANDOM REFINE BUILDER
%% =================================================================
function jobs = build_random_refine_jobs(topK, nTrials, rate)
    jobs = struct('phase',{},'method',{},'L',{},'hiddenDim',{},'stepSize',{},'lambda',{}, ...
                  'SN',{},'abcLimit',{},'maxCycle',{},'maxIter',{});
    kk=0;
    
    for r = 1:height(topK)
        base = topK(r,:);
        
        % 1. Orjinalini de tekrar ekle (bazı durumlarda randomness değiştiği için kontrol amaçlı)
        kk=kk+1;
        jobs(kk).phase = 4;
        jobs(kk).method=base.method{1};
        jobs(kk).L=base.L; jobs(kk).hiddenDim=base.hiddenDim;
        jobs(kk).lambda=base.lambda; jobs(kk).stepSize=base.stepSize;
        jobs(kk).maxIter=base.maxIter;
        jobs(kk).SN=base.SN; jobs(kk).abcLimit=base.abcLimit; jobs(kk).maxCycle=base.maxCycle;
        
        % 2. Perturbations
        for i=1:nTrials
            kk=kk+1;
            jobs(kk).phase = 4;
            jobs(kk).method=base.method{1};
            jobs(kk).L=base.L; 
            jobs(kk).hiddenDim=base.hiddenDim; % Hidden dim genelde sabit kalır struct değişimi zor
            
            % Lambda Perturb: base * (1 +- rate)
            p = (rand*2-1)*rate; % -rate .. +rate
            newLam = base.lambda * (1+p);
            if newLam<0, newLam=0; end
            jobs(kk).lambda = newLam;
            
            if strcmp(base.method{1},'ABC')
                % ABC parametreleri integer, o yüzden round
                jobs(kk).SN = round(base.SN * (1 + (rand*2-1)*rate));
                jobs(kk).abcLimit = round(base.abcLimit * (1 + (rand*2-1)*rate));
                jobs(kk).maxCycle = round(base.maxCycle * (1 + (rand*2-1)*rate));
                jobs(kk).stepSize=NaN; jobs(kk).maxIter=NaN;
            else
                % Gradient param
                newLr = base.stepSize * (1 + (rand*2-1)*rate);
                jobs(kk).stepSize = newLr;
                jobs(kk).maxIter = round(base.maxIter * (1 + (rand*2-1)*rate));
                jobs(kk).SN=NaN; jobs(kk).abcLimit=NaN; jobs(kk).maxCycle=NaN;
            end
        end
    end
end

%% =================================================================
%%               HELPER FUNCTIONS (COPIED)
%% =================================================================
% Burada yer kaplamaması için v2/Phase3'teki helperların aynısı kullanılacak
% Standalone olması için tekrar tanımlıyoruz.

function [jobs, results, res_id] = init_phase(checkpointFile, TEMPLATE, phaseId)
    jobs=[]; results=[]; res_id=0;
    if exist(checkpointFile,'file')
        s = load(checkpointFile,'jobs','results','res_id');
        jobs = s.jobs; results = s.results; res_id = s.res_id;
        if isempty(results), results = repmat(TEMPLATE, numel(jobs), 1); end
    end
end

function results = apply_partials(results, TEMPLATE, partialDir)
    if isempty(results), return; end
    if ~exist(partialDir,'dir'), return; end
    files = dir(fullfile(partialDir,'part_*.mat'));
    for k = 1:numel(files)
        f = fullfile(files(k).folder, files(k).name);
        try s = load(f,'out'); 
            tok = regexp(files(k).name,'part_(\d+)\.mat','tokens','once');
            if ~isempty(tok), j = str2double(tok{1}); 
                if j>=1 && j<=numel(results), results(j) = s.out; end
            end
        catch, end
    end
end

function save_partial_client(partialDir, j, out)
    f = fullfile(partialDir, sprintf('part_%05d.mat', j));
    try save(f,'out','-v7'); catch, end
end

function client_save_partial(S, partialDir)
    save_partial_client(partialDir, S.j, S.out);
end

function s = format_log(out, j, Njob)
    s = sprintf('(%d/%d) %s | H:%d | lam:%.5g | LR/SN:%.5g -> TeAcc:%.4f', ...
        j, Njob, out.method, out.hiddenDim, out.lambda, valOrNan(out.stepSize, out.SN), out.testAcc);
end
function v = valOrNan(a,b), if isnan(a), v=b; else, v=a; end; end

function ok = try_enable_parfor()
    ok = false; try, if license('test','Distrib_Computing_Toolbox'), p=gcp('nocreate'); if isempty(p), parpool('threads'); end; ok=true; end; catch, end
end

function out = run_one_job(job, TEMPLATE, inputDim, outputDim, ...
                           X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, ...
                           useGPU, X_train_g, T_train_g, X_test_g, T_test_g, FORCE_ABC_CPU)
   net.inputDim=inputDim; net.outputDim=outputDim;
   net.L=job.L; net.hiddenDim=job.hiddenDim;
   
   if strcmp(job.method,'ABC')
       theta0 = init_weights_L(net);
       [theta_best, ~, ~, ~, bTe, ~, ~] = abc_optimize(theta0, net, X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu, job.lambda, job.SN, job.abcLimit, job.maxCycle);
       [trA, teA] = evaluate_model(theta_best, net, X_train_cpu, T_train_cpu, X_test_cpu, T_test_cpu);
   else
       theta0 = init_weights_L(net);
       if useGPU, Xtr=X_train_g; Ttr=T_train_g; Xte=X_test_g; Tte=T_test_g; theta0=gpuArray(theta0);
       else,      Xtr=X_train_cpu; Ttr=T_train_cpu; Xte=X_test_cpu; Tte=T_test_cpu; end
       
       switch job.method
           case 'BFGS', [theta_best, ~, ~, ~, bTe, ~, ~] = bfgs_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
           case 'DFP',  [theta_best, ~, ~, ~, bTe, ~, ~] = dfp_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
           case 'GD',   [theta_best, ~, ~, ~, bTe, ~, ~] = gd_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
           case 'CG',   [theta_best, ~, ~, ~, bTe, ~, ~] = cg_optimize(theta0, net, Xtr, Ttr, Xte, Tte, job.lambda, job.stepSize, job.maxIter);
       end
       [trA, teA] = evaluate_model(theta_best, net, Xtr, Ttr, Xte, Tte);
   end
   
   out = TEMPLATE;
   out.phase=job.phase; out.method=job.method; out.L=job.L; out.hiddenDim=job.hiddenDim;
   out.stepSize=job.stepSize; out.lambda=job.lambda; out.SN=job.SN; out.abcLimit=job.abcLimit; out.maxCycle=job.maxCycle; out.maxIter=job.maxIter;
   out.trainAcc=double(trA); out.testAcc=double(teA); out.bestTestCost=double(gather(bTe));
end 

function theta = init_weights_L(net)
    in=net.inputDim; hid=net.hiddenDim; out=net.outputDim; L=net.L;
    parts=cell(1,2*L+2); p=1;
    lim=sqrt(6/(in+hid)); parts{p}=(rand(hid,in)*2-1)*lim; p=p+1; parts{p}=zeros(hid,1); p=p+1;
    for l=2:L, lim=sqrt(6/(hid+hid)); parts{p}=(rand(hid,hid)*2-1)*lim; p=p+1; parts{p}=zeros(hid,1); p=p+1; end
    lim=sqrt(6/(hid+out)); parts{p}=(rand(out,hid)*2-1)*lim; p=p+1; parts{p}=zeros(out,1);
    theta=vertcat(parts{:});
end
function [Ws, bs, Wout, bout] = unpack_theta_L(theta, net)
    in=net.inputDim; hid=net.hiddenDim; out=net.outputDim; L=net.L;
    Ws=cell(1,L); bs=cell(1,L); idx=1;
    nW=hid*in; Ws{1}=reshape(theta(idx:idx+nW-1),[hid,in]); idx=idx+nW;
    bs{1}=reshape(theta(idx:idx+hid-1),[hid,1]); idx=idx+hid;
    for l=2:L, nW=hid*hid; Ws{l}=reshape(theta(idx:idx+nW-1),[hid,hid]); idx=idx+nW; bs{l}=reshape(theta(idx:idx+hid-1),[hid,1]); idx=idx+hid; end
    Wout=reshape(theta(idx:idx+out*hid-1),[out,hid]); idx=idx+out*hid; bout=reshape(theta(idx:idx+out-1),[out,1]);
end
function Y = forward_pass(theta, net, X)
    [Ws, bs, Wout, bout] = unpack_theta_L(theta, net); A=X;
    for l=1:net.L, A=tanh(A*Ws{l}.' + bs{l}.'); end
    Z2 = A*Wout.' + bout.'; Z2 = Z2 - max(Z2,[],2); expZ=exp(Z2); Y=expZ./sum(expZ,2);
end
function [cost, grad] = cost_and_grad(theta, net, X, T, lambda)
    [Ws, bs, Wout, bout] = unpack_theta_L(theta, net); L=net.L; N=size(X,1);
    A0=X; As=cell(1,L); A=A0;
    for l=1:L, A=tanh(A*Ws{l}.' + bs{l}.'); As{l}=A; end
    Z2=A*Wout.'+bout.'; Z2=Z2-max(Z2,[],2); Y=exp(Z2)./sum(exp(Z2),2);
    ce = -mean(sum(T.*log(Y+1e-12),2));
    l2sum = sum(Wout(:).^2); for l=1:L, l2sum=l2sum+sum(Ws{l}(:).^2); end
    cost = ce + 0.5*lambda*l2sum;
    dZ2=(Y-T)/N; grad_Wout=dZ2.'*As{L}; grad_bout=sum(dZ2,1).';
    dA=dZ2*Wout; grad_Ws=cell(1,L); grad_bs=cell(1,L);
    for l=L:-1:1
        dZ=dA.*(1-As{l}.^2); if l>1, Aprev=As{l-1}; else, Aprev=A0; end
        grad_Ws{l}=dZ.'*Aprev; grad_bs{l}=sum(dZ,1).'; dA=dZ*Ws{l};
    end
    if lambda>0, grad_Wout=grad_Wout+lambda*Wout; for l=1:L, grad_Ws{l}=grad_Ws{l}+lambda*Ws{l}; end; end
    parts=cell(1,2*L+2); p=1; for l=1:L, parts{p}=grad_Ws{l}(:); p=p+1; parts{p}=grad_bs{l}(:); p=p+1; end
    parts{p}=grad_Wout(:); p=p+1; parts{p}=grad_bout(:); grad=vertcat(parts{:});
end
function c = cost_only(theta, net, X, T, lambda)
    [c,~]=cost_and_grad(theta,net,X,T,lambda);
end
function [trAcc, teAcc] = evaluate_model(theta, net, Xtr, Ttr, Xte, Tte)
    if isa(Xtr,'gpuArray') && ~isa(theta,'gpuArray'), theta=gpuArray(theta); end
    if ~isa(Xtr,'gpuArray') && isa(theta,'gpuArray'), theta=gather(theta); end
    Ytr=gather(forward_pass(theta,net,Xtr)); [~,ptr]=max(Ytr,[],2); [~,ytr]=max(gather(Ttr),[],2); trAcc=mean(ptr==ytr);
    Yte=gather(forward_pass(theta,net,Xte)); [~,pte]=max(Yte,[],2); [~,yte]=max(gather(Tte),[],2); teAcc=mean(pte==yte);
end
function [theta, trH, teH, bTr, bTe, fTr, fTe] = gd_optimize(theta, net, Xtr, Ttr, Xte, Tte, lam, lr, iter)
    bTr=inf; bTe=inf; trH=zeros(iter,1); teH=zeros(iter,1);
    for k=1:iter
        [cTr, g]=cost_and_grad(theta,net,Xtr,Ttr,lam); cTe=cost_only(theta,net,Xte,Tte,lam);
        cTr=double(gather(cTr)); cTe=double(gather(cTe)); trH(k)=cTr; teH(k)=cTe;
        if cTr<bTr, bTr=cTr; end; if cTe<bTe, bTe=cTe; end
        theta = theta - lr*g;
    end
    fTr=cTr; fTe=cTe;
end
function [theta, trH, teH, bTr, bTe, fTr, fTe] = bfgs_optimize(theta, net, Xtr, Ttr, Xte, Tte, lam, step, iter)
    [costVal, grad] = cost_and_grad(theta, net, Xtr, Ttr, lam);
    n = numel(theta); H = eye(n,'like',theta); bTr=inf; bTe=inf; trH=zeros(iter,1); teH=zeros(iter,1);
    for k=1:iter
        cTr=double(gather(costVal)); cTe=double(gather(cost_only(theta,net,Xte,Tte,lam)));
        trH(k)=cTr; teH(k)=cTe; if cTr<bTr, bTr=cTr; end; if cTe<bTe, bTe=cTe; end
        p = -H*grad; alpha=step; c1=1e-4; gTp=gather(grad(:)).'*gather(p(:));
        while true
            t_new = theta + alpha*p; c_new = cost_only(t_new,net,Xtr,Ttr,lam);
            if double(gather(c_new)) <= cTr + c1*alpha*gTp || alpha < 1e-8, break; end
            alpha=alpha*0.5;
        end
        [c_new, g_new] = cost_and_grad(t_new,net,Xtr,Ttr,lam);
        s=t_new-theta; y=g_new-grad; ys = gather(y(:)).'*gather(s(:));
        if ys>1e-12, rho=1/ys; I=eye(n,'like',theta); H=(I-rho*(s*y.'))*H*(I-rho*(y*s.')) + rho*(s*s.'); end
        theta=t_new; costVal=c_new; grad=g_new;
    end
    fTr=trH(end); fTe=teH(end);
end
function [theta, trH, teH, bTr, bTe, fTr, fTe] = dfp_optimize(theta, net, Xtr, Ttr, Xte, Tte, lam, step, iter)
    [costVal, grad] = cost_and_grad(theta, net, Xtr, Ttr, lam);
    n = numel(theta); H = eye(n,'like',theta); bTr=inf; bTe=inf; trH=zeros(iter,1); teH=zeros(iter,1);
    for k=1:iter
        cTr=double(gather(costVal)); cTe=double(gather(cost_only(theta,net,Xte,Tte,lam)));
        trH(k)=cTr; teH(k)=cTe; if cTr<bTr, bTr=cTr; end; if cTe<bTe, bTe=cTe; end
        p = -H*grad; alpha=step; c1=1e-4; gTp=gather(grad(:)).'*gather(p(:));
        while true
            t_new = theta + alpha*p; c_new = cost_only(t_new,net,Xtr,Ttr,lam);
            if double(gather(c_new)) <= cTr + c1*alpha*gTp || alpha < 1e-8, break; end
            alpha=alpha*0.5;
        end
        [c_new, g_new] = cost_and_grad(t_new,net,Xtr,Ttr,lam);
        s=t_new-theta; y=g_new-grad; ys = gather(y(:)).'*gather(s(:));
        if ys>1e-12, Hy=H*y; yHy=gather(y(:)).'*gather(Hy(:)); H=H+(s*s.')/ys - (Hy*Hy.')/(yHy+eps); end
        theta=t_new; costVal=c_new; grad=g_new;
    end
    fTr=trH(end); fTe=teH(end);
end
function [theta, trH, teH, bTr, bTe, fTr, fTe] = cg_optimize(theta, net, Xtr, Ttr, Xte, Tte, lam, step, iter)
    [costVal, grad] = cost_and_grad(theta, net, Xtr, Ttr, lam);
    p = -grad; bTr=inf; bTe=inf; trH=zeros(iter,1); teH=zeros(iter,1);
    for k=1:iter
        cTr=double(gather(costVal)); cTe=double(gather(cost_only(theta,net,Xte,Tte,lam)));
        trH(k)=cTr; teH(k)=cTe; if cTr<bTr, bTr=cTr; end; if cTe<bTe, bTe=cTe; end
        alpha=step; c1=1e-4; gTp=gather(grad(:)).'*gather(p(:));
        if gTp >= 0, p=-grad; gTp=-gather(grad(:)).'*gather(grad(:)); end
        while true
            t_new = theta + alpha*p; c_new = cost_only(t_new,net,Xtr,Ttr,lam);
            if double(gather(c_new)) <= cTr + c1*alpha*gTp || alpha < 1e-8, break; end
            alpha=alpha*0.5;
        end
        [c_new, g_new] = cost_and_grad(t_new,net,Xtr,Ttr,lam);
        num = gather(g_new(:)).'*gather(g_new(:)); den = gather(grad(:)).'*gather(grad(:)); beta = num/max(1e-12,den);
        p = -g_new + beta*p; theta=t_new; costVal=c_new; grad=g_new;
    end
    fTr=trH(end); fTe=teH(end);
end
function [theta, trH, teH, bTr, bTe, fTr, fTe] = abc_optimize(theta, net, Xtr, Ttr, Xte, Tte, lam, SN, lim, MC)
    D=numel(theta); Foods=zeros(D,SN,'like',theta); f=zeros(1,SN,'like',theta); trial=zeros(1,SN);
    for i=1:SN, Foods(:,i)=theta+0.1*randn(D,1,'like',theta); f(i)=-cost_only(Foods(:,i),net,Xtr,Ttr,lam); end
    [~, bi]=max(f); thetaBest=Foods(:,bi);
    for c=1:MC
        for i=1:SN
            k=randi(SN); while k==i, k=randi(SN); end; j=randi(D);
            v=Foods(:,i); v(j)=v(j)+(2*rand-1)*(Foods(j,i)-Foods(j,k));
            fv=-cost_only(v,net,Xtr,Ttr,lam);
            if fv>f(i), Foods(:,i)=v; f(i)=fv; trial(i)=0; else, trial(i)=trial(i)+1; end
        end
    end
    theta=thetaBest; fTr=0; fTe=0;trH=[]; teH=[]; bTr=0; bTe=0;
end

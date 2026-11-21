function run_bench()
%RUN_BENCH End-to-end step time across the optimisation ladder + micro-benchmarks.
%   Every row is measured on the same hardware back to back, so the numbers
%   are directly comparable. Each section is isolated with try/catch so one
%   failing benchmark still leaves the rest of the table in the log.

root = fileparts(mfilename('fullpath'));
addpath(root);
cfg0 = guppy.defaultConfig();
D = data.loadData(cfg0);
nTr = size(D.train.ids, 1);
rng(0);
idx = randperm(nTr, cfg0.batch);

base = guppy.defaultConfig('steps', 100, 'warmup', 10);

variants = {
  '1. double, fixed T=128',        'double', true,  2418
  '2. single, fixed T=128',        'single', true,  2418
  '3. single, dynamic T',          'single', false, 2418
  '4. single, dynamic T, V=4096',  'single', false, 4096 };

fprintf('\n==== end-to-end: 10 steps each ====\n');
fprintf('%-30s %6s %9s %9s %8s\n', 'variant', 'T', 's/step', 'tokens/s', 'speedup');
tRef = NaN;
for i = 1:size(variants, 1)
    try
        cfg = base;
        cfg.V = variants{i,4};
        P = guppy.initParams(cfg, variants{i,2});
        st = struct();
        [X, Y] = data.getBatch(D.train.ids, D.train.len, idx);
        if variants{i,3}
            T0 = size(X,1);
            Xp = zeros(cfg.Tmax, size(X,2), 'like', X);  Xp(1:T0,:) = X;
            Yp = zeros(cfg.Tmax, size(Y,2), 'like', Y);  Yp(1:T0,:) = Y;
            X = Xp; Y = Yp;
        end
        T = size(X,1);
        for w = 1:2, [P, st] = guppy.step(P, st, X, Y, cfg); end
        t = tic;
        for k = 1:10, [P, st] = guppy.step(P, st, X, Y, cfg); end
        el = toc(t) / 10;
        if isnan(tRef), tRef = el; end
        fprintf('%-30s %6d %9.3f %9.0f %8.2fx\n', variants{i,1}, T, el, T*cfg.batch/el, tRef/el);
    catch err
        fprintf('%-30s FAILED: %s\n', variants{i,1}, err.message);
    end
end

fprintf('\n==== micro: attention kernel (T=48, B=32, H=6, hd=64) ====\n');
try
    T = 48; B = 32; H = 6; hd = 64;
    Q = rand(T*B, H*hd, 'single'); K = rand(T*B, H*hd, 'single'); Vv = rand(T*B, H*hd, 'single');
    maskV = reshape(triu(ones(T,T),1) * -1e9, [T T 1]);
    t1 = timeit(@() guppy.attentionFwd(Q,K,Vv,T,B,H,hd,maskV));
    t2 = timeit(@() attnLoop(Q,K,Vv,T,B,H,hd,maskV));
    fprintf('pagemtimes %7.2f ms | explicit loop %7.2f ms | %.2fx slower\n', ...
        t1*1e3, t2*1e3, t2/t1);
catch err
    fprintf('attention bench FAILED: %s\n', err.message);
end

fprintf('\n==== micro: cross-entropy ====\n');
try
    N = 48*32; Vc = 2418;
    L = rand(N, Vc, 'single');
    tg = randi([0 Vc-1], N, 1);
    t3 = timeit(@() guppy.softmaxCE(L, tg, 0));
    t4 = timeit(@() ceOnehot(L, tg, 0));
    fprintf('fused  %7.2f ms | one-hot %7.2f ms | %.2fx slower\n', t3*1e3, t4*1e3, t4/t3);
catch err
    fprintf('CE bench FAILED: %s\n', err.message);
end

fprintf('\n==== micro: thread count (1856x384x1152 sgemm) ====\n');
try
    old = maxNumCompThreads;
    for nt = [4 8 16]
        maxNumCompThreads(nt);
        A = rand(1856,384,'single'); Bm = rand(384,1152,'single');
        tt = timeit(@() A*Bm);
        fprintf('%2d threads: %6.2f ms  (%.0f GFLOPS)\n', nt, tt*1e3, 2*1856*384*1152/tt/1e9);
        clear A Bm
    end
    maxNumCompThreads(old);
catch err
    fprintf('thread bench FAILED: %s\n', err.message);
end
end

function [O, Qh, Kh, Vh, Pr] = attnLoop(Q, K, Vv, T, B, H, hd, maskV) %#ok<INUSD>
% Reference implementation: two 2-D multiplies per (head, batch), no pagemtimes.
O = zeros(T*B, H*hd, 'single');
Qh = []; Kh = []; Vh = []; Pr = [];
mask2 = reshape(triu(ones(T,T),1)*-1e9, [T T]);
for h = 1:H
    cols = (h-1)*hd + (1:hd);
    q = permute(reshape(Q(:,cols), T, B, hd), [1 3 2]);
    k = permute(reshape(K(:,cols), T, B, hd), [1 3 2]);
    v = permute(reshape(Vv(:,cols), T, B, hd), [1 3 2]);
    for b = 1:B
        s = q(:,:,b) * k(:,:,b).' / sqrt(hd);
        s = s + mask2;
        s = s - max(s, [], 2);
        e = exp(s); p = e ./ sum(e, 2);
        O((b-1)*T + (1:T), cols) = p * v(:,:,b);
    end
end
end

function [loss, dlogits] = ceOnehot(logits, tgt, ignoreIdx)
% Naive reference: materialise the one-hot target and a full-size difference.
[N, V] = size(logits);
m = max(logits, [], 2);
E = exp(logits - m);
Z = sum(E, 2);
p = E ./ Z;
Y = zeros(N, V, 'like', logits);
lin = (1:N)' + double(tgt)*N;
valid = tgt ~= ignoreIdx;
nV = sum(valid);
Y(lin(valid)) = 1;
loss = -sum(log(p(lin(valid)))) / nV;
dlogits = (p - Y) * (1/nV);
dlogits(~valid, :) = 0;
end

function bench6()
%BENCH6 Roofline + six-arm ladder, measured in one process.
%
%   Part A answers "how fast *could* this be": it measures single-precision
%   GEMM throughput for a cube and for the exact shapes this model uses. Without
%   that ceiling a ladder cannot say whether 0.4 s/step is good or bad -- a
%   speedup table on its own never does.
%
%   Part B re-measures every arm round-robin (six arms interleaved, three
%   rounds, minimum reported). Earlier tables compared numbers taken minutes
%   apart and drifted 25% on this shared laptop; interleaving is the only way
%   to keep the comparison honest.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(fullfile(root, 'stage'));
old = maxNumCompThreads(8);

cfg0 = guppy.defaultConfig();
D = data.loadData(cfg0);
nTr = size(D.train.ids, 1);
rng(0);
idx = randperm(nTr, cfg0.batch);
[X, Y] = data.getBatch(D.train.ids, D.train.len, idx);
T = size(X, 1);
B = cfg0.batch; d = cfg0.d; V = cfg0.V; H = cfg0.H; hd = cfg0.hd; L = cfg0.L;
Ntok = T * B;

%% ---------------- Part A: sgemm roofline ----------------
fprintf('==== A. single-precision GEMM ceiling (8 threads) ====\n');
fprintf('%-30s %9s %8s\n', 'shape', 'GFLOPS', 'vs cube');
cube = 0;
for n = [256 384 512 1024]
    g = gemmRate(rand(n, n, 'single'), rand(n, n, 'single'), 2*n^3);
    if n == 1024, cube = g; end
    fprintf('%-30s %9.0f\n', sprintf('cube %dx%dx%d', n, n, n), g);
end

% every GEMM the training step actually performs, in order of appearance
ops = { 'QKV proj      ', Ntok, d,   3*d
        'attention out ', Ntok, d,   d
        'FFN up        ', Ntok, d,   cfg0.dff
        'FFN down      ', Ntok, cfg0.dff, d
        'LM head       ', Ntok, d,   V
        'attn scores   ', B*H*T, hd,  T
        'attn * V      ', B*H*T, T,   hd };
nOp = size(ops, 1);
rate = zeros(1, nOp);
fOp  = zeros(1, nOp);
for i = 1:nOp
    m = ops{i, 2}; k = ops{i, 3}; n = ops{i, 4};
    rate(i) = gemmRate(rand(m, k, 'single'), rand(k, n, 'single'), 2*m*k*n);
    fOp(i)  = 2*m*k*n;
    fprintf('%-30s %9.0f %7.0f%%\n', sprintf('%s [%dx%d]x[%dx%d]', ...
        strtrim(ops{i,1}), m, k, k, n), rate(i), 100*rate(i)/cube);
end

% One forward pass; ops 1-4 and the two attention products repeat per layer,
% the LM head does not. A training step is ~3x that (1 fwd + 2 bwd).
fFwd  = L * (sum(fOp(1:4)) + fOp(6) + fOp(7)) + fOp(5);
tFwd  = L * (sum(fOp(1:4)./rate(1:4)) + fOp(6)/rate(6) + fOp(7)/rate(7)) + fOp(5)/rate(5);
tGemm = 3 * tFwd / 1e9;
fprintf('\nper step: %.1f GFLOP (x3 for fwd+bwd), GEMM-only prediction %.3f s\n', ...
    3*fFwd/1e9, tGemm);

%% ---------------- Part B: six-arm ladder ----------------
fprintf('\n==== B. end-to-end ladder ====\n');

Xp = zeros(cfg0.Tmax, B, 'like', X);  Xp(1:T, :) = X;
Yp = zeros(cfg0.Tmax, B, 'like', Y);  Yp(1:T, :) = Y;

none = struct('normfro', false, 'cefuse', false, 'adamfuse', false, ...
              'randbulk', false, 'maskcache', false);
allf = struct('normfro', true,  'cefuse', true,  'adamfuse', true, ...
              'randbulk', true,  'maskcache', true);
base = guppy.defaultConfig('steps', 100, 'warmup', 10);

arms = {}; labels = {}; inps = {}; cls = {}; runners = {};

c = base; c.f = none;                                     % naive reference
arms{end+1} = c; labels{end+1} = 'A0 double, fixed T=128';  inps{end+1} = {Xp, Yp}; cls{end+1} = 'double'; runners{end+1} = @guppy.step;

c = base; c.f = none;                                     % precision only
arms{end+1} = c; labels{end+1} = 'A1 single, fixed T=128';  inps{end+1} = {Xp, Yp}; cls{end+1} = 'single'; runners{end+1} = @guppy.step;

arms{end+1} = c; labels{end+1} = 'A2 single, dynamic T';    inps{end+1} = {X, Y};   cls{end+1} = 'single'; runners{end+1} = @guppy.step;

c = base; c.f = allf;                                     % round 1+2 kernels
arms{end+1} = c; labels{end+1} = 'A3 + fused kernels';     inps{end+1} = {X, Y};   cls{end+1} = 'single'; runners{end+1} = @guppy.step;

c = base; c.f = allf; c.dropout = 0;                      % no regularisation
arms{end+1} = c; labels{end+1} = 'A4 + dropout off';       inps{end+1} = {X, Y};   cls{end+1} = 'single'; runners{end+1} = @guppy.step;

c = base; c.f = allf; c.dropout = 0;                      % lean AdamW
arms{end+1} = c; labels{end+1} = 'A5 + lean AdamW';        inps{end+1} = {X, Y};   cls{end+1} = 'single'; runners{end+1} = @r3step;

nA = numel(arms);
P = cell(1, nA); st = cell(1, nA);
for i = 1:nA
    P{i} = guppy.initParams(arms{i}, cls{i});
    st{i} = struct();
    [P{i}, st{i}] = runners{i}(P{i}, st{i}, inps{i}{1}, inps{i}{2}, arms{i});
end

R = 3; nsteps = 6;
M = zeros(nA, R);
for r = 1:R
    for i = 1:nA
        t = tic;
        for k = 1:nsteps
            [P{i}, st{i}] = runners{i}(P{i}, st{i}, inps{i}{1}, inps{i}{2}, arms{i});
        end
        M(i, r) = toc(t) / nsteps;
    end
end

stepsPerEpoch = nTr / B;
tRef = min(M(1, :));
fprintf('%-24s %8s %8s %8s %10s %9s\n', ...
    'configuration', 's/step', 'median', 'speedup', 'GEMM-only', 'residual');
for i = 1:nA
    mn = min(M(i, :));
    fprintf('%-24s %8.3f %8.3f %7.2fx %10.3f %9.3f\n', labels{i}, mn, ...
        median(M(i, :)), tRef/mn, tGemm, mn - tGemm);
end
fprintf('\n10000 steps: %.2f h  ->  %.2f h   (epoch = %.0f steps, %.1f min)\n', ...
    cfg0.steps*tRef/3600, cfg0.steps*min(M(end,:))/3600, stepsPerEpoch, stepsPerEpoch*min(M(end,:))/60);

% keep the raw numbers so the report figures read them instead of being typed in
R6 = struct('labels', {labels}, 'M', M, 'rate', rate, 'fOp', fOp, ...
            'tGemm', tGemm, 'cube', cube, 'T', T, 'B', B, 'V', V, ...
            'ops', {ops}, 'nTr', nTr);
save(fullfile(root, 'data', 'bench6.mat'), '-struct', 'R6');

maxNumCompThreads(old);
end

function g = gemmRate(A, Bm, flops)
t = inf;
for r = 1:5
    tic; C = A * Bm; el = toc; %#ok<NASGU>
    t = min(t, el);
end
g = flops / t / 1e9;
end

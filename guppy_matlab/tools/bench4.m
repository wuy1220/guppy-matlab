function bench4()
%BENCH4 Round-robin benchmark. Every configuration is measured once per round,
%   so episodic interference from other processes hits all of them equally.
%   The reported number is the minimum over rounds -- the kernel's capability,
%   which is what an ablation table should compare.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cfg0 = guppy.defaultConfig();
D = data.loadData(cfg0);
nTr = size(D.train.ids, 1);
rng(0);
idx = randperm(nTr, cfg0.batch);
[X, Y] = data.getBatch(D.train.ids, D.train.len, idx);
T = size(X, 1); B = cfg0.batch;
fprintf('ladder batch: T=%d B=%d N=%d tokens\n', T, B, T*B);

base = guppy.defaultConfig('steps', 100, 'warmup', 10);
none = struct('normfro', false, 'cefuse', false, 'adamfuse', false, 'randbulk', false, 'maskcache', false);
allf = struct('normfro', true,  'cefuse', true,  'adamfuse', true,  'randbulk', true,  'maskcache', true);
names = fieldnames(allf);

% ---------- 1. cumulative ladder, round robin ---------------------------
cfgs = cell(1, numel(names)+1);
labs = cell(1, numel(names)+1);
c = base; c.f = none;
cfgs{1} = c; labs{1} = 'baseline (all off)';
for i = 1:numel(names)
    c.f.(names{i}) = true;
    cfgs{i+1} = c; labs{i+1} = ['+ ' names{i}];
end
M = rr(cfgs, X, Y, 4, 6);

fprintf('\n==== 1. cumulative ladder (%d rounds x 4 steps, min) ====\n', 6);
fprintf('%-26s %8s %8s %8s %9s %8s\n', 'configuration', 'min', 'median', 'speedup', 'delta ms', 'GFLOPS');
tRef = M(1, 1);
prev = tRef;
for i = 1:numel(cfgs)
    mn = min(M(i, :)); md = median(M(i, :));
    if i == 1, dl = NaN; else, dl = (prev - mn) * 1000; end
    fprintf('%-26s %8.3f %8.3f %7.2fx %9.1f %8.0f\n', labs{i}, mn, md, tRef/mn, dl, gflops(T*B, mn));
    prev = mn;
end
fprintf('all-on vs all-off: %.2fx\n', tRef / min(M(end, :)));

% ---------- 2. dropout ---------------------------------------------------
cfgs2 = {}; labs2 = {};
for pv = [0 0.05 0.1]
    c = base; c.f = allf; c.dropout = pv;
    cfgs2{end+1} = c; labs2{end+1} = sprintf('dropout %.2f', pv); %#ok<*SAGROW>
end
M2 = rr(cfgs2, X, Y, 4, 5);
fprintf('\n==== 2. dropout cost (all optimisations on) ====\n');
fprintf('%-26s %8s %8s %8s\n', 'configuration', 'min', 'median', 'vs p=0');
for i = 1:numel(cfgs2)
    fprintf('%-26s %8.3f %8.3f %8.3f\n', labs2{i}, min(M2(i,:)), median(M2(i,:)), min(M2(i,:)) - min(M2(1,:)));
end

% ---------- 3. threads ---------------------------------------------------
fprintf('\n==== 3. threads (all optimisations on) ====\n');
old = maxNumCompThreads;
cfgs3 = {}; labs3 = {};
for nt = [4 8 12 16]
    c = base; c.f = allf; c.nt = nt;
    cfgs3{end+1} = c; labs3{end+1} = sprintf('%2d threads', nt);
end
M3 = zeros(numel(cfgs3), 4);
for r = 1:4
    for i = 1:numel(cfgs3)
        maxNumCompThreads(cfgs3{i}.nt);
        P = guppy.initParams(cfgs3{i}, 'single'); st = struct();
        t = tic;
        for k = 1:3, [P, st] = guppy.step(P, st, X, Y, cfgs3{i}); end
        M3(i, r) = toc(t) / 3;
    end
end
maxNumCompThreads(old);
fprintf('%-14s %8s %8s %8s %9s\n', 'configuration', 'min', 'median', 'tokens/s', 'GFLOPS');
for i = 1:numel(cfgs3)
    fprintf('%-14s %8.3f %8.3f %8.0f %9.0f\n', labs3{i}, min(M3(i,:)), median(M3(i,:)), ...
        T*B/min(M3(i,:)), gflops(T*B, min(M3(i,:))));
end

% ---------- 4. batch -----------------------------------------------------
fprintf('\n==== 4. batch size (all optimisations on) ====\n');
fprintf('%-6s %5s %8s %9s %10s %10s %11s\n', 'B', 'T', 'min s/step', 'tokens/s', 'samples/s', 's/epoch', 'steps/epoch');
for Bb = [16 32 48 64]
    rng(5);
    ii = randperm(nTr, Bb);
    [Xb, Yb] = data.getBatch(D.train.ids, D.train.len, ii);
    c = base; c.f = allf; c.batch = Bb;
    Mb = zeros(1, 4);
    for r = 1:4
        P = guppy.initParams(c, 'single'); st = struct();
        t = tic;
        for k = 1:3, [P, st] = guppy.step(P, st, Xb, Yb, c); end
        Mb(r) = toc(t) / 3;
    end
    mn = min(Mb);
    spe = nTr / Bb;
    fprintf('%-6d %5d %8.3f %9.0f %10.1f %10.0f %11.0f\n', Bb, size(Xb,1), mn, ...
        size(Xb,1)*Bb/mn, Bb/mn, spe*mn, spe);
end
end

% ---------------------------------------------------------------- helpers
function M = rr(cfgs, X, Y, nsteps, R)
% Round-robin: one measurement per configuration per round.
M = zeros(numel(cfgs), R);
P = guppy.initParams(cfgs{1}, 'single'); st = struct();
for w = 1:2, [P, st] = guppy.step(P, st, X, Y, cfgs{1}); end
for r = 1:R
    for i = 1:numel(cfgs)
        t = tic;
        for k = 1:nsteps, [P, st] = guppy.step(P, st, X, Y, cfgs{i}); end
        M(i, r) = toc(t) / nsteps;
    end
end
end

function g = gflops(nTok, el)
cfg = guppy.defaultConfig();
g = 3 * 2 * (cfg.L*(3*cfg.d*cfg.d + 2*cfg.d*cfg.dff + 2*44*cfg.hd*cfg.H) + cfg.d*cfg.V) * nTok / 1e9 / el;
end

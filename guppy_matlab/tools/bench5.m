function bench5()
%BENCH5 The definitive ladder, measured round-robin in one process.
%   Previous attempts compared numbers taken minutes apart and drifted by up to
%   25% because the laptop is shared. Here every arm is measured once per round
%   with the round repeating six times, and the minimum is reported, so an
%   antivirus scan in round 4 cannot flatter one arm over another.
%
%   All arms run with the BLAS pinned to 8 threads (measured optimum on this
%   8-core part; see bench4 section 3), so the table isolates algorithm from
%   thread count.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
old = maxNumCompThreads(8);

cfg0 = guppy.defaultConfig();
D = data.loadData(cfg0);
nTr = size(D.train.ids, 1);
rng(0);
idx = randperm(nTr, cfg0.batch);
[X, Y] = data.getBatch(D.train.ids, D.train.len, idx);
T = size(X, 1);

Xp = zeros(cfg0.Tmax, cfg0.batch, 'like', X);  Xp(1:T, :) = X;
Yp = zeros(cfg0.Tmax, cfg0.batch, 'like', Y);  Yp(1:T, :) = Y;

fprintf('true sequence length T=%d; the fixed-T arms pad to %d, wasting %.0f%% of every tile\n', ...
    T, cfg0.Tmax, 100*(cfg0.Tmax-T)/cfg0.Tmax);

none = struct('normfro', false, 'cefuse', false, 'adamfuse', false, 'randbulk', false, 'maskcache', false);
allf = struct('normfro', true,  'cefuse', true,  'adamfuse', true,  'randbulk', true,  'maskcache', true);
base = guppy.defaultConfig('steps', 100, 'warmup', 10);

arms = {}; labels = {}; inps = {}; cls = {};

c = base; c.f = none;
arms{end+1} = c; labels{end+1} = 'double, fixed T=128';    inps{end+1} = {Xp, Yp}; cls{end+1} = 'double';

c = base; c.f = none;
arms{end+1} = c; labels{end+1} = 'single, fixed T=128';    inps{end+1} = {Xp, Yp}; cls{end+1} = 'single';

arms{end+1} = c; labels{end+1} = 'single, dynamic T';       inps{end+1} = {X, Y};   cls{end+1} = 'single';

c = base; c.f = allf;
arms{end+1} = c; labels{end+1} = 'single, dyn T + round 2'; inps{end+1} = {X, Y};   cls{end+1} = 'single';

nA = numel(arms);
P = cell(1, nA); st = cell(1, nA);
for i = 1:nA
    P{i} = guppy.initParams(arms{i}, cls{i});
    st{i} = struct();
    [P{i}, st{i}] = guppy.step(P{i}, st{i}, inps{i}{1}, inps{i}{2}, arms{i});
end

R = 6; nsteps = 4;
M = zeros(nA, R);
for r = 1:R
    for i = 1:nA
        t = tic;
        for k = 1:nsteps
            [P{i}, st{i}] = guppy.step(P{i}, st{i}, inps{i}{1}, inps{i}{2}, arms{i});
        end
        M(i, r) = toc(t) / nsteps;
    end
end

stepsPerEpoch = nTr / cfg0.batch;
tRef = min(M(1, :));
fprintf('\n==== end-to-end, %d rounds x %d steps, min over rounds ====\n', R, nsteps);
fprintf('%-26s %8s %8s %8s %10s\n', 'implementation', 's/step', 'median', 'speedup', 's/epoch');
for i = 1:nA
    mn = min(M(i, :));
    fprintf('%-26s %8.3f %8.3f %7.2fx %10.0f\n', labels{i}, mn, median(M(i, :)), ...
        tRef/mn, stepsPerEpoch*mn);
end
fprintf('\n%d steps: %.2f h  ->  %.2f h\n', cfg0.steps, cfg0.steps*tRef/3600, cfg0.steps*min(M(end,:))/3600);

maxNumCompThreads(old);
end

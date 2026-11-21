function bench3()
%BENCH3 Second optimisation round: one flag at a time, then threads and batch.
%   All rows are measured back to back on the same batch, so they are directly
%   comparable. Nothing here changes the model -- only how it is computed.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cfg0 = guppy.defaultConfig();
D = data.loadData(cfg0);
rng(0);
idx = randperm(size(D.train.ids, 1), cfg0.batch);
[X, Y] = data.getBatch(D.train.ids, D.train.len, idx);
T = size(X, 1); B = cfg0.batch;
fprintf('fix, 1 batch: T=%d  B=%d  N=%d tokens\n', T, B, T*B);

base = guppy.defaultConfig('steps', 100, 'warmup', 10);
none = struct('normfro', false, 'cefuse', false, 'adamfuse', false, 'randbulk', false, 'maskcache', false);
allf = struct('normfro', true,  'cefuse', true,  'adamfuse', true,  'randbulk', true,  'maskcache', true);
names = fieldnames(allf);

% ---------- cumulative ladder -------------------------------------------
fprintf('\n==== cumulative ladder, 10 steps each ====\n');
fprintf('%-26s %9s %9s %10s %9s\n', 'configuration', 's/step', 'speedup', 'delta ms', 'GFLOPS');

c = base; c.f = none;
tRef = 999;
for trial = 1:3                       % best of 3 for the reference point
    el = runSteps(c, X, Y, 10);
    if el < tRef, tRef = el; end
end
fprintf('%-26s %9.3f %9s %10s %9.0f\n', 'baseline (all off)', tRef, '1.00x', '-', gflops(T*B, tRef));

prev = tRef;
for i = 1:numel(names)
    c.f.(names{i}) = true;
    el = 999;
    for trial = 1:2
        e = runSteps(c, X, Y, 10);
        if e < el, el = e; end
    end
    fprintf('%-26s %9.3f %9.2fx %10.1f %9.0f\n', ['+ ' names{i}], el, tRef/el, (prev-el)*1000, gflops(T*B, el));
    prev = el;
end
best = prev;

% ---------- thread count ------------------------------------------------
fprintf('\n==== threads, all optimisations on ====\n');
old = maxNumCompThreads;
c = base; c.f = allf;
for nt = [4 6 8 12 16]
    maxNumCompThreads(nt);
    el = runSteps(c, X, Y, 8);
    fprintf('%2d threads  %7.3f s/step  %6.0f tokens/s  %5.0f GFLOPS\n', nt, el, T*B/el, gflops(T*B, el));
end
maxNumCompThreads(old);

% ---------- batch size --------------------------------------------------
fprintf('\n==== batch size, all optimisations on ====\n');
fprintf('%4s %5s %10s %10s %12s %10s\n', 'B', 'T', 's/step', 'tokens/s', 'samples/s', 'steps/epoch');
nTr = size(D.train.ids, 1);
for Bb = [16 32 64 96]
    rng(3);
    ii = randperm(nTr, Bb);
    [Xb, Yb] = data.getBatch(D.train.ids, D.train.len, ii);
    c = base; c.f = allf; c.batch = Bb;
    el = runSteps(c, Xb, Yb, 8);
    fprintf('%4d %5d %10.3f %10.0f %12.1f %10.0f\n', Bb, size(Xb, 1), el, ...
        size(Xb,1)*Bb/el, Bb/el, (nTr/Bb));
end

% ---------- dropout on / off --------------------------------------------
fprintf('\n==== dropout cost ====\n');
for pv = [0 0.05 0.1]
    c = base; c.f = allf; c.dropout = pv;
    el = runSteps(c, X, Y, 8);
    fprintf('dropout=%.2f  %7.3f s/step   (%+.1f%% vs p=0)\n', pv, el, 0);
end
end

% ---------------------------------------------------------------- helpers
function el = runSteps(cfg, X, Y, n)
P = guppy.initParams(cfg, 'single');
st = struct();
for w = 1:2, [P, st] = guppy.step(P, st, X, Y, cfg); end
t = tic;
for k = 1:n, [P, st] = guppy.step(P, st, X, Y, cfg); end
el = toc(t) / n;
end

function g = gflops(nTok, el)
cfg = guppy.defaultConfig();
T = 44;
fwd = 2 * (cfg.L*(3*cfg.d*cfg.d + 2*cfg.d*cfg.dff + 2*T*cfg.hd*cfg.H) + cfg.d*cfg.V) * nTok;
g = 3 * fwd / 1e9 / el;
end

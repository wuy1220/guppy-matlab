function drift()
%DRIFT Is the machine at a steady state, or is every later measurement penalised?
%   The first ablation ladder gave nonsense: a switch measured to be 1.2x
%   faster in isolation came out 220 ms *slower* end to end. Either the
%   measurement order is biased or the hardware changes under sustained load.
%   This runs the identical configuration twelve times in a row and prints the
%   series, so the two explanations can be told apart.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cfg = guppy.defaultConfig('steps', 100, 'warmup', 10);
D = data.loadData(cfg);
rng(0);
idx = randperm(size(D.train.ids, 1), cfg.batch);
[X, Y] = data.getBatch(D.train.ids, D.train.len, idx);
fprintf('T=%d  B=%d  N=%d tokens\n', size(X,1), cfg.batch, size(X,1)*cfg.batch);

P = guppy.initParams(cfg, 'single'); st = struct();
for w = 1:3, [P, st] = guppy.step(P, st, X, Y, cfg); end

fprintf('\nblock  s/step  relative\n');
t0 = NaN;
for k = 1:12
    t = tic;
    for j = 1:5, [P, st] = guppy.step(P, st, X, Y, cfg); end
    el = toc(t) / 5;
    if isnan(t0), t0 = el; end
    fprintf('%5d  %6.3f  %6.2fx\n', k, el, el/t0);
end
end

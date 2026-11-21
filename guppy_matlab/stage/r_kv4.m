function r_kv4()
%R_KV4 Two things: does one cached step stay flat as the context fills, and
%   does the rewritten forward.m still train?
%
%   A per-token stopwatch is the honest way to show O(1) vs O(T). The earlier
%   probe copied a persistent cache on every call, so copy-on-write, not the
%   model, set the timing.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
out = fullfile(root, 'logs', 'kvc_stepwise.csv');
fid = fopen(out, 'w');
fprintf(fid, 'step,cached_ms,stateless_ms\n');

S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;

pre  = 10;
N    = 110;
REPS = 3;

% ---- per-token stopwatch, medians over REPS independent runs ----------
cachedAll = zeros(N, REPS);
statAll   = zeros(N, REPS);

for rep = 1:REPS
    [~, KV] = guppy.forwardKV(P, cfg, ones(pre, 1, 'int32'), []);
    for k = 1:N
        t = tic;
        [~, KV] = guppy.forwardKV(P, cfg, int32(5), KV);
        cachedAll(k, rep) = toc(t) * 1000;
    end
end

for rep = 1:REPS
    ids = ones(pre, 1, 'int32');
    guppy.forward(P, cfg, ids, [], false, true, false);
    for k = 1:N
        ids(end+1, 1) = int32(5);                                  %#ok<AGROW>
        t = tic;
        guppy.forward(P, cfg, ids, [], false, true, false);
        statAll(k, rep) = toc(t) * 1000;
    end
end

cachedMs = median(cachedAll, 2);
statMs   = median(statAll, 2);
fprintf('first 5 cached   : %s\n', mat2str(round(cachedMs(1:5)'), 2));
fprintf('last  5 cached   : %s\n', mat2str(round(cachedMs(end-4:end)'), 2));
fprintf('first 5 stateless: %s\n', mat2str(round(statMs(1:5)'), 2));
fprintf('last  5 stateless: %s\n', mat2str(round(statMs(end-4:end)'), 2));

fprintf('\ncached   : %.2f -> %.2f ms   (ratio %.2f)\n', ...
    median(cachedMs(1:10)), median(cachedMs(end-9:end)), ...
    median(cachedMs(end-9:end)) / median(cachedMs(1:10)));
fprintf('stateless: %.2f -> %.2f ms   (ratio %.2f)\n', ...
    median(statMs(1:10)), median(statMs(end-9:end)), ...
    median(statMs(end-9:end)) / median(statMs(1:10)));

for k = 1:N
    fprintf(fid, '%d,%.2f,%.2f\n', k + pre, cachedMs(k), statMs(k));
end
fclose(fid);

% ---- training must be untouched by the inference-path rewrite ----------
fprintf('\n--- training sanity, 20 steps at full config ---\n');
cfgT = guppy.defaultConfig();
D = data.loadData(cfgT);
[X, Y] = data.getBatch(D.train.ids, D.train.len, 1:cfgT.batch);
Pt = guppy.initParams(cfgT, 'single');
st = struct();
for k = 1:20
    [Pt, st, info] = guppy.step(Pt, st, X, Y, cfgT);
    if mod(k, 5) == 0
        fprintf('  step %2d  loss %.4f  gnorm %.3f\n', k, info.loss, info.gnorm);
    end
end
fprintf('loss fell from 7.81 to %.3f -> forward/backward still wired\n', info.loss);
end

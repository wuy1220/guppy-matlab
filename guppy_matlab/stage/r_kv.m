function r_kv()
%R_KV Does the KV cache compute the same thing, and how much does it buy?
%   Nothing here is trusted until the cached path reproduces the stateless one
%   to floating-point noise, so every speed number sits next to an error number.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
out = fullfile(root, 'logs', 'kv_check.txt');
fid = fopen(out, 'w');

S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;

lines = readlines(fullfile(cfg.dataDir, 'prompts.txt'));
p = str2double(split(strtrim(lines(1)), ','))';
p = p(:);

% ---- 1. prefill: one shot over the prompt vs the stateless last row ----
[~, ~, lf] = guppy.forward(P, cfg, p, [], false, true);
[lk, KV]  = guppy.forwardKV(P, cfg, p, []);
e1 = max(abs(double(lf(end, :)) - double(lk(end, :))));
fprintf('1  prefill vs full forward        : max|diff| %.3e\n', e1);
fprintf(fid, 'prefill_maxdiff,%.3e\n', e1);

% ---- 2. twenty incremental steps, each checked against a full forward --
ids = p;
kk  = KV;
rng(3);
errs = zeros(20, 1);
for i = 1:20
    nxt = randi(cfg.V) - 1;
    ids(end+1, 1) = nxt;                                     %#ok<AGROW>
    [lk1, kk] = guppy.forwardKV(P, cfg, nxt, kk);
    [~, ~, lf1] = guppy.forward(P, cfg, ids, [], false, true);
    errs(i) = max(abs(double(lf1(end, :)) - double(lk1(end, :))));
end
fprintf('2  20 incremental steps           : max %.3e   median %.3e\n', ...
    max(errs), median(errs));
fprintf(fid, 'step_maxdiff_max,%.3e\nstep_maxdiff_median,%.3e\n', max(errs), median(errs));

% ---- 3. the sampler must produce the same stream from the same seed -----
rng(11); a = guppy.generate(P, cfg, p, 40, 0.7, 50);
rng(11); b = guppy.generateCached(P, cfg, p, 40, 0.7, 50);
same = isequal(a, b);
fprintf('3  identical token stream         : %d  (%d vs %d tokens)\n', ...
    same, numel(a), numel(b));
fprintf(fid, 'same_stream,%d\nn_tokens,%d,%d\n', same, numel(a), numel(b));

% ---- 4. wall clock, interleaved so a thermal dip cannot be blamed ------
tA = inf; tB = inf;
for r = 1:3
    t = tic; guppy.generate(P, cfg, p, 40, 0.7, 50);        tA = min(tA, toc(t));
    t = tic; guppy.generateCached(P, cfg, p, 40, 0.7, 50);  tB = min(tB, toc(t));
end
fprintf('4  40 tokens  stateless %.3f s   cached %.3f s   %.2fx\n', ...
    tA, tB, tA / tB);
fprintf('   per token %.2f ms -> %.2f ms\n', 1000*tA/40, 1000*tB/40);
fprintf(fid, 'gen40_stateless,%.3f\ngen40_cached,%.3f\nspeedup,%.2f\n', tA, tB, tA/tB);

% ---- 5. the training path shares this forward.m: prove it still works ---
cfg2 = guppy.defaultConfig('steps', 4, 'warmup', 1);
D = data.loadData(cfg2);
[X, Y] = data.getBatch(D.train.ids, D.train.len, 1:cfg2.batch);
P2 = guppy.initParams(cfg2, 'single');
l0 = guppy.forward(P2, cfg2, X, Y, true);
[~, C] = guppy.forward(P2, cfg2, X, Y, true);
G = guppy.backward(P2, C, cfg2);
gn = guppy.gradNorm(G);
fprintf('5  training path                  : loss %.4f  gnorm %.4f  finite %d\n', ...
    l0, gn, isfinite(l0) && isfinite(gn));
fprintf(fid, 'train_loss,%.4f\ntrain_gnorm,%.4f\n', l0, gn);

% ---- 6. every prompt in the set, end to end ----------------------------
t = tic;
n = 0;
for i = 1:numel(lines)
    s = strtrim(lines(i));
    if strlength(s) == 0, continue; end
    q = str2double(split(s, ','))';
    guppy.generateCached(P, cfg, q, 40, 0.7, 50);
    n = n + 1;
end
elAll = toc(t);
fprintf('6  %d prompts, cached              : %.2f s  (%.0f ms each)\n', ...
    n, elAll, 1000 * elAll / n);
fprintf(fid, 'all_prompts,%.2f\nn_prompts,%d\n', elAll, n);

fclose(fid);
fprintf('\n--- csv ---\n');
type(fullfile(root, 'logs', 'kv_check.txt'));
end

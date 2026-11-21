% R_PE  Is the model really using its 128-token context?
%
% The trainer builds batches with T = max(len)-1 and the corpus caps out at 59
% tokens, so positional rows 59..128 should never receive a gradient. This is a
% two-minute check: compare every positional vector against the distribution it
% would have had at initialisation, then look at the per-position eval loss and
% at what a 128-token generation actually does.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
S = load(fullfile(root, 'checkpoints', 'full3_final.mat'));
P = S.P; cfg = S.cfg;
d = cfg.d; Tmax = cfg.Tmax;

nrm  = sqrt(sum(double(P.pe).^2, 2)).';      % 1 x Tmax, row norms
mu   = 0.02^2 * d;                          % E||row||^2 at init (sd = 0.02)
sdv  = 0.02^2 * sqrt(2 * d);                % sd of ||row||^2 at init
zsc  = (nrm.^2 - mu) / sdv;                 % z-score against the init law
dead = abs(zsc) < 3;

fprintf('%-4s %-8s %-9s %s\n', 'pos', 'norm', 'z(init)', 'untouched');
for k = 1:Tmax
    if k <= 6 || mod(k, 6) == 0 || k >= 54
        fprintf('%3d  %.4f  %8.1f  %d\n', k, nrm(k), zsc(k), dead(k));
    end
end
fprintf('untouched rows: %d / %d\n', sum(dead), Tmax);
if any(dead)
    fprintf('untouched span : %d..%d\n', find(dead, 1), find(dead, 1, 'last'));
end

D = data.loadData(cfg);
TmaxTrain = max(D.train.len) - 1;
fprintf('train N=%d maxLen=%d -> T=%d ; eval N=%d maxLen=%d\n', ...
    size(D.train.ids, 1), max(D.train.len), TmaxTrain, ...
    size(D.eval.ids, 1), max(D.eval.len));
fprintf('padding share of a random batch: %.0f%% of positions carry no gradient\n', ...
    100 * (1 - mean(D.train.len) / mean(max(D.train.len))));

fid = fopen(fullfile(cfg.dataDir, 'pe_profile.csv'), 'w');
fprintf(fid, 'pos,norm,z,untouched\n');
for k = 1:Tmax
    fprintf(fid, '%d,%.6f,%.4f,%d\n', k, nrm(k), zsc(k), dead(k));
end
fclose(fid);

% --- per-position eval loss (only positions the corpus actually reaches) -----
nb = 20; per = zeros(Tmax, 1); cnt = zeros(Tmax, 1);
for b = 1:nb
    idx = (b-1) * cfg.batch + (1:cfg.batch);
    [X, Y] = data.getBatch(D.eval.ids, D.eval.len, idx);
    [~, ~, lg] = guppy.forward(P, cfg, X, [], false, false);
    Tb = size(X, 1);
    L2 = reshape(lg, [], cfg.V);
    y  = reshape(double(Y), [], 1);
    n  = size(L2, 1);
    m  = max(L2, [], 2);
    Z  = sum(exp(L2 - m), 2);
    sel = (1:n)' + y * n;
    ll  = L2(sel) - m - log(Z);
    ll(y == 0) = NaN;
    ll = reshape(ll, Tb, []);
    for t = 1:Tb
        v = ll(t, :); v = v(~isnan(v));
        per(t) = per(t) + sum(v); cnt(t) = cnt(t) + numel(v);
    end
end
pl = per ./ max(cnt, 1);
fprintf('\nper-position eval loss (n = docs long enough to reach it)\n');
for t = 1:TmaxTrain
    if t <= 4 || mod(t, 6) == 0 || t == TmaxTrain
        fprintf('pos %3d : %.4f  (n=%d)\n', t, pl(t), cnt(t));
    end
end
fid = fopen(fullfile(cfg.dataDir, 'pos_loss.csv'), 'w');
fprintf(fid, 'pos,loss,count\n');
for t = 1:TmaxTrain
    fprintf(fid, '%d,%.6f,%d\n', t, pl(t), cnt(t));
end
fclose(fid);

% --- what does a 128-token generation do with positions it never saw? --------
lines = readlines(fullfile(cfg.dataDir, 'prompts.txt'));
fid = fopen(fullfile(cfg.dataDir, 'gen_long_base.txt'), 'w');
rand('seed', 7); randn('seed', 7);
for i = 1:3
    s = strtrim(lines(i));
    ids = str2double(split(s, ','))';
    out = guppy.generate(P, cfg, ids, 128, 0.7, 50);
    fprintf(fid, '%s\n', strjoin(string(out(:)'), ','));
    fprintf('prompt %d: %d -> %d tokens\n', i, numel(ids), numel(out));
end
fclose(fid);
fprintf('wrote data/gen_long_base.txt, pe_profile.csv, pos_loss.csv\n');

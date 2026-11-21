function r_infer()
%R_INFER Where does inference time actually go? Numbers land in a CSV inside
%   the same job -- stdout from the agent loop is not reliable enough to
%   quote in a report.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);

S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;
Tmax = cfg.Tmax;

o = fopen(fullfile(root, 'data', 'infer_probe.csv'), 'w');
fprintf(o, 'what,T,ms\n');

info = whos('P');
fprintf('weights = %.2f MB (single)\n', info.bytes / 1e6);
fprintf(o, 'weights_MB,0,%.3f\n', info.bytes / 1e6);

% ---- 1. forward cost vs sequence length -------------------------------
fprintf('\n--- forward, B=1, single ---\n');
for T = [1 8 32 64 Tmax]
    ids = ones(T, 1, 'int32');
    guppy.forward(P, cfg, ids, [], false);           % warm up
    best = inf;
    for r = 1:5
        t = tic;
        for k = 1:8, guppy.forward(P, cfg, ids, [], false); end
        best = min(best, toc(t) / 8);
    end
    fprintf(o, 'fwd_T%d,1,%.4f\n', T, best * 1000);
    fprintf('T=%3d   %7.3f ms\n', T, best * 1000);
end

% ---- 2. the tied LM head: full T rows vs the last row only -----------
fprintf('\n--- lm head ---\n');
Xf = P.te(1:Tmax, :);                                 % [128 x 384]
t = tic; for k = 1:20, Lf = Xf * P.te.'; end;       eFull = toc(t) / 20;
t = tic; for k = 1:20, l1 = Xf(end, :) * P.te.'; end; eOne = toc(t) / 20;
fprintf('full [T x d]*[d x V] : %6.3f ms\n', eFull * 1000);
fprintf('last row only        : %6.3f ms\n', eOne * 1000);
fprintf(o, 'lmhead_full,128,%.4f\n', eFull * 1000);
fprintf(o, 'lmhead_one,1,%.4f\n', eOne * 1000);

% ---- 3. what generation actually costs today -------------------------
fprintf('\n--- generate, no KV cache ---\n');
ids = ones(30, 1, 'int32');
t = tic; out = guppy.generate(P, cfg, ids, 40, 0.7, 50); el = toc(t);
fprintf('40 new tokens: %.2f s   (%.2f ms / token, %d tokens emitted)\n', ...
    el, 1000 * el / 40, numel(out));
fprintf(o, 'gen40,40,%.2f\n', el * 1000);

% ---- 4. batching: same weight read, more rows ------------------------
fprintf('\n--- batched forward, T=128 ---\n');
for B = [1 4 8]
    ids = ones(Tmax, B, 'int32');
    guppy.forward(P, cfg, ids, [], false);
    t = tic;
    for k = 1:3, guppy.forward(P, cfg, ids, [], false); end
    el = toc(t) / 3;
    fprintf('B=%d  %8.2f ms   (%.2f ms per sequence, speedup %.1fx)\n', ...
        B, el * 1000, el * 1000 / B, 1); %#ok<NASGU>
    fprintf(o, 'fwd_B%d,128,%.3f\n', B, el * 1000);
end

fclose(o);
fprintf('\n=== CSV ===\n');
type(fullfile(root, 'data', 'infer_probe.csv'));
end

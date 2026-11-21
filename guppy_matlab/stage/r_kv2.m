function r_kv2()
%R_KV2 How much does the cache buy, as a function of how many tokens you ask
%   for? A fixed "speedup" number would be a lie: the stateless cost grows with
%   the prefix, the cached cost does not, so the ratio is a function of N.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
out = fullfile(root, 'logs', 'kv_curve.csv');
fid = fopen(out, 'w');
fprintf(fid, 'kind,N,ms_per_token\n');

S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;

lines = readlines(fullfile(cfg.dataDir, 'prompts.txt'));
base  = str2double(split(strtrim(lines(1)), ','))';
base  = base(:);
pre   = numel(base);
fprintf('prompt length = %d tokens, Tmax = %d\n\n', pre, cfg.Tmax);

% ---- A. stateless forward cost vs prefix length ------------------------
fprintf('--- stateless forward, B=1 ---\n');
fprintf('%-8s %10s %10s\n', 'T', 'full', 'last-row');
for T = [pre, 32, 64, 96, 128]
    ids = ones(T, 1, 'int32');
    guppy.forward(P, cfg, ids, [], false, false, false);          % warm
    t = tic; for k = 1:20, guppy.forward(P, cfg, ids, [], false, false, true);  end; eA = toc(t)/20;
    t = tic; for k = 1:20, guppy.forward(P, cfg, ids, [], false, true,  false); end; eB = toc(t)/20;
    fprintf('%-8d %8.2f   %8.2f  ms\n', T, eA*1000, eB*1000);
    fprintf(fid, 'fwd_full,%d,%.3f\nfwd_last,%d,%.3f\n', T, eA*1000, T, eB*1000);
end

% ---- B. one cached step, as the cache fills up ------------------------
fprintf('\n--- one cached step vs context already in the cache ---\n');
for L0 = [pre, 40, 80, 120]
    if L0 > cfg.Tmax - 20, continue; end
    [~, KV] = guppy.forwardKV(P, cfg, base(1:min(L0, pre)), []);
    if L0 > pre
        for k = 1:(L0 - pre), [~, KV] = guppy.forwardKV(P, cfg, 5, KV); end
    end
    t = tic;
    for k = 1:20, [~, KV] = guppy.forwardKV(P, cfg, 5, KV); end
    e = toc(t)/20;
    fprintf('len %3d -> %3d : %6.2f ms/step\n', L0, L0+20, e*1000);
    fprintf(fid, 'kv_step,%d,%.3f\n', L0, e*1000);
end

% ---- C. end-to-end: N generated tokens, identical ids, both paths -----
fprintf('\n--- end to end, prompt %d + N generated ---\n', pre);
fprintf('%-6s %14s %14s %9s\n', 'N', 'stateless', 'cached', 'speedup');
for N = [8 16 32 64 100]
    if pre + N > cfg.Tmax, continue; end

    ids = base;
    guppy.forward(P, cfg, ids, [], false, true, false);            % warm
    t = tic;
    for k = 1:N
        ids(end+1, 1) = 5;                                         %#ok<AGROW>
        guppy.forward(P, cfg, ids, [], false, true, false);
    end
    eA = toc(t) / N;

    t = tic;
    [~, KV] = guppy.forwardKV(P, cfg, base, []);
    for k = 1:N, [~, KV] = guppy.forwardKV(P, cfg, 5, KV); end
    eB = toc(t) / N;                                               % prefill amortised

    fprintf('%-6d %10.2f ms %10.2f ms %8.2fx\n', N, eA*1000, eB*1000, eA/eB);
    fprintf(fid, 'e2e_stateless,%d,%.3f\ne2e_cached,%d,%.3f\n', N, eA*1000, N, eB*1000);
end

% ---- D. what the forward.m rewrite alone bought ----------------------
fprintf('\n--- forward.m rewrite, T=1 (single token) ---\n');
ids1 = ones(1, 1, 'int32');
guppy.forward(P, cfg, ids1, [], false, false, false);
t = tic; for k = 1:100, guppy.forward(P, cfg, ids1, [], false, false, false); end; eOld = toc(t)/100;
t = tic; for k = 1:100, guppy.forward(P, cfg, ids1, [], false, true,  false); end; eNew = toc(t)/100;
fprintf('before  (cache + full head) : %6.2f ms\n', eOld*1000);
fprintf('after   (no cache, last row): %6.2f ms   (%.0f%% saved)\n', eNew*1000, 100*(1-eNew/eOld));
fprintf(fid, 'fwd_before,1,%.3f\nfwd_after,1,%.3f\n', eOld*1000, eNew*1000);

fclose(fid);
fprintf('\nwrote %s\n', out);
end

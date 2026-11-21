function r_kv3()
%R_KV3 Interleaved, min-of-rounds benchmark for the inference paths.
%   The first attempt averaged single runs, which let GC and background load
%   masquerade as a 20% effect, and its "before" arm accidentally passed the
%   new flags too. Every condition here is warmed, repeated, and rotated so a
%   thermal dip hits all arms equally.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
out = fullfile(root, 'logs', 'kv_curve2.csv');
fid = fopen(out, 'w');
fprintf(fid, 'kind,param,arm,ms\n');

S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;

WARM = 3; ROUNDS = 5; PER = 4;

% ---- A. the three forward.m modes, rotated inside each T ---------------
fprintf('--- one stateless forward, B=1 (min of %d rounds) ---\n', ROUNDS);
fprintf('%-6s %14s %14s %14s\n', 'T', 'old', 'no cache', 'no cache+last');
for T = [1 16 32 64 96 128]
    ids = ones(T, 1, 'int32');
    fns = { @() guppy.forward(P, cfg, ids, [], false, false, true), ...   % as before
            @() guppy.forward(P, cfg, ids, [], false, false, false), ...  % cache dropped
            @() guppy.forward(P, cfg, ids, [], false, true,  false) };    % + last row
    b = runArms(fns, WARM, ROUNDS, PER);
    fprintf('%-6d %11.2f ms %11.2f ms %11.2f ms   (%.0f%% then %.0f%%)\n', ...
        T, b(1)*1000, b(2)*1000, b(3)*1000, ...
        100*(1-b(2)/b(1)), 100*(1-b(3)/b(2)));
    fprintf(fid, 'fwd_old,%d,1,%.3f\nfwd_nocache,%d,2,%.3f\nfwd_last,%d,3,%.3f\n', ...
        T, b(1)*1000, T, b(2)*1000, T, b(3)*1000);
end

% ---- B. one cached step, at four cache fills ---------------------------
fprintf('\n--- one KV step (min of %d rounds) ---\n', ROUNDS);
pre = 10;
for L0 = [10 40 80 112]
    fns = {@() kvStepProbe(P, cfg, L0, pre)};
    b = runArms(fns, 2, ROUNDS, PER);
    fprintf('cache %3d tokens : %6.2f ms/step\n', L0, b(1)*1000);
    fprintf(fid, 'kv_step,%d,1,%.3f\n', L0, b(1)*1000);
end

% ---- C. end to end, both arms rotated --------------------------------
fprintf('\n--- prompt %d + N generated, arms rotated ---\n', pre);
fprintf('%-6s %14s %14s %10s\n', 'N', 'stateless', 'cached', 'speedup');
for N = [8 16 32 64 100]
    fns = { @() statelessRun(P, cfg, pre, N), @() cachedRun(P, cfg, pre, N) };
    b = runArms(fns, 2, ROUNDS, 2);
    eA = b(1) / N; eB = b(2) / N;
    fprintf('%-6d %11.2f ms %11.2f ms %9.2fx\n', N, eA*1000, eB*1000, eA/eB);
    fprintf(fid, 'e2e_stateless,%d,1,%.3f\ne2e_cached,%d,2,%.3f\n', ...
        N, eA*1000, N, eB*1000);
end

fclose(fid);
fprintf('\nwrote %s\n', out);
end

% ------------------------------------------------------------------------
function b = runArms(fns, nWarm, nRound, nPer)
%RUNARMS Warm every arm, then rotate rounds so drift is shared.
nc = numel(fns);
for i = 1:nc
    for w = 1:nWarm, fns{i}(); end
end
b = inf(1, nc);
for r = 1:nRound
    for i = 1:nc
        t = tic;
        for k = 1:nPer, fns{i}(); end
        b(i) = min(b(i), toc(t) / nPer);
    end
end
end

function el = statelessRun(P, cfg, pre, N)
ids = ones(pre, 1, 'int32');
guppy.forward(P, cfg, ids, [], false, true, false);
t = tic;
for k = 1:N
    ids(end+1, 1) = int32(5);
    guppy.forward(P, cfg, ids, [], false, true, false);
end
el = toc(t);
end

function el = cachedRun(P, cfg, pre, N)
ids = ones(pre, 1, 'int32');
t = tic;
[~, KV] = guppy.forwardKV(P, cfg, ids, []);
for k = 1:N
    [~, KV] = guppy.forwardKV(P, cfg, int32(5), KV);
end
el = toc(t);
end

function kvStepProbe(P, cfg, L0, pre)
%KVSTEPPROBE Start every probe from the same filled cache.
%   The base state is built once per L0 and then copied by value on each call;
%   MATLAB's copy-on-write makes that a 2.4 MB duplicate, ~1% of four steps.
persistent KV0 L0cached
if isempty(KV0) || L0cached ~= L0
    [~, KV0] = guppy.forwardKV(P, cfg, ones(pre, 1, 'int32'), []);
    for k = 1:(L0 - pre), [~, KV0] = guppy.forwardKV(P, cfg, int32(5), KV0); end
    L0cached = L0;
end
KV = KV0;
for k = 1:4, [~, KV] = guppy.forwardKV(P, cfg, int32(5), KV); end
end

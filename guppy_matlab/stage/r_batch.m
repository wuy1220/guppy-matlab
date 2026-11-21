% R_BATCH  Does batching the decoder move the roofline knee where theory says?
%
% Decoding is bandwidth bound: 32.27 MB of weights per step, arithmetic
% intensity 0.5 FLOP/byte against a measured machine balance point of
% 8.5 FLOP/byte. Batching B sequences raises the intensity to 0.5*B without
% touching the weight traffic, so tokens/s should keep rising until B~17 and
% flatten after that. This measures the curve and checks the batched path
% against the single-sequence one.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
S = load(fullfile(root, 'checkpoints', 'full3_final.mat'));
P = S.P; cfg = S.cfg;

lines = readlines(fullfile(cfg.dataDir, 'prompts.txt'));
p = str2double(split(strtrim(lines(4)), ','))';
fprintf('prompt %d tokens\n', numel(p));

% --- correctness: same numbers as the single-sequence path -------------------
n = numel(p);
[lg1, ~] = guppy.forwardKV(P, cfg, p, []);
[lgB, ~] = guppy.forwardBatch(P, cfg, repmat(p(:), 1, 4), []);
d1 = 0;
for b = 1:4
    blk = lgB((b-1)*n + (1:n), :);
    d1 = max(d1, max(abs(double(blk(:)) - double(lg1(:)))));
end
fprintf('prefill 4 seqs : max|batch - single| = %.2e\n', d1);

[~, kvA] = guppy.forwardKV(P, cfg, p, []);
[lg1b, ~] = guppy.forwardKV(P, cfg, 7, kvA);
[~, kvB] = guppy.forwardBatch(P, cfg, repmat(p(:), 1, 4), []);
lgBb = guppy.forwardBatch(P, cfg, repmat(7, 1, 4), kvB);
d2 = 0;
for b = 1:4
    d2 = max(d2, max(abs(double(lgBb(b, :)) - double(lg1b(:))')));
end
fprintf('after 1 step   : max|batch - single| = %.2e\n', d2);

% --- single-sequence baseline (KV cache, as in section 5.6) -------------------
N = 32;
tt = tic; guppy.generateCached(P, cfg, p, N, 0.7, 50); el1 = toc(tt);
fprintf('B= 1 (generateCached) %6.2f ms/step %6.0f tok/s\n', 1000*el1/N, N/el1);

Bs = [1 2 4 8 16 32 64];
rows = zeros(numel(Bs), 4);
for i = 1:numel(Bs)
    B = Bs(i);
    guppy.generateBatch(P, cfg, p, B, 4, 0.7, 50);          % warm-up
    rep = 3; best = inf;
    for q = 1:rep
        tt = tic; guppy.generateBatch(P, cfg, p, B, N, 0.7, 50); e = toc(tt);
        best = min(best, e);
    end
    rows(i, :) = [B, 1000*best/N, B*N/best, 0.5*B];
    fprintf('B=%2d  %6.2f ms/step  %6.0f tok/s  AI=%4.1f  speedup %.2fx\n', ...
        B, rows(i,2), rows(i,3), rows(i,4), rows(i,3)/(N/el1));
end

fid = fopen(fullfile(cfg.dataDir, 'batch_decode.csv'), 'w');
fprintf(fid, 'B,ms_per_step,tok_per_s,arith_intensity\n');
for i = 1:numel(Bs)
    fprintf(fid, '%d,%.3f,%.1f,%.1f\n', rows(i,1), rows(i,2), rows(i,3), rows(i,4));
end
fclose(fid);
fprintf('wrote data/batch_decode.csv\n');

% --- why the knee lands at 8 and not 17: GEMM throughput at small M ----------
% The roofline in figure 4 was measured at M = T*B ~ 1200. Decoding runs the
% same weights with M = B, and BLAS loses most of its peak there, so the
% compute side of the roofline is much lower than 509 GFLOPS.
Ms = [1 2 4 8 16 32 64 128 384 1264];
g = zeros(numel(Ms), 1);
for i = 1:numel(Ms)
    M = Ms(i);
    A = rand(M, 384, 'single');
    W = rand(384, 1152, 'single');
    C = A * W;
    rep = 20; tt = tic;
    for r = 1:rep, C = A * W; end   %#ok<NASGU>
    e = toc(tt) / rep;
    g(i) = 2 * M * 384 * 1152 / e / 1e9;
    fprintf('M=%4d   %6.1f GFLOPS\n', M, g(i));
end
fid = fopen(fullfile(cfg.dataDir, 'gemm_m.csv'), 'w');
fprintf(fid, 'M,gflops\n');
for i = 1:numel(Ms), fprintf(fid, '%d,%.1f\n', Ms(i), g(i)); end
fclose(fid);
fprintf('wrote data/gemm_m.csv\n');

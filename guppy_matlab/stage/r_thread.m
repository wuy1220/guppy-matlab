function r_thread()
%R_THREAD Does one training step get faster with fewer BLAS threads?
%   The step is half GEMM (compute bound, wants every core) and half
%   elementwise (bandwidth bound, extra threads only fight over memory).
%   On a 8P+8E part the logical cores are not equal either.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;
V = cfg.V; T = 56;

fprintf('numcores: %d   maxNumCompThreads: %d\n', feature('numcores'), maxNumCompThreads);

ids = reshape(int32(randi(V - 1, T * cfg.batch, 1)), T, cfg.batch);   % [T x B], 0-based
y   = reshape(int32(randi(V - 1, T * cfg.batch, 1)), T, cfg.batch);

% one fixed GEMM shape from the model, for a clean compute-bound reading
A = single(randn(T * cfg.batch, cfg.d));
Bm = single(randn(cfg.d, 3 * cfg.d));
flop = 2 * size(A, 1) * size(A, 2) * size(Bm, 2);

nc = feature('numcores');
ks = [1 2 4 6 8 10 12 14 16];
ks = ks(ks <= nc);
old = maxNumCompThreads;
res = zeros(numel(ks), 3);

for q = 1:numel(ks)
    k = ks(q);
    maxNumCompThreads(k);
    st = struct();
    [P, st, ~] = guppy.step(P, st, ids, y, cfg);      % warm pages / re-init pool
    [P, st, ~] = guppy.step(P, st, ids, y, cfg);
    n = 4; dt = zeros(n, 1);
    for r = 1:n
        tic; [P, st, ~] = guppy.step(P, st, ids, y, cfg); dt(r) = toc;
    end
    step = min(dt);

    g = zeros(5, 1);
    for r = 1:5
        tic; C = A * Bm; g(r) = toc;                        %#ok<NASGU>
    end
    gt = min(g);

    res(q, :) = [k, step, flop / gt / 1e9];
    fprintf('threads %2d   step %7.3f ms   gemm %6.1f GFLOPS\n', k, 1e3 * step, res(q, 3));
end

maxNumCompThreads(old);
fprintf('restored to %d\n', maxNumCompThreads);

fid = fopen('data\thr.csv', 'w');
fprintf(fid, 'threads,step_ms,gflops\n');
for q = 1:numel(ks)
    fprintf(fid, '%d,%.4f,%.2f\n', res(q, 1), 1e3 * res(q, 2), res(q, 3));
end
fclose(fid);
fprintf('wrote data/thr.csv\n');
end

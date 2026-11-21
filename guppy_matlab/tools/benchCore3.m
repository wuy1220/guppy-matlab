function S = benchCore3()
%BENCHCORE3 Calibrate the CPU: BLAS throughput, elementwise throughput, key ops.

S = struct();
S.version    = version;
S.computer   = computer;
S.numcores   = feature('numcores');
S.maxthreads = maxNumCompThreads;

fprintf('==== environment ====\n');
fprintf('MATLAB        : %s (%s)\n', S.version, S.computer);
fprintf('physical cores: %d   maxNumCompThreads: %d\n', S.numcores, S.maxthreads);

%% --- GEMM throughput: the single most important number -------------------
fprintf('\n==== GEMM (timeit) ====\n');
cases = { ...
    'head  4096x384x2418', 4096, 384, 2418; ...
    'head  4096x384x4096', 4096, 384, 4096; ...
    'qkv   1856x384x1152', 1856, 384, 1152; ...
    'ffn   1856x384x768' , 1856, 384, 768; ...
    'big   2048x2048x2048',2048, 2048, 2048 };

S.gemm = struct('name', {}, 'M', {}, 'K', {}, 'N', {}, 't_single', {}, ...
                't_double', {}, 'gflops_single', {}, 'gflops_double', {});
for i = 1:size(cases, 1)
    M = cases{i,2}; K = cases{i,3}; N = cases{i,4};
    As = rand(M, K, 'single'); Bs = rand(K, N, 'single');
    Ad = rand(M, K);           Bd = rand(K, N);
    ts = timeit(@() As * Bs);
    td = timeit(@() Ad * Bd);
    gs = 2*M*N*K/ts/1e9;  gd = 2*M*N*K/td/1e9;
    fprintf('%-21s single %7.2f ms %6.1f GF | double %7.2f ms %6.1f GF\n', ...
        cases{i,1}, ts*1e3, gs, td*1e3, gd);
    S.gemm(i).name = cases{i,1}; S.gemm(i).M = M; S.gemm(i).K = K; S.gemm(i).N = N;
    S.gemm(i).t_single = ts; S.gemm(i).t_double = td;
    S.gemm(i).gflops_single = gs; S.gemm(i).gflops_double = gd;
    clear As Bs Ad Bd
end

%% --- attention: pagemtimes vs explicit loop ------------------------------
fprintf('\n==== attention (T=64 hd=64, 192 pages) ====\n');
T = 64; hd = 64;
Q = rand(T, hd, 192, 'single'); Kk = rand(T, hd, 192, 'single');
tp = timeit(@() pagemtimes(Q, 'none', Kk, 'transpose'));
tl = timeit(@() loopmm(Q, Kk));
fprintf('pagemtimes : %7.2f ms\n', tp*1e3);
fprintf('explicit   : %7.2f ms   (loop/pagemtimes = %.2f)\n', tl*1e3, tl/tp);
S.attn_pagemtimes = tp; S.attn_loop = tl;

%% --- elementwise ---------------------------------------------------------
fprintf('\n==== elementwise (4096 x 2418 single, 39.6 MB) ====\n');
L = rand(4096, 2418, 'single');
te = timeit(@() exp(L));                    fprintf('exp           : %7.2f ms\n', te*1e3);
tx = timeit(@() max(L, [], 2));             fprintf('max dim2      : %7.2f ms\n', tx*1e3);
ty = timeit(@() sum(L, 2));                 fprintf('sum dim2      : %7.2f ms\n', ty*1e3);
tz = timeit(@() max(L, [], 1));             fprintf('max dim1      : %7.2f ms\n', tz*1e3);
tr = timeit(@() rand(4096, 384, 'single'));  fprintf('rand 4096x384 : %7.2f ms\n', tr*1e3);
S.exp = te; S.maxdim2 = tx; S.sumdim2 = ty; S.maxdim1 = tz; S.rand = tr;
clear L

%% --- embedding grad scatter ---------------------------------------------
fprintf('\n==== embedding scatter-add (N=4096, V=2418, d=384) ====\n');
V = 2418; d = 384; Nn = 4096;
subs = randi(V, Nn, 1);
vals = rand(Nn, d, 'single');
t2 = timeit(@() sparseScatter(subs, vals, V, Nn));
t3 = timeit(@() accVec(subs, vals, V, d));
fprintf('sparse(idx)*vals  : %7.2f ms\n', t2*1e3);
fprintf('accumarray vec idx: %7.2f ms\n', t3*1e3);
S.acc_sparse = t2; S.acc_vec = t3;

%% --- copying cost: how expensive is one 4M-element copy? -----------------
fprintf('\n==== memory traffic ====\n');
A = rand(4096, 1024, 'single');
tc = timeit(@() (A .* 1.0));
fprintf('4M-element multiply-copy : %6.2f ms  (%.1f GB/s)\n', tc*1e3, 4096*1024*4*2/tc/1e9);
S.copy4M = tc;
clear A

fprintf('\n==== misc ====\n');
fprintf('single * double scalar -> %s\n', class(single(1) * 2));
fprintf('\n');
end

function C = loopmm(Q, K)
C = zeros(size(Q,1), size(Q,1), size(Q,3), 'single');
for i = 1:size(Q, 3)
    C(:,:,i) = Q(:,:,i) * K(:,:,i).';
end
end

function G = sparseScatter(subs, vals, V, Nn)
Sm = sparse(subs, 1:Nn, 1, V, Nn);
G = Sm * vals;
end

function G = accVec(subs, vals, V, d)
idx = subs + V*(0:d-1);
G = accumarray(idx(:), vals(:), [V*d 1]);
G = reshape(G, V, d);
end

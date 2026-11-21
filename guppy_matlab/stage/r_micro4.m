function r_micro4()
%R_MICRO4 Is a per-token forward dominated by real work, or by call overhead?
%   Each candidate is timed in the exact shape the model uses at T=1.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
o = fopen(fullfile(root, 'data', 'micro4.csv'), 'w');
fprintf(o, 'op,n,ms\n');
N = 400;

% --- 1. pagemtimes on a one-token query --------------------------------
Qh = randn(1, 64, 6, 'single');
Kh = randn(1, 64, 6, 'single');
Vh = randn(1, 64, 6, 'single');
Pr = randn(1, 1, 6, 'single');
pagemtimes(Qh, 'none', Kh, 'transpose');              % warm
t = tic; for k = 1:N, S = pagemtimes(Qh, 'none', Kh, 'transpose'); end; e1 = toc(t)/N;
t = tic; for k = 1:N, Oh = pagemtimes(Pr, 'none', Vh, 'none'); end;      e2 = toc(t)/N;

% --- 2. the same math, done as one flat matrix product -----------------
Qf = randn(1, 384, 'single'); Kf = reshape(Kh, 1, 384); Vf = reshape(Vh, 1, 384);
t = tic; for k = 1:N, S2 = Qf * Kf.'; end;            e3 = toc(t)/N;

% --- 3. bookkeeping the loop does every layer --------------------------
t = tic
for k = 1:N
    s = struct('A',1,'b',2,'c',3,'d',4,'e',5,'f',6,'g',7,'h',8,'i',9, ...
               'j',10,'k',11,'l',12,'m',13,'n',14,'o',15,'p',16,'q',17,'r',18);
end
e4 = toc(t)/N;

W = cell(1,6);
for l = 1:6, W{l} = randn(384, 1152, 'single'); end
t = tic; for k = 1:N, for l = 1:6, x = W{l}; end; end; e5 = toc(t)/N;

% --- 4. matrix-vector products at the shapes the model uses ------------
x  = randn(1, 384, 'single');
Wa = randn(384, 1152, 'single');   % QKV, 1.77 MB
Wb = randn(384,  384, 'single');   % Wo,  0.59 MB
Wc = randn(384,  768, 'single');   % W1/W2, 1.18 MB
Wd = randn(384, 2418, 'single');   % tied head, 3.71 MB
t = tic; for k = 1:N, y = x * Wa; end; e6 = toc(t)/N;
t = tic; for k = 1:N, y = x * Wb; end; e7 = toc(t)/N;
t = tic; for k = 1:N, y = x * Wc; end; e8 = toc(t)/N;
t = tic; for k = 1:N, y = x * Wd; end; e9 = toc(t)/N;

% measured bytes per call -> effective bandwidth
bw = @(bytes, e) bytes / e / 1e9;
fprintf('\n--- page GEMMs (one token) ---\n');
fprintf('pagemtimes Q*K^T  [1x64x6] : %7.1f us\n', e1*1e6);
fprintf('pagemtimes P*V    [1x1x6]  : %7.1f us\n', e2*1e6);
fprintf('same as flat matmul        : %7.1f us   (%.1fx)\n', e3*1e6, e1/e3);
fprintf('\n--- per-layer bookkeeping ---\n');
fprintf('20-field struct construct  : %7.1f us\n', e4*1e6);
fprintf('6 cell indexings           : %7.1f us\n', e5*1e6);
fprintf('\n--- matvec, effective bandwidth ---\n');
fprintf('1x384 x 384x1152  (1.77MB) : %7.1f us  -> %5.1f GB/s\n', e6*1e6, bw(384*1152*4, e6));
fprintf('1x384 x 384x384   (0.59MB) : %7.1f us  -> %5.1f GB/s\n', e7*1e6, bw(384*384*4, e7));
fprintf('1x384 x 384x768   (1.18MB) : %7.1f us  -> %5.1f GB/s\n', e8*1e6, bw(384*768*4, e8));
fprintf('1x384 x 384x2418  (3.71MB) : %7.1f us  -> %5.1f GB/s\n', e9*1e6, bw(384*2418*4, e9));

% --- 5. what one T=1 forward "should" cost from these parts ------------
gemm = 6*(e6 + e7 + 2*e8) + e9;      % per layer: QKV, Wo, W1, W2; plus head
page = 6*2*e1;                        % both page GEMMs, xN layers (upper bound)
book = 6*(e4 + e5);
fprintf('\n--- budget for one T=1 forward (ms) ---\n');
fprintf('weights (all matvec)  %6.2f\n', gemm*1000);
fprintf('attention page GEMMs  %6.2f\n', page*1000);
fprintf('struct + cell index   %6.2f\n', book*1000);
fprintf('sum of parts          %6.2f\n', (gemm+page+book)*1000);
fprintf('measured earlier      %6.2f\n', 3.5566);

fprintf(o, 'pagemtimes_QK,6,%.4f\npagemtimes_PV,6,%.4f\nflat_matmul,1,%.4f\nstruct20,1,%.4f\ncellidx6,1,%.4f\nmv_1152,1,%.4f\nmv_384,1,%.4f\nmv_768,1,%.4f\nmv_2418,1,%.4f\nbudget,0,%.3f\n', ...
    e1*1000, e2*1000, e3*1000, e4*1000, e5*1000, e6*1000, e7*1000, e8*1000, e9*1000, (gemm+page+book)*1000);
fclose(o);
end

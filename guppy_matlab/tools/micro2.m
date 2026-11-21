function micro2()
%MICRO2 Candidate micro-optimisations, measured in isolation.
%   Every row is a drop-in replacement for something the profiler flagged as
%   non-GEMM overhead. Only the winners get promoted into the package.

T = 44; B = 32; N = T*B; d = 384; V = 2418; p = 0.1;

fprintf('\n==== A. dropout mask generation ([%d x %d] single) ====\n', N, d);
X = zeros(N, d, 'single');
t1 = timeit(@() rand(size(X), 'like', X));
t2 = timeit(@() rand(N, d, 'single'));
t3 = timeit(@() rand(N, d));
t4 = timeit(@() rand(N*d, 1, 'single'));
tm1 = timeit(@() rand(N, d*13, 'single'));
fprintf('rand(size(X),''like'',X)   %6.2f ms\n', t1*1e3);
fprintf('rand(N,d,''single'')        %6.2f ms\n', t2*1e3);
fprintf('rand(N,d) double           %6.2f ms\n', t3*1e3);
fprintf('rand(N*d,1,''single'')      %6.2f ms\n', t4*1e3);
fprintf('13 masks in one call       %6.2f ms  (%.2f ms/mask)\n', tm1*1e3, tm1*1e3/13);
fprintf('=> per-mask ns/element: %.2f\n', t2*1e9/(N*d));

fprintf('\n==== B. applying the mask ====\n');
keep = rand(N, d, 'single') >= p;
ks   = single(keep) / (1-p);
t1 = timeit(@() gemul(X, keep, 1/(1-p)));
t2 = timeit(@() gemul(X, ks, 1));
fprintf('logical mask + scale       %6.2f ms\n', t1*1e3);
fprintf('single mask, no scale      %6.2f ms\n', t2*1e3);

fprintf('\n==== C. AdamW on the real parameter budget (8.06M singles) ====\n');
np = 8.06e6;
pp = rand(np, 1, 'single'); gg = rand(np, 1, 'single')*1e-3;
mm = zeros(np, 1, 'single'); vv = zeros(np, 1, 'single');
b1 = 0.9; b2 = 0.95; lr = 3e-4; wd = 0.1; epsA = 1e-8; bc1 = 0.1; bc2 = 0.001;
t1 = timeit(@() adamOne(pp, gg, mm, vv, lr, wd, b1, b2, bc1, bc2, epsA));
t2 = timeit(@() adamFused(pp, gg, mm, vv, lr, wd, b1, b2, bc1, bc2, epsA));
fprintf('current (5 passes)         %6.2f ms\n', t1*1e3);
fprintf('fused   (4 passes)         %6.2f ms  (%.2fx)\n', t2*1e3, t1/t2);

fprintf('\n==== D. gradient norm over the same budget ====\n');
t1 = timeit(@() sqrt(sum(gg.^2, 'all')));
t2 = timeit(@() norm(gg, 'fro'));
t3 = timeit(@() sum(abs(gg), 'all'));
fprintf('sum(g.^2,''all'')           %6.2f ms\n', t1*1e3);
fprintf('norm(g,''fro'')              %6.2f ms  (%.2fx)\n', t2*1e3, t1/t2);
fprintf('sum(abs(g))  (L1, wrong)   %6.2f ms\n', t3*1e3);

fprintf('\n==== E. embedding scatter  (V=%d, N=%d, d=%d) ====\n', V, N, d);
idx = randi([1 V], N, 1);
dXe = rand(N, d, 'single');
t1 = timeit(@() sparse(idx, 1:N, 1, V, N) * dXe);
t2 = timeit(@() scatterCumsum(idx, dXe, V));
t3 = timeit(@() scatterAccum(idx, dXe, V));
fprintf('sparse(V,N)*dXe            %6.2f ms\n', t1*1e3);
fprintf('sort + cumsum              %6.2f ms  (%.2fx)\n', t2*1e3, t1/t2);
fprintf('accumarray per column      %6.2f ms  (%.2fx)\n', t3*1e3, t1/t3);
ref = full(sparse(idx, 1:N, 1, V, N) * dXe);
fprintf('max |ref - cumsum| = %.2e   |ref - accumarray| = %.2e\n', ...
    max(abs(ref - scatterCumsum(idx, dXe, V)), [], 'all'), ...
    max(abs(ref - scatterAccum(idx, dXe, V)), [], 'all'));

fprintf('\n==== F. softmax + CE  (%d x %d) ====\n', N, V);
Lg = rand(N, V, 'single') * 4;
tg = randi([0 V-1], N, 1);
tg(1:3) = 0;   % a few ignored targets
t1 = timeit(@() guppy.softmaxCE(Lg, double(tg), 0));
t2 = timeit(@() ceFast(Lg, double(tg), 0));
[l1, d1] = guppy.softmaxCE(Lg, double(tg), 0);
[l2, d2] = ceFast(Lg, double(tg), 0);
fprintf('current                    %6.2f ms\n', t1*1e3);
fprintf('fast (no full-size logp)   %6.2f ms  (%.2fx)\n', t2*1e3, t1/t2);
fprintf('loss diff %.3e   grad diff %.3e\n', abs(l1-l2), max(abs(d1-d2), [], 'all'));

fprintf('\n==== G. causal mask construction (T=%d) ====\n', T);
t1 = timeit(@() reshape(cast(triu(ones(T,T),1), 'single') * single(-1e9), [T T 1]));
fprintf('rebuild every step         %6.2f ms  (x13 sites? no: x1 per forward)\n', t1*1e3);

fprintf('\n==== H. exp / elementwise baseline ====\n');
E = rand(N, V, 'single');
t1 = timeit(@() exp(E));
t2 = timeit(@() E .* 2);
fprintf('exp   on %d elems         %6.2f ms  (%.2f ns/elem)\n', numel(E), t1*1e3, t1*1e9/numel(E));
fprintf('mul   on %d elems         %6.2f ms  (%.2f ns/elem)\n', numel(E), t2*1e3, t2*1e9/numel(E));
end

% ---------------------------------------------------------------- helpers
function Y = gemul(X, keep, s)
Y = X .* keep;
Y = Y * s;
end

function [p, m, v] = adamOne(p, g, m, v, lr, wd, b1, b2, bc1, bc2, epsA)
m = b1 .* m + (1 - b1) .* g;
v = b2 .* v + (1 - b2) .* (g .* g);
p = p - (lr * wd) .* p;
p = p - (lr / bc1) .* m ./ (sqrt(v ./ bc2) + epsA);
end

function [p, m, v] = adamFused(p, g, m, v, lr, wd, b1, b2, bc1, bc2, epsA)
%  m and v updates are unavoidable; the two p updates merge into one pass and
%  the 1/bc2 divide folds into a single scalar multiply on sqrt(v).
m = b1 .* m + (1 - b1) .* g;
v = b2 .* v + (1 - b2) .* (g .* g);
p = p .* (1 - lr*wd) - (lr / bc1) * m ./ (sqrt(v) * sqrt(1/bc2) + epsA);
end

function G = scatterCumsum(idx, dXe, V)
% Sort so equal indices are contiguous, then prefix-sum and difference at the
% group boundaries. Pure BLAS/gather, no sparse assembly.
[s, ord] = sort(idx);
Xs  = dXe(ord, :);
cs  = cumsum(Xs, 1);
last = [find(diff(s)); numel(s)];
vals = cs(last, :);
prev = zeros(size(vals), 'like', vals);
prev(2:end, :) = cs(last(1:end-1), :);
G = zeros(V, size(dXe, 2), 'like', dXe);
G(s(last), :) = vals - prev;
end

function G = scatterAccum(idx, dXe, V)
d = size(dXe, 2);
G = zeros(V, d, 'like', dXe);
for j = 1:d
    G(:, j) = accumarray(idx, dXe(:, j), [V 1]);
end
end

function [loss, dlogits, nValid] = ceFast(logits, tgt, ignoreIdx)
N = size(logits, 1);
m  = max(logits, [], 2);
E  = exp(logits - m);
Z  = sum(E, 2);

valid  = tgt ~= ignoreIdx;
nValid = sum(valid);
lin    = (1:N)' + double(tgt) * N;
sel    = lin(valid);
rows   = find(valid);
loss   = -sum(logits(sel) - m(rows) - log(Z(rows))) / nValid;

s = 1 / nValid;
dlogits = E ./ (Z / s);              % == (E./Z) * s, one pass instead of two
dlogits(sel) = dlogits(sel) - s;
dlogits(~valid, :) = 0;
end

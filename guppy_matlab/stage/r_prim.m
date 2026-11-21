function r_prim()
%R_PRIM Throughput of the elementwise primitives the kernels are built from.
%   Same array, same machine: the point is that the spread between two
%   spellings of the same math is larger than any algorithmic difference
%   we have been chasing.

d = 384; dff = 768; N = 1792;
A = single(randn(N, dff)) * 0.5;
B = single(randn(N, dff)) * 0.5;
P = single(randn(N, dff));
n = numel(A); bytes2 = 2 * 4 * n;   % one read + one write of the whole array

runs = 5;
t = @(f) timeop(f, runs);

fprintf('array %d x %d single  = %.2f MB\n', N, dff, 4 * n / 2^20);
fprintf('%-26s %8s %8s %8s\n', 'expression', 'ms', 'ns/el', 'GB/s');
fprintf('%s\n', repmat('-', 1, 54));

T = {};
T(end+1, :) = {'A + B',               t(@() A + B)};
T(end+1, :) = {'A + 1',               t(@() A + 1)};
T(end+1, :) = {'A .* B',              t(@() A .* B)};
T(end+1, :) = {'A .* 1.0',            t(@() A .* 1.0)};
T(end+1, :) = {'A ./ B',              t(@() A ./ B)};
T(end+1, :) = {'A ./ 1.0',            t(@() A ./ 1.0)};
T(end+1, :) = {'1.0 ./ A',            t(@() 1.0 ./ A)};
T(end+1, :) = {'sqrt(A)',             t(@() sqrt(A))};
T(end+1, :) = {'exp(A)',              t(@() exp(A))};
T(end+1, :) = {'abs(A)',              t(@() abs(A))};
T(end+1, :) = {'A.^2',                t(@() A .^ 2)};
T(end+1, :) = {'max(A,0)',            t(@() max(A, 0))};
T(end+1, :) = {'max(A,single(0))',    t(@() max(A, single(0)))};
T(end+1, :) = {'(A+abs(A))*0.5',      t(@() (A + abs(A)) * 0.5)};
T(end+1, :) = {'A .* (A > 0)',        t(@() A .* (A > 0))};
T(end+1, :) = {'A > 0',               t(@() A > 0)};

for k = 1:size(T, 1)
    h = T{k, 2};
    fprintf('%-26s %8.3f %8.3f %8.1f\n', T{k, 1}, 1e3 * h, 1e9 * h / n, bytes2 / h / 1e9);
end

fprintf('\n--- row reductions (dim 2, the strided one) ---\n');
onesd = single(ones(dff, 1)) / dff;
T2 = {};
T2(end+1, :) = {'mean(A,2)',        t(@() mean(A, 2))};
T2(end+1, :) = {'sum(A,2)/dff',     t(@() sum(A, 2) / dff)};
T2(end+1, :) = {'A*ones/dff (gemv)', t(@() A * onesd)};
T2(end+1, :) = {'max(A,[],2)',      t(@() max(A, [], 2))};
T2(end+1, :) = {'A*ones (gemv)',    t(@() A * single(ones(dff, 1)))};

fprintf('%-26s %8s\n', 'expression', 'ms');
fprintf('%s\n', repmat('-', 1, 36));
for k = 1:size(T2, 1)
    fprintf('%-26s %8.3f\n', T2{k, 1}, 1e3 * T2{k, 2});
end

fprintf('\n--- LayerNorm forward: current vs candidates ---\n');
X = single(randn(N, d)) * 0.7 + 0.2;
g = single(randn(1, d)) * 0.1 + 1;
b = single(zeros(1, d));
epsLN = single(1e-5);

[Y0, xc0, rs0] = lnRef(X, g, b, epsLN);
f1 = @() lnRef(X, g, b, epsLN);
f2 = @() lnGemv(X, g, b, epsLN);
[Y2, xc2, rs2] = lnGemv(X, g, b, epsLN);
fprintf('ref   %7.3f ms   |Y-Yref|max %.2e  |rstd-rs|max %.2e\n', ...
    1e3 * t(f1), max(abs(Y2(:) - Y0(:))), max(abs(rs2(:) - rs0(:))));
fprintf('gemv  %7.3f ms\n', 1e3 * t(f2));

fprintf('\n--- LayerNorm backward: current vs candidate ---\n');
dY = single(randn(N, d));
[dx0, dg0, db0] = lnBwdRef(dY, xc0, rs0, g);
[dx2, dg2, db2] = lnBwdGemv(dY, xc0, rs0, g);
fprintf('ref   %7.3f ms\n', 1e3 * t(@() lnBwdRef(dY, xc0, rs0, g)));
fprintf('gemv  %7.3f ms\n', 1e3 * t(@() lnBwdGemv(dY, xc0, rs0, g)));
fprintf('  max|dX| %.2e  max|dg| %.2e  max|db| %.2e\n', ...
    max(abs(dx2(:) - dx0(:))), max(abs(dg2(:) - dg0(:))), max(abs(db2(:) - db0(:))));
end

% ---------------------------------------------------------------- helpers
function h = timeop(f, runs)
f();                       % warm up page cache / JIT
h = inf;
for k = 1:runs
    tic; f(); e = toc;
    if e < h, h = e; end
end
end

function [Y, xc, rstd] = lnRef(X, g, b, epsLN)
mu   = mean(X, 2);
xc   = X - mu;
v    = mean(xc .* xc, 2);
rstd = 1 ./ sqrt(v + epsLN);
Y    = xc .* rstd .* g + b;
end

function [Y, xc, rstd] = lnGemv(X, g, b, epsLN)
d    = size(X, 2);
w    = single(ones(d, 1)) / single(d);      % BLAS does the reduction
mu   = X * w;
xc   = X - mu;
v    = (xc .* xc) * w;
rstd = 1 ./ sqrt(v + epsLN);
Y    = (xc .* rstd) .* g + b;
end

function [dX, dg, db] = lnBwdRef(dY, xc, rstd, g)
dxh = dY .* g;
m1  = mean(dxh, 2);
m2  = mean(dxh .* xc, 2);
dX  = rstd .* (dxh - m1 - xc .* (rstd .* rstd) .* m2);
dg  = sum(dY .* xc .* rstd, 1);
db  = sum(dY, 1);
end

function [dX, dg, db] = lnBwdGemv(dY, xc, rstd, g)
d    = size(dY, 2);
w    = single(ones(d, 1)) / single(d);
dxh  = dY .* g;
m1   = dxh * w;
m2   = (dxh .* xc) * w;
rs2  = rstd .* rstd;
dX   = rstd .* (dxh - m1 - (xc .* m2) .* rs2);
dg   = sum(dY .* xc .* rstd, 1);
db   = sum(dY, 1);
end

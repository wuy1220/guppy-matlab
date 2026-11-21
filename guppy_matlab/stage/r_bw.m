function r_bw()
%R_BW Primitive throughput at full thread count, and real STREAM bandwidth.
%   The roofline number and the decode floor both rest on an assumed
%   bandwidth.  Measure it instead of assuming it, and leave the numbers
%   in data/prim.csv + data/stream.csv so the appendix table is reproducible.

maxNumCompThreads(8);
fprintf('numcores %d, threads %d\n\n', feature('numcores'), maxNumCompThreads);

N = 1792; dff = 768;
A = single(randn(N, dff)) * 0.5;
B = single(randn(N, dff)) * 0.5;
n = numel(A);
fprintf('array %d x %d single = %.2f MB\n', N, dff, 4 * n / 2^20);

names = {'A + B', 'A .* B', 'A.^2', 'abs(A)', 'max(A,0)', 'exp(A)', 'sqrt(A)', '(A+abs(A))*0.5'};
funs  = {@() plus(A, B), @() times(A, B), @() power(A, 2), @() abs(A), ...
         @() max(A, 0), @() exp(A), @() sqrt(A), @() times(plus(A, abs(A)), 0.5)};
traffic = [3 3 2 2 2 2 2 4];           % full-array passes
ms = zeros(numel(funs), 1);
for k = 1:numel(funs)
    ms(k) = 1e3 * minop(funs{k}, 7);
    fprintf('%-16s %7.3f ms  %6.3f ns/el  %6.1f GB/s\n', names{k}, ms(k), ...
        1e6 * ms(k) / n, traffic(k) * 4 * n / (ms(k) / 1e3) / 1e9);
end
fid = fopen(fullfile('data', 'prim.csv'), 'w');
fprintf(fid, 'name,ms,ns_per_el,gbs\n');
for k = 1:numel(funs)
    fprintf(fid, '"%s",%.4f,%.4f,%.2f\n', names{k}, ms(k), 1e6 * ms(k) / n, ...
        traffic(k) * 4 * n / (ms(k) / 1e3) / 1e9);
end
fclose(fid);

mbs = [1 4 16 64 256];
S = zeros(numel(mbs), 5);
fprintf('\nSTREAM  MB/array   copy   scale     add   triad\n');
for q = 1:numel(mbs)
    m = mbs(q) * 2^20 / 4;
    v1 = single(randn(m, 1)); v2 = single(randn(m, 1)); v3 = zeros(m, 1, 'single');
    f = 4 * m / 1e9;
    S(q, :) = [mbs(q), 2 * f / minop(@() copyto(v3, v1), 5), ...
        2 * f / minop(@() scaleinto(v3, v1, single(1.5)), 5), ...
        3 * f / minop(@() addinto(v3, v1, v2), 5), ...
        4 * f / minop(@() triad(v3, v1, v2, single(1.5)), 5)];
    fprintf('%14d %6.1f %7.1f %7.1f %7.1f\n', S(q, :));
end
fid = fopen(fullfile('data', 'stream.csv'), 'w');
fprintf(fid, 'mb,copy,scale,add,triad\n');
for q = 1:numel(mbs)
    fprintf(fid, '%d,%.2f,%.2f,%.2f,%.2f\n', S(q, :));
end
fclose(fid);

maxNumCompThreads('automatic');
fprintf('\nthreads restored to %d; wrote data/prim.csv, data/stream.csv\n', maxNumCompThreads);
end

function h = minop(f, runs)
f();
h = inf;
for k = 1:runs
    tic; f(); e = toc;
    if e < h, h = e; end
end
end

function dst = copyto(dst, src), dst(:) = src; end
function dst = scaleinto(dst, src, s), dst(:) = s * src; end
function dst = addinto(dst, a, b), dst(:) = a + b; end
function dst = triad(dst, a, b, s), dst(:) = a + s * b; end

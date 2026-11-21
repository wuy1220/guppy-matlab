function r_micro3()
%R_MICRO3 Per-call cost of the pieces that a per-token forward executes ~90 times.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
o = fopen(fullfile(root, 'data', 'micro3.csv'), 'w');
fprintf(o, 'op,n,ms\n');

V = 2418;
l = randn(1, V, 'single');
g = ones(1, 384, 'single'); b = zeros(1, 384, 'single');
X1 = randn(1, 384, 'single');
X2 = randn(128, 384, 'single');

N = 300;
maxk(l, 50); exp(l);                                  % warm
t = tic; for k = 1:N, thr = maxk(l, 50); end;        e1 = toc(t) / N;
t = tic; for k = 1:N, p = exp(l - max(l)); p = p / sum(p); kk = find(cumsum(p) >= rand(), 1); end; e2 = toc(t) / N;
t = tic; for k = 1:N, guppy.layerNormFwd(X1, g, b, 1e-5); end; e3 = toc(t) / N;
t = tic; for k = 1:N, guppy.layerNormFwd(X2, g, b, 1e-5); end; e4 = toc(t) / N;
t = tic; for k = 1:N, y = X1 + X1; end;              e5 = toc(t) / N;
t = tic; for k = 1:N, y = max(X2, 0); end;           e6 = toc(t) / N;

fprintf('maxk(50) of 2418      : %6.1f us\n', e1 * 1e6);
fprintf('softmax+inverse-cdf   : %6.1f us\n', e2 * 1e6);
fprintf('layerNorm [1 x 384]   : %6.1f us\n', e3 * 1e6);
fprintf('layerNorm [128 x 384] : %6.1f us\n', e4 * 1e6);
fprintf('X + Y  [1 x 384]      : %6.1f us\n', e5 * 1e6);
fprintf('max(X,0) [128 x 384]  : %6.1f us\n', e6 * 1e6);

fprintf(o, 'maxk50,2418,%.4f\nsoftmaxcdf,2418,%.4f\nln_1x384,384,%.4f\nln_128x384,384,%.4f\nadd_1x384,384,%.4f\nrelu_128x384,384,%.4f\n', ...
    e1*1000, e2*1000, e3*1000, e4*1000, e5*1000, e6*1000);
fclose(o);
type(fullfile(root, 'data', 'micro3.csv'));
end

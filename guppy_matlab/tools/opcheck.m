function opcheck()
%OPCHECK Operator-level gradient checks.
%   A whole-network finite difference is *not* a valid test here: perturbing
%   tens of thousands of weights at once flips ReLU signs, so the numerical
%   derivative is dominated by the kinks rather than by the gradient. Each
%   operator below is smooth on its own, so central differences are exact to
%   ~1e-8 and any wiring error shows up immediately.

fprintf('==== LayerNorm ====\n');
rng(1);
N = 6; d = 8;
x = randn(N,d); g = randn(1,d); b = randn(1,d); Wt = randn(N,d);
[Y, xc, rstd] = guppy.layerNormFwd(x, g, b, 1e-5);
[dX, dg, db] = guppy.layerNormBwd(Wt, xc, rstd, g);
obj = @(x,g,b) sum(Wt(:) .* reshape(guppy.layerNormFwd(x,g,b,1e-5), [], 1));
fprintf('dx  rel %.2e | dg rel %.2e | db rel %.2e\n', ...
    fd3(@(t) obj(t,g,b), x, dX), fd3(@(t) obj(x,t,b), g, dg), fd3(@(t) obj(x,g,t), b, db));

fprintf('\n==== attention (with causal mask) ====\n');
rng(2);
T = 5; B = 2; H = 2; hd = 4;
Q = randn(T*B, H*hd); K = randn(T*B, H*hd); Vv = randn(T*B, H*hd);
maskV = reshape(triu(ones(T,T),1) * -1e9, [T T 1]);
[O, Qh, Kh, Vh, Pr] = guppy.attentionFwd(Q,K,Vv,T,B,H,hd,maskV);
Wt = randn(size(O));
[dQ,dK,dV] = guppy.attentionBwd(Wt, Qh, Kh, Vh, Pr, T, B, H, hd);
obj = @(q,k,v) sum(Wt(:) .* reshape(guppy.attentionFwd(q,k,v,T,B,H,hd,maskV), [], 1));
fprintf('dQ rel %.2e | dK rel %.2e | dV rel %.2e\n', ...
    fd3(@(t) obj(t,K,Vv), Q, dQ), fd3(@(t) obj(Q,t,Vv), K, dK), fd3(@(t) obj(Q,K,t), Vv, dV));

fprintf('\n==== softmax cross-entropy ====\n');
rng(3);
L0 = randn(12, 7); tgt = randi([0 6], 12, 1); tgt(3) = 0; tgt(9) = 0;
[~, dL] = guppy.softmaxCE(L0, tgt, 0);
obj = @(t) guppy.softmaxCE(t, tgt, 0);
fprintf('dlogits rel %.2e\n', fd3(obj, L0, dL));

fprintf('\n==== single-batch overfit (whole training loop) ====\n');
cfg = guppy.defaultConfig('L', 2, 'd', 32, 'H', 2, 'dff', 64, 'Tmax', 16, ...
                          'V', 2418, 'batch', 4, 'dropout', 0, 'steps', 200, 'warmup', 20);
P = guppy.initParams(cfg);
st = struct();
Tt = 12; Bb = 4;
ids = repmat(reshape(1:48, 12, 4), 1, 1);
y   = circshift(ids, -1, 1);
y(end,:) = 0;
fprintf('step 0 loss %.4f\n', guppy.forward(P, cfg, ids, y, false));
for k = 1:200
    [P, st, inf1] = guppy.step(P, st, ids, y, cfg);
    if any(k == [1 5 10 25 50 100 150 200])
        fprintf('step %3d loss %.4f  |g| %.3f\n', k, inf1.loss, inf1.gnorm);
    end
end
end

function rel = fd3(fun, X, dX)
%FD3 Worst relative error between central differences and the analytic grad.
h = 1e-6;
idx = randperm(numel(X), min(12, numel(X)));
worst = 0;
for t = 1:numel(idx)
    i = idx(t);
    Xp = X; Xp(i) = Xp(i) + h;
    Xm = X; Xm(i) = Xm(i) - h;
    num = (fun(Xp) - fun(Xm)) / (2*h);
    ana = dX(i);
    rel = abs(num - ana) / max(1e-14, abs(num) + abs(ana));
    worst = max(worst, rel);
end
rel = worst;
end

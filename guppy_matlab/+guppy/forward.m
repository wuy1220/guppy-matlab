function [loss, C, logits] = forward(P, cfg, ids, y, training, onlyLast, keepCache)
%FORWARD GuppyLM forward pass, hand-written, no toolboxes.
%   ids, y : [T x B] integer token ids, time-major inside each column
%   Activations are [T*B x d] with row n = (b-1)*T + t, which is exactly the
%   column-major order of ids, so ids(:) can index the embedding table directly.
%
%   onlyLast  : run the final LayerNorm and the tied head on the last position
%               only. Sampling reads nothing else, and at T=128 this skips
%               127/128 of a [128 x 384] * [384 x 2418] product.
%   keepCache : keep the per-layer activations that backward consumes.
%               Defaults to `training`. Building them during inference used to
%               cost 16% of a single-token forward, measured line by line: the
%               18-field structs plus the `clear` calls that followed them.

if nargin < 5 || isempty(training),  training  = false; end
if nargin < 6 || isempty(onlyLast),  onlyLast  = false; end
if nargin < 7 || isempty(keepCache), keepCache = training; end

T = size(ids, 1); B = size(ids, 2); N = T * B;
d = cfg.d; H = cfg.H; hd = cfg.hd; L = cfg.L;
dtype = class(P.te);
p = cfg.dropout;
ft = guppy.flags(cfg);

idx = double(ids(:)) + 1;                       % 1-based, [N 1]
pos = repmat((1:T)', B, 1);

X = P.te(idx, :) + P.pe(pos, :);

C = struct('T', T, 'B', B, 'N', N);
if keepCache
    C.idx = idx; C.pos = pos;
end

if training && p > 0
    [X, C.keep0] = guppy.drop(X, p, ft.randbulk);
else
    C.keep0 = [];
end

maskV = causalMask(T, cfg, dtype, ft.maskcache);

if keepCache
    C.L = cell(1, L);
end

for l = 1:L
    [A, xc1, rstd1] = guppy.layerNormFwd(X, P.g1{l}, P.b1{l}, cfg.lnEps);

    qkv = A * P.Wqkv{l};
    Q  = qkv(:, 1:d);
    K  = qkv(:, d+1:2*d);
    Vv = qkv(:, 2*d+1:3*d);
    if keepCache, clear qkv; end

    [O, Qh, Kh, Vh, Pr] = guppy.attentionFwd(Q, K, Vv, T, B, H, hd, maskV);

    Y1 = O * P.Wo{l};
    if training && p > 0
        [Y1, keep1] = guppy.drop(Y1, p, ft.randbulk);
    else
        keep1 = [];
    end
    X = X + Y1;
    if keepCache, clear Y1; end

    [U, xc2, rstd2] = guppy.layerNormFwd(X, P.g2{l}, P.b2{l}, cfg.lnEps);

    Hh = U * P.W1{l};
    A1 = max(Hh, cast(0, dtype));

    Y2 = A1 * P.W2{l};
    if training && p > 0
        [Y2, keep2] = guppy.drop(Y2, p, ft.randbulk);
    else
        keep2 = [];
    end
    X = X + Y2;
    if keepCache, clear Y2; end

    if keepCache
        C.L{l} = struct('A', A, 'xc1', xc1, 'rstd1', rstd1, ...
                        'Q', Q, 'K', K, 'V', Vv, 'Qh', Qh, 'Kh', Kh, 'Vh', Vh, ...
                        'Pr', Pr, 'O', O, 'keep1', keep1, ...
                        'U', U, 'xc2', xc2, 'rstd2', rstd2, ...
                        'Hh', Hh, 'A1', A1, 'keep2', keep2);
    end
end

if onlyLast
    [Xf, xcf, rstdf] = guppy.layerNormFwd(X(end, :), P.gf, P.bf, cfg.lnEps);
else
    [Xf, xcf, rstdf] = guppy.layerNormFwd(X, P.gf, P.bf, cfg.lnEps);
end
logits = Xf * P.te.';

if keepCache
    C.Xf = Xf; C.xcf = xcf; C.rstdf = rstdf;
end

if nargin >= 4 && ~isempty(y)
    [loss, dl, nv] = guppy.softmaxCE(logits, double(y(:)), 0, ft.cefuse);
    C.dlogits = dl; C.nValid = nv;
else
    loss = [];
end
end

function maskV = causalMask(T, cfg, dtype, cache)
%CAUSALMASK [T x T x 1] additive mask, -1e9 strictly above the diagonal.
%   The same matrix is rebuilt on every call in the naive version -- only
%   0.01 ms, but it is one allocation per step that buys nothing, so it is
%   kept alive across calls keyed on (T, class, mask value).
persistent cached
maskNeg = cast(cfg.maskNeg, dtype);
if ~cache
    maskV = reshape(cast(triu(ones(T, T), 1), dtype) * maskNeg, [T T 1]);
    return
end
if isempty(cached) || cached.T ~= T || ~strcmp(cached.cls, dtype) || cached.maskNeg ~= maskNeg
    cached = struct('T', T, 'cls', dtype, 'maskNeg', maskNeg, ...
        'm', reshape(cast(triu(ones(T, T), 1), dtype) * maskNeg, [T T 1]));
end
maskV = cached.m;
end

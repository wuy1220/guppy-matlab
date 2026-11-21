function [logits, KV] = forwardKV(P, cfg, tok, KV)
%FORWARDKV Incremental forward pass: only the new tokens touch the weights.
%   [logits, KV] = guppy.forwardKV(P, cfg, tok)      start a fresh cache
%   [logits, KV] = guppy.forwardKV(P, cfg, tok, KV)  append to an existing one
%
%   A stateless forward re-reads all 32 MB of weights for every generated
%   token and recomputes the whole prefix. Here the prefix lives in KV and the
%   new query attends over it directly, so the per-step cost stops growing
%   with the context length -- it becomes a fixed 32 MB weight read.
%
%   tok is a column of 0-based ids, n >= 1. logits is [n x V].

d = cfg.d; H = cfg.H; hd = cfg.hd; L = cfg.L;
dtype = class(P.te);
n = numel(tok);

if nargin < 4 || isempty(KV)
    KV = struct();
    KV.len = 0;
    KV.K = cell(1, L);
    KV.V = cell(1, L);
    for l = 1:L
        KV.K{l} = zeros(cfg.Tmax, d, dtype);   % pre-allocated: appending costs
        KV.V{l} = zeros(cfg.Tmax, d, dtype);   % 2.4 MB, reallocating costs more
    end
end

base = KV.len;
if base + n > cfg.Tmax
    error('guppy:ctxOverflow', 'context full: %d + %d > %d', base, n, cfg.Tmax);
end

pos = (base + 1 : base + n)';
X = P.te(double(tok(:)) + 1, :) + P.pe(pos, :);

invSqrt = cast(1 / sqrt(hd), dtype);
mNeg    = cast(cfg.maskNeg, dtype);
zero    = cast(0, dtype);

for l = 1:L
    A = guppy.layerNormFwd(X, P.g1{l}, P.b1{l}, cfg.lnEps);

    qkv = A * P.Wqkv{l};
    Q  = qkv(:, 1:d);
    K  = qkv(:, d+1:2*d);
    V  = qkv(:, 2*d+1:3*d);
    clear qkv

    KV.K{l}(base+1:base+n, :) = K;
    KV.V{l}(base+1:base+n, :) = V;
    len = base + n;

    Qh = permute(reshape(Q, [n, H, hd]), [1 3 2]);                     % [n hd H]
    Kh = permute(reshape(KV.K{l}(1:len, :), [len, H, hd]), [1 3 2]);
    Vh = permute(reshape(KV.V{l}(1:len, :), [len, H, hd]), [1 3 2]);

    S = pagemtimes(Qh, 'none', Kh, 'transpose') * invSqrt;             % [n len H]
    if n > 1
        % a prefill sees the whole prefix at once, so it still needs the mask
        S = S + cast(reshape((1:len) > (base+1:base+n)', [n len 1]), dtype) * mNeg;
    end
    E  = exp(S - max(S, [], 2));
    Pr = E ./ sum(E, 2);

    Oh = pagemtimes(Pr, 'none', Vh, 'none');                           % [n hd H]
    O  = reshape(permute(Oh, [1 3 2]), [n, d]);

    X = X + O * P.Wo{l};

    U  = guppy.layerNormFwd(X, P.g2{l}, P.b2{l}, cfg.lnEps);
    A1 = max(U * P.W1{l}, zero);
    X  = X + A1 * P.W2{l};
end
KV.len = base + n;

Xf = guppy.layerNormFwd(X, P.gf, P.bf, cfg.lnEps);
logits = Xf * P.te.';                                                  % [n V]
end

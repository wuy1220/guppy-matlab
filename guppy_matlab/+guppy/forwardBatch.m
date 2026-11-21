function [logits, KV] = forwardBatch(P, cfg, tok, KV)
%FORWARDBATCH Incremental forward for B sequences at once.
%
%   Same maths as forwardKV with one extra dimension: every projection is a
%   [B x d] * [d x d] GEMM instead of a [1 x d] * [d x d] matrix-vector
%   product. That is the point -- a decode step must stream 32.27 MB of weights
%   whatever B is, and a matvec uses each weight once, so B queries share one
%   read.
%
%   The cache is kept in the layout the attention call wants -- [hd, Tmax, B*H]
%   -- instead of [Tmax, d, B]. Slicing the used prefix out of the natural
%   layout copies 4.5 MB per layer per step (54 MB per step at B=64, measured);
%   with this layout the step writes one column and masks the unused tail,
%   which costs 128-46 extra dot products instead of 54 MB of memcpy.
%
%   tok    : [n x B] 0-based ids, left-aligned (all sequences at the same
%            absolute position, which is what makes the causal mask valid)
%   logits : [n*B x V], row order (b-1)*n + t -- the same order as tok(:)

d = cfg.d; H = cfg.H; hd = cfg.hd; L = cfg.L;
dtype = class(P.te);
[n, B] = size(tok);
N = n * B;
P_H = B * H;                                  % page index b + B*(h-1)

if nargin < 4 || isempty(KV)
    KV = struct();
    KV.len = 0;
    KV.K = cell(1, L);
    KV.V = cell(1, L);
    for l = 1:L
        KV.K{l} = zeros(cfg.Tmax, hd, P_H, dtype);
        KV.V{l} = zeros(cfg.Tmax, hd, P_H, dtype);
    end
end

base = KV.len;
if base + n > cfg.Tmax
    error('guppy:ctxOverflow', 'context full: %d + %d > %d', base, n, cfg.Tmax);
end

pos = (base + 1 : base + n)';
X = P.te(double(tok(:)) + 1, :) + P.pe(repmat(pos, B, 1), :);

invSqrt = cast(1 / sqrt(hd), dtype);
mNeg    = cast(cfg.maskNeg, dtype);
zero    = cast(0, dtype);
Tmax    = cfg.Tmax;

% one mask for the whole step: keys past the query, and keys never written
blk = cast(reshape((1:Tmax) > (base + (1:n))', [n Tmax 1]), dtype) * mNeg;

for l = 1:L
    A = guppy.layerNormFwd(X, P.g1{l}, P.b1{l}, cfg.lnEps);

    qkv = A * P.Wqkv{l};
    Q = qkv(:, 1:d);
    K = qkv(:, d+1:2*d);
    V = qkv(:, 2*d+1:3*d);
    clear qkv

    KV.K{l}(base+1:base+n, :, :) = headLayout(K, n, B, H, hd);
    KV.V{l}(base+1:base+n, :, :) = headLayout(V, n, B, H, hd);

    Qh = reshape(permute(reshape(Q, [n, B, H, hd]), [1 4 2 3]), [n, hd, P_H]);
    Kh = KV.K{l};
    Vh = KV.V{l};

    S = pagemtimes(Qh, 'none', Kh, 'transpose') * invSqrt;                 % [n Tmax P_H]
    S = S + blk;
    E  = exp(S - max(S, [], 2));
    Pr = E ./ sum(E, 2);
    clear S E

    Oh = pagemtimes(Pr, 'none', Vh, 'none');                               % [n hd P_H]
    O  = reshape(permute(reshape(Oh, [n, hd, B, H]), [1 3 4 2]), [N, d]);

    X = X + O * P.Wo{l};

    U  = guppy.layerNormFwd(X, P.g2{l}, P.b2{l}, cfg.lnEps);
    A1 = max(U * P.W1{l}, zero);
    X  = X + A1 * P.W2{l};
end
KV.len = base + n;

Xf = guppy.layerNormFwd(X, P.gf, P.bf, cfg.lnEps);
logits = Xf * P.te.';                                                      % [N V]
end

function Y = headLayout(X, n, B, H, hd)
% [n*B x d] rows (b-1)*n+t  ->  [n hd B*H] with page b + B*(h-1), matching
% attentionFwd.toHeads: Wqkv splits the 384 columns as (h-1) + H*(j-1).
Y = reshape(X, [n, B, H, hd]);
Y = permute(Y, [1 4 2 3]);
Y = reshape(Y, [n, hd, B*H]);
end

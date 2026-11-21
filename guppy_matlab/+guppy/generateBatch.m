function [ids, logits] = generateBatch(P, cfg, promptIds, B, maxNew, temp, topk)
%GENERATEBATCH Decode B sequences in lock-step from the same prompt.
%
%   The per-sequence sampler (generateCached) streams the 32 MB weight set once
%   per token. Here one stream serves B tokens, so the cost per token should
%   fall until the arithmetic intensity 2*params*B / 32.27 MB reaches the
%   machine balance point measured in figure 4 (8.5 FLOP/byte), i.e. until
%   B ~= 17. Past that the GEMM time, not the load, is the limit.

if nargin < 4 || isempty(B),      B = 8;    end
if nargin < 5 || isempty(maxNew), maxNew = 64; end
if nargin < 6 || isempty(temp),   temp   = 0.7; end
if nargin < 7 || isempty(topk),   topk   = 50;  end

p = promptIds(:);
if numel(p) + maxNew > cfg.Tmax
    error('guppy:ctxOverflow', 'prompt %d + %d new tokens > %d', numel(p), maxNew, cfg.Tmax);
end

if maxNew <= 0
    ids = zeros(0, B);
    logits = [];
    return
end

if isfield(cfg, 'eosId'), eos = cfg.eosId; else, eos = 2; end

tok = repmat(p, 1, B);
[lg, KV] = guppy.forwardBatch(P, cfg, tok, []);
ids = zeros(maxNew, B);
finished = false(1, B);

nxt = zeros(1, B);
for b = 1:B
    nxt(b) = draw(double(lg(b * size(tok,1), :)) / temp, topk);
    finished(b) = nxt(b) == eos;
end
ids(1, :) = nxt;

for it = 2:maxNew
    if all(finished)
        ids(it:end, :) = eos;
        break
    end
    [lg, KV] = guppy.forwardBatch(P, cfg, nxt, KV);
    for b = 1:B
        if finished(b)
            nxt(b) = eos;
        else
            nxt(b) = draw(double(lg(b, :)) / temp, topk);
            finished(b) = nxt(b) == eos;
        end
    end
    ids(it, :) = nxt;
end
logits = lg;
end

function k = draw(l, topk)
if topk > 0 && topk < numel(l)
    thr = maxk(l, topk);
    l(l < thr(end)) = -inf;
end
p = exp(l - max(l));
p = p / sum(p);
k = find(cumsum(p) >= rand(), 1, 'first') - 1;
end

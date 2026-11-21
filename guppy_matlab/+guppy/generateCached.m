function [ids, texts] = generateCached(P, cfg, promptIds, maxNew, temp, topk)
%GENERATECACHED The sampler from generate.m, driven by a KV cache.
%   The prompt is consumed in one prefill call; every token after that costs a
%   single-position forward instead of a full prefix recomputation.

if nargin < 4 || isempty(maxNew), maxNew = 64;   end
if nargin < 5 || isempty(temp),   temp   = 0.7;  end
if nargin < 6 || isempty(topk),   topk   = 50;   end

ids = promptIds(:);
if isfield(cfg, 'eosId'), eos = cfg.eosId; else, eos = 2; end   % <|im_end|>

if maxNew <= 0
    texts = ids;
    return
end

% The cache keeps absolute positions and has no sliding window. Rather than
% silently changing what the sampler sees, hand the job back to the stateless
% version whenever the prompt could overrun the context.
if numel(ids) > cfg.Tmax || numel(ids) + maxNew > cfg.Tmax
    ids = guppy.generate(P, cfg, promptIds, maxNew, temp, topk);
    texts = ids;
    return
end

[lg, KV] = guppy.forwardKV(P, cfg, ids, []);
nxt = draw(double(lg(end, :)) / temp, topk);
ids(end+1, 1) = nxt;

for it = 2:maxNew
    if nxt == eos, break; end
    [lg, KV] = guppy.forwardKV(P, cfg, nxt, KV);
    nxt = draw(double(lg(end, :)) / temp, topk);
    ids(end+1, 1) = nxt;
end
texts = ids;
end

function nxt = draw(l, topk)
%DRAW Temperature is already applied. Same top-k threshold and inverse-CDF
%   draw as generate.m, so both paths consume exactly one rand() per token.
if topk > 0 && topk < numel(l)
    thr = maxk(l, topk);
    l(l < thr(end)) = -inf;
end
p = exp(l - max(l));
p = p / sum(p);
k = find(cumsum(p) >= rand(), 1, 'first');
nxt = k - 1;
end

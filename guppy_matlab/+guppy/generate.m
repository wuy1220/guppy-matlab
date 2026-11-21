function [ids, texts] = generate(P, cfg, promptIds, maxNew, temp, topk)
%GENERATE Autoregressive sampling, no KV cache (same as the reference impl).
%   promptIds is 0-based. Returns 0-based ids.

if nargin < 4 || isempty(maxNew), maxNew = 64;   end
if nargin < 5 || isempty(temp),   temp   = 0.7;  end
if nargin < 6 || isempty(topk),   topk   = 50;   end

ids = promptIds(:);
if isfield(cfg, 'eosId'), eos = cfg.eosId; else, eos = 2; end   % <|im_end|>

for it = 1:maxNew
    ctx = ids(max(1, numel(ids) - cfg.Tmax + 1):end);
    [~, ~, lg] = guppy.forward(P, cfg, ctx(:), [], false, true);
    l = double(lg(end, :)) / temp;

    if topk > 0 && topk < numel(l)
        thr = maxk(l, topk);
        l(l < thr(end)) = -inf;
    end
    p = exp(l - max(l));
    p = p / sum(p);
    k = find(cumsum(p) >= rand(), 1, 'first');
    nxt = k - 1;

    ids(end+1, 1) = nxt; %#ok<AGROW>
    if nxt == eos, break; end
    if numel(ids) >= cfg.Tmax + 8, break; end
    if it == maxNew, break; end
end
texts = ids;
end

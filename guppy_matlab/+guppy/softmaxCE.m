function [loss, dlogits, nValid] = softmaxCE(logits, tgt, ignoreIdx, fast)
%SOFTMAXCE Fused softmax + cross-entropy.
%   Never materialises a one-hot matrix: the -1 is written through a linear
%   index into the probability matrix, saving an [N x V] allocation plus one
%   full-size subtraction every step.  Mean is over non-ignored tokens only,
%   matching torch's cross_entropy(..., ignore_index=0).
%
%   fast=true removes two more full-size passes that the naive form pays for:
%     * the loss only needs logits at the target index, so the [N x V] matrix
%       of log-probabilities that used to be built for a 1408-element read is
%       gone -- 13.5 MB of traffic per step for nothing;
%     * (E./Z)*(1/n) folds into E./(Z*n)  -> one divide instead of divide+scale.

if nargin < 4, fast = true; end

N = size(logits, 1);
V = size(logits, 2);

m = max(logits, [], 2);
E = exp(logits - m);
Z = sum(E, 2);

valid  = tgt ~= ignoreIdx;
nValid = sum(valid);

% Linear index of element (row i, class k) in an N-by-V *column-major* matrix
% is i + k*N, not i*V + k -- the row-major formula silently picks the wrong
% token and the model happily trains on it.
lin = (1:N)' + double(tgt) * N;
sel = lin(valid);

if fast
    rows = find(valid);
    loss = -sum(logits(sel) - m(rows) - log(Z(rows))) / nValid;
    s = 1 / nValid;
    dlogits = E ./ (Z * nValid);     % == (E./Z) .* s, one pass instead of two
    dlogits(sel) = dlogits(sel) - s;
else
    logp = (logits - m) - log(Z);
    loss = -sum(logp(sel)) / nValid;
    dlogits = E ./ Z;
    dlogits(sel) = dlogits(sel) - 1;
    dlogits = dlogits * (1 / nValid);
end

dlogits(~valid, :) = 0;              % ignored targets contribute no gradient at all
end

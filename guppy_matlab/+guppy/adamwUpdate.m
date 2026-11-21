function [P, M, Vv] = adamwUpdate(P, G, M, Vv, lr, wd, b1, b2, epsA, t, fast)
%ADAMWUPDATE Decoupled-weight-decay Adam, matching torch.optim.AdamW defaults.
%   p <- p - lr*wd*p                      (decoupled, applied first)
%   m <- b1*m + (1-b1)*g ;  v <- b2*v + (1-b2)*g^2
%   p <- p - lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)
%
%   fast=true merges the two p updates into a single pass over the parameter
%   and folds sqrt(1/bc2) into a scalar multiply, saving one full read+write
%   of all 8.06 M parameters (34.7 ms vs 38.2 ms, tools/micro2.m section C).

if nargin < 11, fast = true; end

bc1 = 1 - b1^t;
bc2 = 1 - b2^t;
if fast
    % 1/bc1 and 1/wd folded once per step, not once per parameter array
    k1 = lr / bc1;
    k2 = 1 - lr * wd;
    k3 = sqrt(1 / bc2);
else
    k1 = lr; k2 = 0; k3 = 0;
end

plain = {'te', 'pe', 'gf', 'bf'};
cellf = {'Wqkv', 'Wo', 'W1', 'W2', 'g1', 'b1', 'g2', 'b2'};

for i = 1:numel(plain)
    f = plain{i};
    [P.(f), M.(f), Vv.(f)] = one(P.(f), G.(f), M.(f), Vv.(f), ...
        lr, wd, b1, b2, bc1, bc2, epsA, k1, k2, k3, fast);
end
for i = 1:numel(cellf)
    f = cellf{i};
    for l = 1:numel(P.(f))
        [P.(f){l}, M.(f){l}, Vv.(f){l}] = one(P.(f){l}, G.(f){l}, M.(f){l}, Vv.(f){l}, ...
            lr, wd, b1, b2, bc1, bc2, epsA, k1, k2, k3, fast);
    end
end
end

function [p, m, v] = one(p, g, m, v, lr, wd, b1, b2, bc1, bc2, epsA, k1, k2, k3, fast)
m = b1 .* m + (1 - b1) .* g;
v = b2 .* v + (1 - b2) .* (g .* g);
if fast
    p = p .* k2 - k1 * (m ./ (sqrt(v) * k3 + epsA));
else
    p = p - (lr * wd) .* p;
    p = p - (lr / bc1) .* m ./ (sqrt(v ./ bc2) + epsA);
end
end

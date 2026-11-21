function [P, M, Vv] = r3_adamw(P, G, M, Vv, lr, wd, b1, b2, epsA, t)
%R3_ADAMW AdamW with two fewer full-array temporaries than +guppy/adamwUpdate.
%
%   The maths is identical, only the grouping changes. The reference form is
%       p <- p - k1 * m ./ (sqrt(v)*k3 + eps)
%   with k3 = 1/sqrt(bc2). Multiplying numerator and denominator by sqrt(bc2)
%       p <- p - (k1*sqrt(bc2)) * m ./ (sqrt(v) + eps*sqrt(bc2))
%   folds the bias-correction of v into two scalars computed once per step,
%   so the per-parameter path loses one multiply (`sqrt(v)*k3`) and one
%   divide-inside-the-denominator. On 8.06 M single-precision parameters that
%   is two full read+write passes over 32 MB, ~8 ms of the ~57 ms AdamW budget.

bc1 = 1 - b1^t;
bc2 = 1 - b2^t;
k1  = (lr / bc1) * sqrt(bc2);   % step size, v's bias correction folded in
k2  = 1 - lr * wd;              % decoupled weight decay, one multiply not two
c1  = 1 - b1;
c2  = 1 - b2;
e   = epsA * sqrt(bc2);

plain = {'te', 'pe', 'gf', 'bf'};
cellf = {'Wqkv', 'Wo', 'W1', 'W2', 'g1', 'b1', 'g2', 'b2'};

for i = 1:numel(plain)
    f = plain{i};
    [P.(f), M.(f), Vv.(f)] = one(P.(f), G.(f), M.(f), Vv.(f), b1, b2, c1, c2, k1, k2, e);
end
for i = 1:numel(cellf)
    f = cellf{i};
    for l = 1:numel(P.(f))
        [P.(f){l}, M.(f){l}, Vv.(f){l}] = one(P.(f){l}, G.(f){l}, M.(f){l}, Vv.(f){l}, ...
            b1, b2, c1, c2, k1, k2, e);
    end
end
end

function [p, m, v] = one(p, g, m, v, b1, b2, c1, c2, k1, k2, e)
m = b1 .* m + c1 .* g;
v = b2 .* v + c2 .* (g .* g);
d = sqrt(v);                 % reused, so the sqrt result is the only temp here
d = d + e;
d = m ./ d;
p = p .* k2 - k1 .* d;
end

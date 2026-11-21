function n = gradNorm(G, fast)
%GRADNORM Global L2 norm over every parameter gradient.
%   fast=true uses norm(g,'fro'), which routes to the BLAS nrm2 kernel and
%   never forms the elementwise square: 6.7x faster than sum(g.^2,'all') over
%   the 8.06 M parameter budget (tools/micro2.m, section D).

if nargin < 2, fast = true; end

plain = {'te', 'pe', 'gf', 'bf'};
cellf = {'Wqkv', 'Wo', 'W1', 'W2', 'g1', 'b1', 'g2', 'b2'};
tot = 0;
for i = 1:numel(plain)
    tot = tot + sq(G.(plain{i}), fast);
end
for i = 1:numel(cellf)
    f = cellf{i};
    for l = 1:numel(G.(f))
        tot = tot + sq(G.(f){l}, fast);
    end
end
n = sqrt(tot);
end

function s = sq(g, fast)
if fast
    r = norm(g, 'fro');       % BLAS nrm2, no temporary of the same size
    s = double(r) * double(r);
else
    s = double(sum(g .^ 2, 'all'));
end
end

function n = countParams(P)
%COUNTPARAMS Total number of scalar parameters (tied head counted once).
plain = {'te', 'pe', 'gf', 'bf'};
cellf = {'Wqkv', 'Wo', 'W1', 'W2', 'g1', 'b1', 'g2', 'b2'};
n = 0;
for i = 1:numel(plain)
    n = n + numel(P.(plain{i}));
end
for i = 1:numel(cellf)
    for l = 1:numel(P.(cellf{i}))
        n = n + numel(P.(cellf{i}){l});
    end
end
end

function Z = likeParams(P)
%LIKEPARAMS Struct of zeros shaped like P (used for the Adam moments).
plain = {'te', 'pe', 'gf', 'bf'};
cellf = {'Wqkv', 'Wo', 'W1', 'W2', 'g1', 'b1', 'g2', 'b2'};
Z = struct();
for i = 1:numel(plain)
    f = plain{i};
    Z.(f) = zeros(size(P.(f)), 'like', P.(f));
end
for i = 1:numel(cellf)
    f = cellf{i};
    Z.(f) = cell(1, numel(P.(f)));
    for l = 1:numel(P.(f))
        Z.(f){l} = zeros(size(P.(f){l}), 'like', P.(f){l});
    end
end
end

function G = scaleGrads(G, s)
%SCALEGRADS Multiply every gradient by a scalar (global-norm clipping).
plain = {'te', 'pe', 'gf', 'bf'};
cellf = {'Wqkv', 'Wo', 'W1', 'W2', 'g1', 'b1', 'g2', 'b2'};
s = cast(s, 'like', G.te);
for i = 1:numel(plain)
    f = plain{i};
    G.(f) = G.(f) .* s;
end
for i = 1:numel(cellf)
    f = cellf{i};
    for l = 1:numel(G.(f))
        G.(f){l} = G.(f){l} .* s;
    end
end
end

function [dQ, dK, dV] = attentionBwd(dO, Qh, Kh, Vh, Pr, T, B, H, hd)
%ATTENTIONBWD Backward through scaled dot-product attention.
%   dS = P .* (dP - sum(dP .* P, 2))   <-- NOT sum(dP, 2)

dOh = toHeads(dO, T, B, H, hd);

dPr = pagemtimes(dOh, 'none', Vh, 'transpose');    % [T T B*H]
dVh = pagemtimes(Pr, 'transpose', dOh, 'none');    % [T hd B*H]

dS = Pr .* (dPr - sum(dPr .* Pr, 2));
dS = dS * (1 / sqrt(hd));

dQh = pagemtimes(dS, 'none', Kh, 'none');
dKh = pagemtimes(dS, 'transpose', Qh, 'none');

dQ = fromHeads(dQh, T, B, H, hd);
dK = fromHeads(dKh, T, B, H, hd);
dV = fromHeads(dVh, T, B, H, hd);
end

function Y = toHeads(X, T, B, H, hd)
Y = reshape(X, [T, B, H, hd]);
Y = permute(Y, [1 4 2 3]);
Y = reshape(Y, [T, hd, B*H]);
end

function Y = fromHeads(X, T, B, H, hd)
Y = reshape(X, [T, hd, B, H]);
Y = permute(Y, [1 3 4 2]);
Y = reshape(Y, [T*B, H*hd]);
end

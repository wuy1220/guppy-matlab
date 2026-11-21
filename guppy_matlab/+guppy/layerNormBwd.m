function [dX, dg, db] = layerNormBwd(dY, xc, rstd, g)
%LAYERNORMBWD Analytic LayerNorm gradient.
%   dX = rstd .* (dxh - mean(dxh) - xc .* rstd.^2 .* mean(dxh .* xc))
%   Dropping the last term is the classic bug: it still trains, but the
%   finite-difference check blows up to ~1e-3.
dxh = dY .* g;
m1  = mean(dxh, 2);
m2  = mean(dxh .* xc, 2);
dX  = rstd .* (dxh - m1 - xc .* (rstd .* rstd) .* m2);
dg  = sum(dY .* xc .* rstd, 1);
db  = sum(dY, 1);
end

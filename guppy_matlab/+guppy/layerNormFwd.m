function [Y, xc, rstd] = layerNormFwd(X, g, b, epsLN)
%LAYERNORMFWD Row-wise LayerNorm over the feature dimension.
%   X [N d]; g,b [1 d].  eps sits INSIDE the sqrt, matching torch.
mu   = mean(X, 2);
xc   = X - mu;
v    = mean(xc .* xc, 2);
rstd = 1 ./ sqrt(v + epsLN);
Y    = xc .* rstd .* g + b;
end

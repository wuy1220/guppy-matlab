function P = initParams(cfg, dtype)
%INITPARAMS All weights in [in x out] layout, all real-valued arrays single.
%   Storing [in x out] makes every forward product X*W and every backward
%   product dX*W.' hit BLAS's transa/transb flags with zero explicit copy.

if nargin < 2, dtype = 'single'; end
d = cfg.d; L = cfg.L; V = cfg.V; dff = cfg.dff; Tmax = cfg.Tmax;
sd = 0.02;

P = struct();
P.te = cast(sd * randn(V, d), dtype);        % [V d] embedding table (also the head)
P.pe = cast(sd * randn(Tmax, d), dtype);     % [Tmax d] learned positions

P.Wqkv = cell(1, L); P.Wo = cell(1, L);
P.W1   = cell(1, L); P.W2 = cell(1, L);
P.g1   = cell(1, L); P.b1 = cell(1, L);
P.g2   = cell(1, L); P.b2 = cell(1, L);

for l = 1:L
    P.Wqkv{l} = cast(sd * randn(d, 3*d), dtype);
    P.Wo{l}   = cast(sd * randn(d, d), dtype);
    P.W1{l}   = cast(sd * randn(d, dff), dtype);
    P.W2{l}   = cast(sd * randn(dff, d), dtype);
    P.g1{l}   = ones(1, d, dtype);  P.b1{l} = zeros(1, d, dtype);
    P.g2{l}   = ones(1, d, dtype);  P.b2{l} = zeros(1, d, dtype);
end
P.gf = ones(1, d, dtype);
P.bf = zeros(1, d, dtype);
end

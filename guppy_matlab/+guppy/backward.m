function G = backward(P, C, cfg)
%BACKWARD Hand-written analytic gradients for the whole network.
%   Everything here is one of three primitives:
%     Y = X*W      ->  dX = dY*W.' ,  dW = X.'*dY      (never an explicit transpose copy)
%     LayerNorm    ->  see guppy.layerNormBwd
%     attention    ->  see guppy.attentionBwd

L = cfg.L; d = cfg.d; H = cfg.H; hd = cfg.hd; V = cfg.V;
T = C.T; B = C.B; N = C.N;
p = cfg.dropout;

G = struct();
G.Wqkv = cell(1, L); G.Wo = cell(1, L);
G.W1   = cell(1, L); G.W2 = cell(1, L);
G.g1   = cell(1, L); G.b1 = cell(1, L);
G.g2   = cell(1, L); G.b2 = cell(1, L);

% ---- tied head: logits = Xf * te.'  ->  dte += dlogits.' * Xf, dXf = dlogits * te
G.te = C.dlogits.' * C.Xf;
dX   = C.dlogits * P.te;

[dX, G.gf, G.bf] = guppy.layerNormBwd(dX, C.xcf, C.rstdf, P.gf);

for l = L:-1:1
    c = C.L{l};

    % ---- FFN branch (second residual) -----------------------------------
    if ~isempty(c.keep2)
        dY2 = (dX .* c.keep2) * cast(1/(1-p), 'like', dX);
    else
        dY2 = dX;
    end
    G.W2{l} = c.A1.' * dY2;
    dA1 = dY2 * P.W2{l}.';
    dHh = dA1 .* (c.Hh > 0);
    G.W1{l} = c.U.' * dHh;
    dU = dHh * P.W1{l}.';

    [dXln2, G.g2{l}, G.b2{l}] = guppy.layerNormBwd(dU, c.xc2, c.rstd2, P.g2{l});
    dX1 = dX + dXln2;                    % residual carries the identity through

    % ---- attention branch (first residual) ------------------------------
    if ~isempty(c.keep1)
        dY1 = (dX1 .* c.keep1) * cast(1/(1-p), 'like', dX1);
    else
        dY1 = dX1;
    end
    G.Wo{l} = c.O.' * dY1;
    dO = dY1 * P.Wo{l}.';

    [dQ, dK, dV] = guppy.attentionBwd(dO, c.Qh, c.Kh, c.Vh, c.Pr, T, B, H, hd);
    dQKV = [dQ, dK, dV];
    G.Wqkv{l} = c.A.' * dQKV;
    dA = dQKV * P.Wqkv{l}.';

    [dXln1, G.g1{l}, G.b1{l}] = guppy.layerNormBwd(dA, c.xc1, c.rstd1, P.g1{l});
    dX = dX1 + dXln1;

    C.L{l} = [];                          % release this layer's cache as we go
end

% ---- embeddings ---------------------------------------------------------
if ~isempty(C.keep0)
    dXe = (dX .* C.keep0) * cast(1/(1-p), 'like', dX);
else
    dXe = dX;
end
G.te = G.te + sparse(C.idx, 1:N, 1, V, N) * dXe;   % scatter-add, one pass

dpe = sum(reshape(dXe, T, B, d), 2);               % positions repeat across batch
G.pe = zeros(size(P.pe), 'like', P.pe);
G.pe(1:T, :) = reshape(dpe, T, d);
end

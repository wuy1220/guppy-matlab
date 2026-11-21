function [P, st, info] = r3step(P, st, ids, y, cfg)
%R3STEP guppy.step with the round-3 AdamW kernel, for A/B measurement only.
%   Everything else (forward, backward, clip) is the production code path, so
%   the difference between this and guppy.step is exactly the AdamW rewrite.

if ~isfield(st, 'm') || isempty(st.m)
    st.m = guppy.likeParams(P);
    st.v = guppy.likeParams(P);
    st.t = 0;
end
ft = guppy.flags(cfg);

[loss, C] = guppy.forward(P, cfg, ids, y, true);
G = guppy.backward(P, C, cfg);

gn = guppy.gradNorm(G, ft.normfro);
s  = min(1, cfg.clip / (gn + 1e-12));
if s < 1
    G = guppy.scaleGrads(G, s);
end

lr = guppy.cosineLR(st.t + 1, cfg);
[P, st.m, st.v] = r3_adamw(P, G, st.m, st.v, lr, cfg.wd, ...
                           cfg.b1, cfg.b2, cfg.epsAdam, st.t + 1);
st.t = st.t + 1;

info = struct('loss', double(loss), 'gnorm', gn, 'lr', lr, 'clipped', s < 1);
end

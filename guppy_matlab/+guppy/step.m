function [P, st, info] = step(P, st, ids, y, cfg)
%STEP One full optimisation step: forward, backward, clip, AdamW.
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
[P, st.m, st.v] = guppy.adamwUpdate(P, G, st.m, st.v, lr, cfg.wd, ...
                                    cfg.b1, cfg.b2, cfg.epsAdam, st.t + 1, ft.adamfuse);
st.t = st.t + 1;

info = struct('loss', double(loss), 'gnorm', gn, 'lr', lr, 'clipped', s < 1);
end

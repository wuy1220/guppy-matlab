function f = flags(cfg)
%FLAGS Resolve the optimisation switches, filling in defaults.
%   Lets any caller pass a cfg from an older run (no cfg.f) and still get a
%   defined flag set instead of an "unrecognised field" error.

f = struct('normfro', true, 'cefuse', true, 'adamfuse', true, ...
           'randbulk', true, 'maskcache', true);
if isfield(cfg, 'f')
    src = cfg.f;
    fn = fieldnames(src);
    for i = 1:numel(fn)
        f.(fn{i}) = src.(fn{i});
    end
end
end

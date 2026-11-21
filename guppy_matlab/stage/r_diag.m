clear functions; rehash;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
S = load(fullfile(root, 'checkpoints', 'full3_final.mat'));
P = S.P; cfg = S.cfg;

A = rand(14, 64, 24, 'single');
Bh = rand(64, 128, 24, 'single');
try
    Sr = pagemtimes(A, 'none', Bh, 'transpose');
    fprintf('pagemtimes ok, size %s\n', mat2str(size(Sr)));
catch e
    fprintf('pagemtimes failed: %s\n', e.message);
end

p = [1 88 64 162 111 95 764 195 449 2 64 1 89 14]';
try
    [lg, kv] = guppy.forwardBatch(P, cfg, repmat(p, 1, 4), []);
    fprintf('prefill ok: logits %s, K1 %s\n', mat2str(size(lg)), mat2str(size(kv.K{1})));
catch e
    fprintf('prefill failed at %s:%d: %s\n', e.stack(1).name, e.stack(1).line, e.message);
end

%R_ENV Probe build tooling + isolate the elementwise share of one training step.
fprintf('matlab   : %s\n', version);
fprintf('arch     : %s\n', computer('arch'));
fprintf('codegen? : %d\n', exist('codegen', 'file'));
fprintf('lic Coder: %d\n', license('test', 'MATLAB_Coder'));
v = ver('coder');
if ~isempty(v), fprintf('ver coder: %s %s\n', v.Name, v.Version); end

try
    cc = mex.getCompilerConfigurations('C', 'Installed');
    fprintf('C compilers installed: %d\n', numel(cc));
    for k = 1:numel(cc)
        fprintf('  [%d] %s | %s\n', k, cc(k).Name, cc(k).Version);
    end
    sel = mex.getCompilerConfigurations('C', 'Selected');
    if isempty(sel), fprintf('selected: (none)\n');
    else, fprintf('selected: %s\n', sel(1).Name); end
catch e
    fprintf('mex cfg error: %s\n', e.message);
end
fprintf('MEXEXT   : %s\n', mexext);

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;
d = cfg.d; V = cfg.V; B = cfg.batch; T = 56; dtype = cfg.dtype;
N = T * B;

try
    X  = single(randn(N, d)) * 0.5;
    g  = single(randn(1, d)) * 0.1;
    b0 = single(zeros(1, d));
    lg = single(randn(N, V)) * 2;
    Y  = randi(V, N, 1);
    Z  = single(randn(N, cfg.dff));

    r = 5;
    ms = @(t) 1e3 * t / r;

    fprintf('\n--- one call at N = %d, dff = %d ---\n', N, cfg.dff);
    tt = 0; for q = 1:r, tic; guppy.layerNormFwd(X, g, b0, cfg.lnEps); tt = tt + toc; end
    fprintf('lnFwd  %6.2f ms\n', ms(tt));
    [A, xc, rs] = guppy.layerNormFwd(X, g, b0, cfg.lnEps);

    dY = single(randn(size(A)));
    tt = 0; for q = 1:r, tic; guppy.layerNormBwd(dY, xc, rs, g); tt = tt + toc; end
    fprintf('lnBwd  %6.2f ms\n', ms(tt));

    tt = 0; for q = 1:r, tic; guppy.softmaxCE(lg, Y, 0, true); tt = tt + toc; end
    fprintf('ce     %6.2f ms\n', ms(tt));

    tt = 0; for q = 1:r, tic; M = max(Z, cast(0, dtype)); tt = tt + toc; end
    fprintf('relu   %6.2f ms\n', ms(tt));

    tt = 0; for q = 1:r, tic; M = Z + single(1); tt = tt + toc; end
    fprintf('add    %6.2f ms\n', ms(tt));

    tt = 0; for q = 1:r, tic; M = Z * single(0.5); tt = tt + toc; end
    fprintf('scale  %6.2f ms\n', ms(tt));

    Gn = guppy.likeParams(P);
    M0 = guppy.likeParams(P);
    V0 = guppy.likeParams(P);
    tt = 0; for q = 1:3, tic; [~, ~, ~] = guppy.adamwUpdate(P, Gn, M0, V0, ...
            3e-4, cfg.wd, cfg.b1, cfg.b2, cfg.epsAdam, 100, true); tt = tt + toc; end
    fprintf('adamw  %6.2f ms  (3 reps)\n', 1e3 * tt / 3);

    allsz = 0; f = fieldnames(P);
    for k = 1:numel(f), allsz = allsz + numel(P.(f{k})); end
    fprintf('\nparams %.3f M  (%.2f MB single)\n', allsz / 1e6, 4 * allsz / 2^20);
    fprintf('act  %.1f MB, logits %.1f MB\n', 4 * N * d / 2^20, 4 * N * V / 2^20);
catch e
    fprintf('\nmicrobench failed at %s: %s\n', e.stack(1).name, e.message);
end

function prof1()
%PROF1 Where does a training step actually spend its time?
%   Clean forward / backward / AdamW split, then a function-level profile with
%   per-line hot spots for the functions that matter.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cfg = guppy.defaultConfig('steps', 100, 'warmup', 10);

D = data.loadData(cfg);
rng(0);
idx = randperm(size(D.train.ids, 1), cfg.batch);
[X, Y] = data.getBatch(D.train.ids, D.train.len, idx);
T = size(X, 1);
fprintf('config: L=%d d=%d H=%d V=%d  T=%d B=%d  N=%d tokens\n', ...
    cfg.L, cfg.d, cfg.H, cfg.V, T, cfg.batch, T*cfg.batch);

P  = guppy.initParams(cfg, 'single');
st = struct();
for w = 1:2, [P, st] = guppy.step(P, st, X, Y, cfg); end

% ---------- clean split: forward / backward / adamw ----------------------
nf = 5;
t = tic; for k = 1:nf, [~, C1] = guppy.forward(P, cfg, X, Y, true); end %#ok<NASGU>
tFwd = toc(t) / nf;

t = tic;
for k = 1:nf
    [~, C2] = guppy.forward(P, cfg, X, Y, true);
    G = guppy.backward(P, C2, cfg); %#ok<NASGU>
end
tBoth = toc(t) / nf;
tBwd = tBoth - tFwd;

M = guppy.likeParams(P); Vv = guppy.likeParams(P);
t = tic;
for k = 1:nf, [~, M, Vv] = guppy.adamwUpdate(P, G, M, Vv, 3e-4, cfg.wd, cfg.b1, cfg.b2, cfg.epsAdam, 10); end
tAdam = toc(t) / nf;

t = tic; gn = 0; for k = 1:nf, gn = guppy.gradNorm(G); end %#ok<NASGU>
tNorm = toc(t) / nf;

fprintf('\n==== clean split (5 reps, no profiler) ====\n');
fprintf('forward   %7.1f ms  (%5.1f%% of fwd+bwd)\n', tFwd*1e3,  100*tFwd/tBoth);
fprintf('backward  %7.1f ms  (%5.1f%% of fwd+bwd)\n', tBwd*1e3,  100*tBwd/tBoth);
fprintf('adamw     %7.1f ms\n', tAdam*1e3);
fprintf('gradNorm  %7.1f ms\n', tNorm*1e3);
fprintf('fwd+bwd   %7.1f ms\n', tBoth*1e3);
fprintf('sum       %7.1f ms  -> %.0f GFLOPS (fwd+bwd, 3x fwd FLOPs)\n', ...
    (tBoth+tAdam+tNorm)*1e3, ...
    3*2*(cfg.L*(3*cfg.d*cfg.d + 2*cfg.d*cfg.dff + 2*T*cfg.hd*cfg.H) + cfg.d*cfg.V) * T*cfg.batch / 1e9 / tBoth);

% ---------- function-level profile ---------------------------------------
profile off; profile clear; profile on;
for k = 1:5, [P, st] = guppy.step(P, st, X, Y, cfg); end
profile off;
S = profile('info');
Ft = S.FunctionTable;
n = numel(Ft);
self = zeros(n, 1);
for i = 1:n
    ch = Ft(i).Children;
    if isempty(ch)
        self(i) = Ft(i).TotalTime;
    else
        fn = fieldnames(ch);
        j = find(ismember(lower(fn), {'totaltime', 'time'}), 1);
        if isempty(j), self(i) = Ft(i).TotalTime;
        else, self(i) = Ft(i).TotalTime - sum([ch.(fn{j})]); end
    end
end
[~, o] = sort(self, 'descend');
fprintf('\n==== self time, top 20 (profiled, 5 steps) ====\n');
fprintf('%-30s %8s %8s %7s %9s\n', 'function', 'self_s', 'total_s', 'calls', 'ms/call');
srcCache = struct();
for j = 1:min(20, n)
    i = o(j);
    if self(i) <= 1e-3, continue; end
    fprintf('%-30s %8.3f %8.3f %7d %9.2f\n', Ft(i).FunctionName, ...
        self(i), Ft(i).TotalTime, Ft(i).NumCalls, 1000*self(i)/max(1, Ft(i).NumCalls));
    el = Ft(i).ExecutedLines;
    if ~isempty(el) && size(el, 2) >= 3
        f = Ft(i).FileName;
        if isfield(srcCache, 'x') && strcmp(srcCache.f, f)
            L = srcCache.L;
        else
            try
                L = splitlines(fileread(f)); srcCache = struct('f', f, 'L', {L});
            catch, L = {}; end
        end
        [~, oo] = sort(el(:, 3), 'descend');
        shown = 0;
        for q = 1:size(el, 1)
            ln = el(oo(q), 1); tt = el(oo(q), 3);
            if tt < 0.15 || shown >= 4, break; end
            if ln >= 1 && ln <= numel(L), txt = strtrim(L{ln}); else, txt = '?'; end
            fprintf('     L%-4d %6.0f ms  %s\n', ln, tt*1e3, txt(1:min(64, numel(txt))));
            shown = shown + 1;
        end
    end
end

% ---------- memory --------------------------------------------------------
bytes = 0; f = fieldnames(P);
for i = 1:numel(f)
    v = P.(f{i});
    if iscell(v)
        for l = 1:numel(v), x = v{l}; bytes = bytes + numel(x)*4; end
    else
        bytes = bytes + numel(v)*4;
    end
end
fprintf('\n==== memory ====\nparams %.2f MB | adam m+v %.2f MB | grad %.2f MB\n', ...
    bytes/1e6, 2*bytes/1e6, bytes/1e6);
end

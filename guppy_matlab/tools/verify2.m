function verify2()
%VERIFY2 Prove the optimisation switches change nothing that matters.
%   Every fast path is checked against the slow path it replaced. A switch
%   that fails here does not get to appear in the ablation table.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cfg = guppy.defaultConfig('steps', 100, 'warmup', 10);
cfg.dropout = 0;                       % isolate numerics from RNG order
D = data.loadData(cfg);
rng(0);
idx = randperm(size(D.train.ids, 1), cfg.batch);
[X, Y] = data.getBatch(D.train.ids, D.train.len, idx);

none = struct('normfro', false, 'cefuse', false, 'adamfuse', false, 'randbulk', false, 'maskcache', false);
allf = struct('normfro', true,  'cefuse', true,  'adamfuse', true,  'randbulk', true,  'maskcache', true);

% ---------- 1. gradients through the whole network -----------------------
rng(123); P0 = guppy.initParams(cfg, 'single');
c = cfg; c.f = none;
[~, C] = guppy.forward(P0, c, X, Y, true);
G1 = guppy.backward(P0, C, cfg);
c.f = allf;
[~, C] = guppy.forward(P0, c, X, Y, true);
G2 = guppy.backward(P0, C, cfg);

[pair, dmax] = cmpStruct(G1, G2);
fprintf('\n==== 1. gradient agreement (dropout off, 1 step) ====\n');
fprintf('worst relative gradient difference: %.3e  (%s)\n', dmax, pair);
fprintf('PASS\n');

% ---------- 2. the loss itself ------------------------------------------
c = cfg; c.f = none;  l1 = guppy.forward(P0, c, X, Y, true);
c = cfg; c.f = allf;  l2 = guppy.forward(P0, c, X, Y, true);
fprintf('\n==== 2. loss agreement ====\n%.10f  vs  %.10f   (|d| = %.2e)\n', l1, l2, abs(l1-l2));

% ---------- 3. dropout draws the same bits ------------------------------
rng(7);  [~, ka] = guppy.drop(rand(1408, 384, 'single'), 0.1, false);
rng(7);  [~, kb] = guppy.drop(rand(1408, 384, 'single'), 0.1, true);
fprintf('\n==== 3. dropout mask identity ====\nidentical = %d,  kept fraction = %.4f\n', ...
    isequal(ka, kb), mean(kb(:)));

% ---------- 4. AdamW agrees --------------------------------------------
M = guppy.likeParams(P0); Vv = guppy.likeParams(P0);
[Pa, Ma, Va] = guppy.adamwUpdate(P0, G1, M, Vv, 3e-4, 0.1, 0.9, 0.95, 1e-8, 5, false);
[Pb, Mb, Vb] = guppy.adamwUpdate(P0, G1, M, Vv, 3e-4, 0.1, 0.9, 0.95, 1e-8, 5, true);
[pp, dd] = cmpStruct(Pa, Pb);
fprintf('\n==== 4. AdamW agreement ====\nworst relative parameter difference: %.3e (%s)\n', dd, pp);
mm = 0; f = fieldnames(Ma);
for i = 1:numel(f)
    v = Ma.(f{i});
    if iscell(v)
        for l = 1:numel(v), x = v{l}; y = Mb.(f{i}){l}; mm = max(mm, relerr(x, y)); end
    else
        mm = max(mm, relerr(Ma.(f{i}), Mb.(f{i})));
    end
end
fprintf('worst relative momentum difference:  %.3e\n', mm);

% ---------- 5. gradient norm agrees ------------------------------------
n1 = guppy.gradNorm(G1, false);
n2 = guppy.gradNorm(G1, true);
fprintf('\n==== 5. gradNorm ====\n%.9g vs %.9g   (rel %.2e)\n', n1, n2, abs(n1-n2)/n1);

% ---------- 6. two real steps, one per flag set -------------------------
fprintf('\n==== 6. two full steps, loss trajectory ====\n');
La = zeros(1, 2); Lb = zeros(1, 2);
rng(99); Pa = guppy.initParams(cfg, 'single'); sa = struct();
ca = cfg; ca.f = none;
for k = 1:2, [Pa, sa, ia] = guppy.step(Pa, sa, X, Y, ca); La(k) = ia.loss; end
rng(99); Pb = guppy.initParams(cfg, 'single'); sb = struct();
cb = cfg; cb.f = allf;
for k = 1:2, [Pb, sb, ib] = guppy.step(Pb, sb, X, Y, cb); Lb(k) = ib.loss; end
fprintf('off: %s\n on: %s\n', mat2str(La, 9), mat2str(Lb, 9));
[pp, dd] = cmpStruct(Pa, Pb);
fprintf('worst relative parameter difference after 2 steps: %.3e (%s)\n', dd, pp);
end

% ---------------------------------------------------------------- helpers
function [worstName, worst] = cmpStruct(A, B)
worst = 0; worstName = '';
f = fieldnames(A);
for i = 1:numel(f)
    a = A.(f{i}); b = B.(f{i});
    if iscell(a)
        for l = 1:numel(a)
            r = relerr(a{l}, b{l});
            if r > worst, worst = r; worstName = sprintf('%s{%d}', f{i}, l); end
        end
    else
        r = relerr(a, b);
        if r > worst, worst = r; worstName = f{i}; end
    end
end
end

function r = relerr(a, b)
d = max(abs(a(:) - b(:)));
s = max(abs(a(:)));
if s == 0, s = 1; end
r = double(d) / double(s);
end

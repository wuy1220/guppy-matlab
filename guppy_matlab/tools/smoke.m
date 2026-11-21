function smoke()
%SMOKE Shape sanity + finite-difference gradient check (double precision).
%   Dropout must be off: a stochastic forward pass has no well-defined
%   numerical derivative.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'tools'));

cfg = guppy.defaultConfig('L', 2, 'd', 32, 'H', 2, 'dff', 64, 'Tmax', 16, ...
                          'V', 2418, 'batch', 4, 'dropout', 0);
P = guppy.initParams(cfg, 'double');

fprintf('==== shapes ====\n');
fprintf('params           : %d\n', guppy.countParams(P));
fprintf('te %s  pe %s  Wqkv1 %s  W1 %s\n', mat2str(size(P.te)), mat2str(size(P.pe)), ...
        mat2str(size(P.Wqkv{1})), mat2str(size(P.W1{1})));

T = 8; B = 4;
rng(7);
ids = randi([0 60], T, B);
y   = randi([0 60], T, B);
y(1:2, :) = 0;                      % simulate right padding

[loss, C] = guppy.forward(P, cfg, ids, y, false, false, true);
G = guppy.backward(P, C, cfg);
fprintf('loss             : %.6f   (nValid %d, expected %d)\n', loss, C.nValid, sum(y(:) ~= 0));
fprintf('|G| global norm  : %.6f\n', guppy.gradNorm(G));

fprintf('\n==== finite differences (central, h=1e-5) ====\n');
h = 1e-5;
f = @(Pm) guppy.forward(Pm, cfg, ids, y, false);

targets = { ...
    'te',   [], 10,  2; ...
    'pe',   [],  3,  4; ...
    'Wqkv',  1,  5,  7; ...
    'Wo',    1,  2,  3; ...
    'W1',    2,  4,  6; ...
    'W2',    1,  8,  5; ...
    'g1',    1,  1,  9; ...
    'g2',    2,  1,  3; ...
    'gf',   [],  1,  6; ...
    'bf',   [],  1,  2 };

worst = 0;
for i = 1:size(targets, 1)
    nm = targets{i,1}; l = targets{i,2}; a = targets{i,3}; b = targets{i,4};
    Pp = bump(P, nm, l, a, b,  h);
    Pm = bump(P, nm, l, a, b, -h);
    lp = f(Pp); lm = f(Pm);
    num = (lp - lm) / (2*h);
    if isempty(l)
        ana = G.(nm)(a, b);
    else
        ana = G.(nm){l}(a, b);
    end
    rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana));
    worst = max(worst, rel);
    fprintf('%-5s l=%-2s (%2d,%2d)  num=%+12.6e  ana=%+12.6e  rel=%.2e\n', ...
        nm, num2str(l), a, b, num, ana, rel);
end
fprintf('\nworst relative error: %.3e   ->  %s\n', worst, ...
    string(worst < 1e-6));

% a loss sanity check: untrained model should sit near log(V)
fprintf('ln(V) = %.4f\n', log(cfg.V));
end

function P2 = bump(P, name, l, i, j, d)
P2 = P;
if isempty(l)
    P2.(name)(i, j) = P2.(name)(i, j) + d;
else
    P2.(name){l}(i, j) = P2.(name){l}(i, j) + d;
end
end

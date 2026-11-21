function smoke2()
%SMOKE2 Gradient verification with a directional derivative (high SNR) plus
%   a per-entry step-size sweep, then a full-config step-time measurement.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'tools'));

cfg = guppy.defaultConfig('L', 2, 'd', 32, 'H', 2, 'dff', 64, 'Tmax', 16, ...
                          'V', 2418, 'batch', 4, 'dropout', 0);
P = guppy.initParams(cfg, 'double');
T = 8; B = 4;
rng(7);
ids = randi([0 60], T, B);
y   = randi([0 60], T, B);  y(1:2, :) = 0;

f = @(Pm) guppy.forward(Pm, cfg, ids, y, false);
[loss, C] = guppy.forward(P, cfg, ids, y, false, false, true);
G = guppy.backward(P, C, cfg);
fprintf('loss %.6f   |G| %.6f\n\n', loss, guppy.gradNorm(G));

% ---- directional derivative: averages over all 94k entries -> high SNR ----
Vd = randnStruct(P);
fprintf('==== directional derivative ====\n');
ana = dotStruct(G, Vd);
for h = [1e-2 1e-3 1e-4]
    num = (f(addStruct(P, Vd, h)) - f(addStruct(P, Vd, -h))) / (2*h);
    fprintf('h=%-6g  FD=%+.10e  analytic=%+.10e  rel=%.3e\n', ...
        h, num, ana, abs(num-ana)/abs(ana));
end

% ---- per-entry step sweep -------------------------------------------------
fprintf('\n==== per-entry sweep ====\n');
targets = {'te',[],10,2; 'pe',[],3,4; 'Wqkv',1,5,7; 'Wo',1,2,3; ...
           'W1',2,4,6; 'W2',1,8,5; 'g1',1,1,9; 'g2',2,1,3; 'gf',[],1,6; 'bf',[],1,2};
for i = 1:size(targets,1)
    nm = targets{i,1}; l = targets{i,2}; a = targets{i,3}; b = targets{i,4};
    if isempty(l), g = G.(nm)(a,b); else, g = G.(nm){l}(a,b); end
    best = inf;
    for h = [1e-2 1e-3 1e-4 1e-5]
        num = (f(bump(P,nm,l,a,b,h)) - f(bump(P,nm,l,a,b,-h))) / (2*h);
        rel = abs(num-g)/max(1e-30, abs(num)+abs(g));
        best = min(best, rel);
    end
    fprintf('%-5s l=%-2s (%2d,%2d)  ana=%+11.4e  best rel=%.2e\n', nm, num2str(l), a, b, g, best);
end

% ---- full-config step timing --------------------------------------------
fprintf('\n==== full config timing ====\n');
cfg2 = guppy.defaultConfig('V', 2418);
P2 = guppy.initParams(cfg2);
st = struct();
Tt = 58; Bb = 32;
ids2 = randi([0 2417], Tt, Bb);
y2   = randi([0 2417], Tt, Bb);  y2(30:end,:) = 0;
fprintf('params %d\n', guppy.countParams(P2));

t0 = tic; [P2, st, inf1] = guppy.step(P2, st, ids2, y2, cfg2); t1 = toc(t0);
fprintf('step 1 (cold)  : %.2f s   loss %.4f  gnorm %.3f\n', t1, inf1.loss, inf1.gnorm);
ts = zeros(1,4);
for k = 1:4
    tk = tic; [P2, st, inf1] = guppy.step(P2, st, ids2, y2, cfg2); ts(k) = toc(tk);
end
fprintf('steps 2-5      : %s  mean %.2f s\n', mat2str(round(ts,2)), mean(ts));
fprintf('=> 10000 steps ~ %.1f h , 7000 steps ~ %.1f h\n', mean(ts)*10000/3600, mean(ts)*7000/3600);
fprintf('loss after 5   : %.4f  gnorm %.3f\n', inf1.loss, inf1.gnorm);
end

function P2 = bump(P, name, l, i, j, d)
P2 = P;
if isempty(l)
    P2.(name)(i,j) = P2.(name)(i,j) + d;
else
    P2.(name){l}(i,j) = P2.(name){l}(i,j) + d;
end
end

function S = randnStruct(P)
plain = {'te','pe','gf','bf'};
cellf = {'Wqkv','Wo','W1','W2','g1','b1','g2','b2'};
S = struct();
for i = 1:numel(plain)
    S.(plain{i}) = randn(size(P.(plain{i})));
end
for i = 1:numel(cellf)
    f = cellf{i}; S.(f) = cell(numel(P.(f)),1);
    for l = 1:numel(P.(f))
        S.(f){l} = randn(size(P.(f){l}));
    end
end
end

function P3 = addStruct(P, S, a)
P3 = P;
plain = {'te','pe','gf','bf'};
cellf = {'Wqkv','Wo','W1','W2','g1','b1','g2','b2'};
for i = 1:numel(plain)
    P3.(plain{i}) = P3.(plain{i}) + a*S.(plain{i});
end
for i = 1:numel(cellf)
    f = cellf{i};
    for l = 1:numel(P.(f))
        P3.(f){l} = P3.(f){l} + a*S.(f){l};
    end
end
end

function d = dotStruct(A, B)
plain = {'te','pe','gf','bf'};
cellf = {'Wqkv','Wo','W1','W2','g1','b1','g2','b2'};
d = 0;
for i = 1:numel(plain)
    d = d + sum(A.(plain{i})(:) .* B.(plain{i})(:));
end
for i = 1:numel(cellf)
    f = cellf{i};
    for l = 1:numel(A.(f))
        d = d + sum(A.(f){l}(:) .* B.(f){l}(:));
    end
end
end

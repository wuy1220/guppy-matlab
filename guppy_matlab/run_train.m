function run_train(cfg, tag)
%RUN_TRAIN Main training entry point.
%   run_train()                        full guppylm configuration, 10000 steps
%   run_train(cfg,'small')             anything else, tagged log files
%
%   Writes logs/<tag>.csv (one row per step) so progress can be tailed while
%   the training loop occupies the MATLAB session, plus a checkpoint every
%   cfg.saveEvery steps that can be resumed from.

if nargin < 1 || isempty(cfg), cfg = guppy.defaultConfig(); end
if nargin < 2 || isempty(tag), tag = 'run'; end

root = fileparts(mfilename('fullpath'));
logDir = fullfile(root, 'logs');
ckDir  = fullfile(root, 'checkpoints');
if ~exist(logDir, 'dir'), mkdir(logDir); end
if ~exist(ckDir, 'dir'),  mkdir(ckDir);  end

rng(cfg.seed);
D = data.loadData(cfg);
nTr = size(D.train.ids, 1);
nEv = size(D.eval.ids, 1);
fprintf('train %d / eval %d samples\n', nTr, nEv);

P  = guppy.initParams(cfg, cfg.dtype);
st = struct();
fprintf('params %d  (%.2f M)\n', guppy.countParams(P), guppy.countParams(P)/1e6);

csvPath = fullfile(logDir, [tag '.csv']);
fid = fopen(csvPath, 'w');
fprintf(fid, 'step,loss,gnorm,lr,sec\n');
fclose(fid);

perm = randperm(nTr);
ptr  = 1;
t0   = tic;
bestEval = inf;

nRun = cfg.steps;
if isfield(cfg, 'stopAt') && ~isempty(cfg.stopAt)
    nRun = min(nRun, cfg.stopAt);   % stop early but keep the full-length LR schedule
end

for k = 1:nRun
    if ptr + cfg.batch - 1 > nTr
        perm = randperm(nTr);
        ptr = 1;
    end
    idx = perm(ptr:ptr+cfg.batch-1);
    ptr = ptr + cfg.batch;

    [X, Y, T] = data.getBatch(D.train.ids, D.train.len, idx);
    [P, st, info] = guppy.step(P, st, X, Y, cfg);

    if mod(k, 20) == 1 || k == nRun
        fid = fopen(csvPath, 'a');
        fprintf(fid, '%d,%.6f,%.6f,%.8f,%.2f\n', k, info.loss, info.gnorm, info.lr, toc(t0));
        fclose(fid);
    end
    if mod(k, 100) == 0 || k == 1
        fprintf('step %5d/%d  loss %.4f  |g| %.3f  lr %.2e  T=%d  %.1f s  (%.3f s/step)\n', ...
            k, cfg.steps, info.loss, info.gnorm, info.lr, T, toc(t0), toc(t0)/k);
    end

    if mod(k, cfg.evalEvery) == 0 || k == nRun
        el = evalLoss(P, cfg, D, nEv);
        fprintf('  eval @%d : %.4f\n', k, el);
        if el < bestEval
            bestEval = el;
            rngState = rng;
            save(fullfile(ckDir, [tag '_best.mat']), 'P', 'st', 'cfg', 'k', 'el', ...
                 'perm', 'ptr', 'rngState', '-v7.3');
        end
    end
    if mod(k, cfg.saveEvery) == 0
        rngState = rng;
        save(fullfile(ckDir, sprintf('%s_%05d.mat', tag, k)), 'P', 'st', 'cfg', 'k', ...
             'perm', 'ptr', 'rngState', '-v7.3');
        fprintf('  checkpoint %s_%05d.mat\n', tag, k);
    end
end

rngState = rng;
save(fullfile(ckDir, [tag '_final.mat']), 'P', 'st', 'cfg', 'k', 'perm', 'ptr', 'rngState', '-v7.3');
fprintf('DONE %s in %.1f min, best eval %.4f\n', tag, toc(t0)/60, bestEval);
end

function el = evalLoss(P, cfg, D, nEv)
nb = min(20, floor(nEv / cfg.batch));
tot = 0;
for b = 1:nb
    idx = (b-1)*cfg.batch + (1:cfg.batch);
    [X, Y] = data.getBatch(D.eval.ids, D.eval.len, idx);
    l = guppy.forward(P, cfg, X, Y, false);
    tot = tot + double(l);
end
el = tot / nb;
end

function run_train_resume(ckpt, tag, nMax)
%RUN_TRAIN_RESUME Continue a run from a checkpoint, keeping the same LR schedule.
%   Current checkpoints store Adam moments, the RNG state and the batch cursor,
%   so resuming follows the same training stream as an uninterrupted run.
%
%   nMax lets a short dry run reuse exactly this code path, which is the only
%   way to be sure the long run will behave.
%
%   One checkpoint is 85 MB, so the 16-checkpoint schedule of a full run takes
%   1.4 GB -- keep checkpoints/ on a volume with room, or symlink it elsewhere.

if nargin < 1 || isempty(ckpt), ckpt = 'full_final'; end
if nargin < 2 || isempty(tag),  tag  = 'full3';      end

root = fileparts(mfilename('fullpath'));
S = load(fullfile(root, 'checkpoints', [ckpt '.mat']));
P = S.P; cfg = S.cfg;
k0 = 0;
if isfield(S, 'k'), k0 = S.k; end
if isfield(S, 'st')
    st = S.st;
    if ~isfield(st, 't'), st.t = k0; end
else
    % Older best checkpoints omitted Adam moments. Preserve their step number
    % while starting a fresh optimizer state instead of silently resetting LR.
    st = struct('m', guppy.likeParams(P), 'v', guppy.likeParams(P), 't', k0);
end
if nargin < 3 || isempty(nMax), nMax = cfg.steps; end

% 8 physical cores: 16 threads measured *slower* than 8 (bench4, section 3)
maxNumCompThreads(8);

logDir = fullfile(root, 'logs');
if ~exist(logDir, 'dir'), mkdir(logDir); end
ckDir = fullfile(root, 'checkpoints');
if ~exist(ckDir, 'dir'), mkdir(ckDir); end

D = data.loadData(cfg);
nTr = size(D.train.ids, 1);
nEv = size(D.eval.ids, 1);
if isfield(S, 'rngState'), rng(S.rngState); end
fprintf('resume from step %d, target %d, params %d, threads %d\n', ...
    st.t, nMax, guppy.countParams(P), maxNumCompThreads);
fprintf('checkpoints -> %s\n', ckDir);

csvPath = fullfile(logDir, [tag '.csv']);
fid = fopen(csvPath, 'w'); fprintf(fid, 'step,loss,gnorm,lr,sec\n'); fclose(fid);

if isfield(S, 'perm') && isfield(S, 'ptr')
    perm = S.perm; ptr = S.ptr;
else
    perm = randperm(nTr); ptr = 1;
end
t0 = tic; bestEval = inf;
kStart = st.t;

for k = (st.t + 1):nMax
    if ptr + cfg.batch - 1 > nTr
        perm = randperm(nTr); ptr = 1;
    end
    idx = perm(ptr:ptr+cfg.batch-1); ptr = ptr + cfg.batch;

    [X, Y, T] = data.getBatch(D.train.ids, D.train.len, idx);
    [P, st, info] = guppy.step(P, st, X, Y, cfg);

    if mod(k, 20) == 1 || k == nMax
        fid = fopen(csvPath, 'a');
        fprintf(fid, '%d,%.6f,%.6f,%.8f,%.2f\n', k, info.loss, info.gnorm, info.lr, toc(t0));
        fclose(fid);
    end
    if mod(k, 100) == 0
        fprintf('step %5d/%d  loss %.4f  |g| %.3f  lr %.2e  T=%d  %5.1f min  %5.2f s/step\n', ...
            k, nMax, info.loss, info.gnorm, info.lr, T, toc(t0)/60, toc(t0)/(k - kStart));
    end
    if mod(k, cfg.evalEvery) == 0
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
    end
end
rngState = rng;
save(fullfile(ckDir, [tag '_final.mat']), 'P', 'st', 'cfg', 'k', 'perm', 'ptr', 'rngState', '-v7.3');
fprintf('DONE resume %s at %d (best eval %.4f), %.1f min\n', tag, nMax, bestEval, toc(t0)/60);
end

function el = evalLoss(P, cfg, D, nEv)
nb = min(20, floor(nEv / cfg.batch));
tot = 0;
for b = 1:nb
    idx = (b-1)*cfg.batch + (1:cfg.batch);
    [X, Y] = data.getBatch(D.eval.ids, D.eval.len, idx);
    tot = tot + double(guppy.forward(P, cfg, X, Y, false));
end
el = tot / nb;
end

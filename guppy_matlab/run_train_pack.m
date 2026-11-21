function run_train_pack(tag, nSteps, dropout)
%RUN_TRAIN_PACK Train on fixed 128-token windows sliced from the packed stream.
%
%   The comparison this exists for: the per-document trainer spends 31% of every
%   position on padding, so 10 000 steps only show the model 8.8 M real tokens.
%   A window batch of 32 x 127 shows it 4 064 real tokens per step, i.e. 4.6x
%   more per step at ~2x the step cost.
%
%   Everything else -- init scale, LR shape, weight decay, clip, dropout, seed --
%   is identical to the run in section 5 so that only the token stream changes.

if nargin < 1 || isempty(tag),     tag = 'pack';   end
if nargin < 2 || isempty(nSteps),  nSteps = 2158;  end
if nargin < 3 || isempty(dropout), dropout = 0.1;  end

cfg = guppy.defaultConfig('steps', nSteps, 'dropout', dropout);
cfg.warmup = max(20, round(0.02 * nSteps));      % same 2% warmup as the baseline

maxNumCompThreads(8);
rng(cfg.seed);
P = guppy.initParams(cfg, cfg.dtype);
st = struct(); st.t = 0;

D = data.loadData(cfg);
W = data.packData(D, cfg.Tmax - 1);
nEv = size(D.eval.ids, 1);

root = fileparts(mfilename('fullpath'));
logDir = fullfile(root, 'logs');
if ~exist(logDir, 'dir'), mkdir(logDir); end
ckDir = fullfile(root, 'checkpoints');
if ~exist(ckDir, 'dir'), mkdir(ckDir); end

fprintf('packed: %d windows of T=%d, %d tokens, %d steps, batch %d, dropout %.2f\n', ...
    W.n, W.T, W.ntok, nSteps, cfg.batch, cfg.dropout);
fprintf('real tokens per step %d (per-document path: ~%d)\n', cfg.batch * W.T, ...
    cfg.batch * round(mean(D.train.len)));

csvPath = fullfile(logDir, [tag '.csv']);
fid = fopen(csvPath, 'w'); fprintf(fid, 'step,loss,gnorm,lr,sec,tok\n'); fclose(fid);

ord = randperm(W.n); ptr = 1;
t0 = tic; bestEval = inf; dtHist = zeros(1, 100); nh = 0;

for k = 1:nSteps
    if ptr + cfg.batch - 1 > numel(ord)
        ord = randperm(W.n); ptr = 1;
    end
    idx = ord(ptr:ptr+cfg.batch-1); ptr = ptr + cfg.batch;

    X = W.x(:, idx); Y = W.y(:, idx);
    tt = tic;
    [P, st, info] = guppy.step(P, st, X, Y, cfg);
    dtHist(mod(k-1, 100) + 1) = toc(tt);
    sp = mean(dtHist(1:min(k, 100)));

    if mod(k, 20) == 1 || k == nSteps
        fid = fopen(csvPath, 'a');
        fprintf(fid, '%d,%.6f,%.6f,%.8f,%.2f,%d\n', k, info.loss, info.gnorm, info.lr, toc(t0), k * cfg.batch * W.T);
        fclose(fid);
    end
    if mod(k, 100) == 0
        fprintf('step %5d/%d  loss %.4f  |g| %.3f  lr %.2e  %.1f min  %.3f s/step  %.0f tok/s\n', ...
            k, nSteps, info.loss, info.gnorm, info.lr, toc(t0)/60, sp, ...
            cfg.batch * W.T / sp);
    end
    if mod(k, 200) == 0
        el = evalLoss(P, cfg, D, nEv);
        fprintf('  eval @%d (%.2f M tokens) : %.4f\n', k, k * cfg.batch * W.T / 1e6, el);
        if el < bestEval
            bestEval = el;
            rngState = rng;
            save(fullfile(ckDir, [tag '_best.mat']), 'P', 'st', 'cfg', 'k', 'el', ...
                 'ord', 'ptr', 'rngState', '-v7.3');
        end
    end
end
rngState = rng;
save(fullfile(ckDir, [tag '_final.mat']), 'P', 'st', 'cfg', 'k', 'ord', 'ptr', 'rngState', '-v7.3');
fprintf('DONE %s at %d steps (%.2f M tokens), best eval %.4f, %.1f min\n', ...
    tag, nSteps, nSteps * cfg.batch * W.T / 1e6, bestEval, toc(t0)/60);
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

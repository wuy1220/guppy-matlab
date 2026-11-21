% R_BUCK  Random batches vs length-bucketed batches, same budget, same seed.
%
% T = max(len)-1 within a batch, so a random batch of 32 draws is almost always
% stretched to T=58 by its longest member, while the average member only has
% 27.4 tokens: half of every position in the batch is padding that the loss
% mask throws away -- but that the forward and backward still pay for.
%
% Bucketing sorts by length once, keeps the sorted order inside a shuffle
% buffer of 20 batches, and shuffles only the buffer order. Every batch is then
% nearly length-homogeneous and T collapses to ~28.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
cfg = guppy.defaultConfig();
cfg.steps = 300;
D = data.loadData(cfg);
nTr = size(D.train.ids, 1);
B = cfg.batch;

% --- the two batch orders ----------------------------------------------------
rand('seed', 42); randn('seed', 42);
permRand = randperm(nTr);

[~, byLen] = sort(D.train.len);
buf = 20 * B;
nBuf = floor(nTr / buf);
blocks = cell(nBuf, 1);
for b = 1:nBuf
    blocks{b} = byLen((b-1)*buf + (1:buf));   % ascending inside the buffer
end
rand('seed', 42);
bord = randperm(nBuf);                        % shuffle the buffers, not their contents
order = vertcat(blocks{bord});
order = order(:);
nBufUsed = nBuf;

modes = {'random', 'bucket'};
res = struct();
for m = 1:2
    if m == 1
        seq = permRand;
    else
        seq = order;
    end

    rand('seed', 7); randn('seed', 7);      % identical init and dropout stream
    P = guppy.initParams(cfg, cfg.dtype);
    st = struct(); st.t = 0;
    ptr = 1; dt = zeros(cfg.steps, 1); Ty = zeros(cfg.steps, 1); tok = zeros(cfg.steps, 1);

    for k = 1:cfg.steps
        if ptr + B - 1 > numel(seq), ptr = 1; end
        idx = seq(ptr:ptr+B-1); ptr = ptr + B;
        [X, Y, T] = data.getBatch(D.train.ids, D.train.len, idx);
        tt = tic;
        [P, st, info] = guppy.step(P, st, X, Y, cfg);
        dt(k) = toc(tt);
        Ty(k) = T; tok(k) = sum(D.train.len(idx));
    end

    fprintf('%-7s  s/step mean %.4f  median %.4f  |  T mean %.1f  |  real tok/step %.0f  |  tok/s %.0f  |  loss@300 %.4f\n', ...
        modes{m}, mean(dt), median(dt), mean(Ty), mean(tok), mean(tok) / mean(dt), info.loss);
    res(m).mode = modes{m}; res(m).s = mean(dt); res(m).T = mean(Ty);
    res(m).tok = mean(tok); res(m).loss = double(info.loss); res(m).dt = dt;
    res(m).Ty = Ty; res(m).tokv = tok;
end

fprintf('\nspeedup per step      : %.2fx\n', res(1).s / res(2).s);
fprintf('speedup per real token: %.2fx\n', (res(2).tok / res(2).s) / (res(1).tok / res(1).s));
fprintf('padding share  random : %.0f%%\n', 100 * (1 - res(1).tok / (B * res(1).T)));
fprintf('padding share  bucket : %.0f%%\n', 100 * (1 - res(2).tok / (B * res(2).T)));

fid = fopen(fullfile(cfg.dataDir, 'buck_probe.csv'), 'w');
fprintf(fid, 'mode,step,sec,T,tokens\n');
for m = 1:2
    for k = 1:cfg.steps
        fprintf(fid, '%s,%d,%.5f,%d,%d\n', res(m).mode, k, res(m).dt(k), res(m).Ty(k), res(m).tokv(k));
    end
end
fclose(fid);
fprintf('wrote data/buck_probe.csv\n');

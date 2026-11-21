function r_thr2()
%R_THR2 What is the session actually running at right now, and does
%   'automatic' restore full speed?  r_thread left the pool at 1.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
S = load(fullfile(root, 'checkpoints', 'full3_10000.mat'));
P = S.P; cfg = S.cfg;
T = 40; B = cfg.batch;
ids = reshape(int32(randi(cfg.V - 1, T * B, 1)), T, B);
y   = reshape(int32(randi(cfg.V - 1, T * B, 1)), T, B);

function t = onestep(P, cfg, ids, y)
    st = struct();
    for r = 1:3, [P, st, ~] = guppy.step(P, st, ids, y, cfg); end
    n = 4; dt = zeros(n, 1);
    for r = 1:n
        tic; [P, st, ~] = guppy.step(P, st, ids, y, cfg); dt(r) = toc;
    end
    t = min(dt);
end

fprintf('now: maxNumCompThreads = %d, numcores = %d\n', maxNumCompThreads, feature('numcores'));
fprintf('as-is            %7.1f ms/step\n', 1e3 * onestep(P, cfg, ids, y));

maxNumCompThreads('automatic');
fprintf('after automatic: %d\n', maxNumCompThreads);
fprintf('automatic        %7.1f ms/step\n', 1e3 * onestep(P, cfg, ids, y));

for k = [6 8]
    maxNumCompThreads(k);
    fprintf('explicit %d       %7.1f ms/step\n', k, 1e3 * onestep(P, cfg, ids, y));
end

maxNumCompThreads('automatic');
fprintf('final: maxNumCompThreads = %d\n', maxNumCompThreads);
end
